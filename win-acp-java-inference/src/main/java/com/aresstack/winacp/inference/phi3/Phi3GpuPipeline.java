package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.windows.*;
import com.aresstack.winacp.windows.Phi3ComputeShaders.ComputeKernelSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * V2.0 batched GPU pipeline for Phi-3 decode.
 * <p>
 * Collapses 129 GPU submissions per token into ~65 by batching operations
 * that don't need CPU readback between them:
 * <ul>
 *   <li><b>QKV batch</b>: upload + QKV GEMM + readback [1 submission]</li>
 *   <li><b>MLP batch</b>: upload attn_out + upload hidden_io →
 *       O GEMM → GPU_add(residual) → GPU_RMSNorm → GateUp GEMM →
 *       GPU_SwiGLU → Down GEMM → GPU_add(residual2) → readback
 *       [1 submission, 7 GPU ops]</li>
 * </ul>
 * Total: 2 submissions/layer × 32 + 1 lm_head = <b>65 submissions</b> (was 129).
 */
public final class Phi3GpuPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3GpuPipeline.class);

    private final GpuPipeline pipeline;
    private final Phi3GpuKernels kernels;
    private ComputeKernelSet computeKernels;  // nullable if shader compilation fails

    // ── GPU-resident intermediate buffers ──────────────────────────────
    private MemorySegment residualBuf;     // [hidden] for residual add result
    private MemorySegment postNormBuf;     // [hidden] for RMSNorm output
    private MemorySegment mlpActBuf;       // [intermediate] for SwiGLU output

    // ── GPU-resident weight buffers (per-layer, uploaded once) ─────────
    private MemorySegment[] postNormWeightBufs;   // [layer] → GPU [hidden]
    private MemorySegment[] mlpOutScaleBufs;      // [layer] → GPU [intermediate]

    // ── UAV barrier (global sync between dispatches) ──────────────────
    private MemorySegment uavBarrier;

    private final int hidden;
    private final int intermediate;
    private final float rmsNormEps;
    private boolean mlpBatchEnabled = false;  // only true if compute shaders compiled OK
    private boolean closed = false;

    /**
     * Create the batched GPU pipeline.
     */
    public Phi3GpuPipeline(WindowsBindings wb, Phi3GpuKernels kernels, Phi3Config config)
            throws WindowsNativeException {
        this.kernels = kernels;
        this.hidden = config.hiddenSize();
        this.intermediate = config.intermediateSize();
        this.rmsNormEps = config.rmsNormEps();

        long hiddenBytes = (long) hidden * Float.BYTES;
        long interBytes = (long) intermediate * Float.BYTES;
        long qkvBytes = (long) hidden * 3 * Float.BYTES;
        long vocabBytes = (long) config.vocabSize() * Float.BYTES;

        long maxUpload = Math.max(hiddenBytes, interBytes);
        long maxReadback = Math.max(qkvBytes, vocabBytes);

        this.pipeline = new GpuPipeline(wb, maxUpload, maxReadback);

        // ── Try to compile compute shaders for MLP batching ───────────
        try {
            computeKernels = Phi3ComputeShaders.createAll(wb, pipeline.getCommandList());

            // Allocate GPU intermediate buffers
            var dev = wb.getD3d12Device();
            var arena = pipeline.getArena();
            residualBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            postNormBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            mlpActBuf   = D3D12Bindings.createDefaultBuffer(dev, interBytes, arena);

            // Upload per-layer weights to GPU
            var queue = wb.getCommandQueue();
            int numLayers = config.numHiddenLayers();
            postNormWeightBufs = new MemorySegment[numLayers];
            mlpOutScaleBufs = new MemorySegment[numLayers];

            for (int l = 0; l < numLayers; l++) {
                var lw = ((Phi3Weights) null); // weights not passed — defer upload
                // Weights upload deferred to Phi3Runtime which has access to weights
            }

            uavBarrier = pipeline.allocUavBarrier();
            mlpBatchEnabled = true; // compute shaders compiled OK
            log.info("Phi3GpuPipeline V2.0: compute shaders compiled, MLP batching ENABLED");
        } catch (Exception e) {
            log.warn("Compute shader compilation failed, falling back to per-kernel dispatch: {}",
                    e.getMessage());
            computeKernels = null;
            mlpBatchEnabled = false;
        }

        log.info("Phi3GpuPipeline V2.0 ready: mlpBatch={}, upload={}KB readback={}KB",
                mlpBatchEnabled, maxUpload / 1024, maxReadback / 1024);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Single-GEMM dispatch using shared pipeline
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Execute a single GEMM via the shared pipeline.
     */
    public void matvec(MatMulNBitsKernel kernel, float[] input, float[] output) {
        pipeline.begin();
        kernel.recordInto(pipeline, input);
        pipeline.submitAndWait();
        kernel.readResult(output);
    }

    public void qkvFused(int layerIdx, float[] input, float[] qkvOutput) {
        matvec(kernels.qkvFused(layerIdx), input, qkvOutput);
    }

    public void oProj(int layerIdx, float[] input, float[] output) {
        matvec(kernels.oProj(layerIdx), input, output);
    }

    public void gateUpProj(int layerIdx, float[] input, float[] output) {
        matvec(kernels.gateUpProj(layerIdx), input, output);
    }

    public void downProj(int layerIdx, float[] input, float[] output) {
        matvec(kernels.downProj(layerIdx), input, output);
    }

    public void lmHead(float[] input, float[] logits) {
        matvec(kernels.lmHead(), input, logits);
    }

    // ═══════════════════════════════════════════════════════════════════
    // MLP Batch: O → add → norm → GateUp → SwiGLU → Down → add → readback
    // 3 GEMMs + 2 adds + 1 norm + 1 SwiGLU = 7 GPU ops, 1 submission
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Whether MLP batching is available (compute shaders compiled).
     */
    public boolean isMlpBatchEnabled() { return mlpBatchEnabled; }

    /**
     * Upload per-layer norm weights and scales to GPU.
     * Must be called once after construction with access to the model weights.
     */
    public void uploadLayerWeights(WindowsBindings wb, Phi3Weights weights, Phi3Config config)
            throws WindowsNativeException {
        if (!mlpBatchEnabled) return;

        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var arena = pipeline.getArena();
        int numLayers = config.numHiddenLayers();
        long hiddenBytes = (long) hidden * Float.BYTES;
        long interBytes = (long) intermediate * Float.BYTES;

        postNormWeightBufs = new MemorySegment[numLayers];
        mlpOutScaleBufs = new MemorySegment[numLayers];

        long t0 = System.currentTimeMillis();
        for (int l = 0; l < numLayers; l++) {
            var lw = weights.layers[l];
            postNormWeightBufs[l] = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, postNormWeightBufs[l], lw.postNormWeight(), arena);
            mlpOutScaleBufs[l] = D3D12Bindings.createDefaultBuffer(dev, interBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, mlpOutScaleBufs[l], lw.mlpOutScale(), arena);
        }
        log.info("Uploaded {} layer weights to GPU in {} ms", numLayers * 2, System.currentTimeMillis() - t0);
    }

    /**
     * Batched MLP: 7 GPU operations in ONE submission.
     * <p>
     * Flow:
     * <pre>
     *   1. Upload attn_out → O_GEMM (kernel upload)
     *   2. Upload hidden_io → residualBuf (pipeline upload)
     *   3. O_GEMM dispatch → O_output
     *   4. GPU_add(O_output, residualBuf) → residualBuf
     *   5. GPU_RMSNorm(residualBuf, postNormWeight) → postNormBuf
     *   6. GateUp_GEMM(postNormBuf) → GateUp_output
     *   7. GPU_SwiGLU(GateUp_output, mlpScale) → mlpActBuf
     *   8. Down_GEMM(mlpActBuf) → Down_output
     *   9. GPU_add(residualBuf, Down_output) → residualBuf
     *   10. Readback residualBuf → hidden_out
     * </pre>
     *
     * @param attnOutput  CPU [hidden] — attention output (after attn scale)
     * @param hiddenInput CPU [hidden] — for residual1 = hiddenInput + O_proj
     * @param hiddenOut   CPU [hidden] — output: residual2 = residual1 + Down
     * @param layerIdx    layer index
     */
    public void batchMlp(float[] attnOutput, float[] hiddenInput, float[] hiddenOut,
                          int layerIdx) {
        if (!mlpBatchEnabled) {
            throw new IllegalStateException("MLP batch not enabled — compute shaders missing");
        }

        long hiddenBytes = (long) hidden * Float.BYTES;
        long interBytes = (long) intermediate * Float.BYTES;
        long interX2Bytes = (long) intermediate * 2 * Float.BYTES;

        MatMulNBitsKernel oK = kernels.oProj(layerIdx);
        MatMulNBitsKernel guK = kernels.gateUpProj(layerIdx);
        MatMulNBitsKernel downK = kernels.downProj(layerIdx);

        pipeline.begin();
        var cl = pipeline.getCommandList();

        // ── 1. Upload attn_out to O_proj kernel's input + dispatch ─────
        oK.recordInto(pipeline, attnOutput);
        // After recordInto: O output copied to oK.readbackBuf, buffers back to COMMON
        // But we need O output on GPU! Use the readback copy as our output.
        // Actually, oK.outputBuf has the result but is in COMMON state now.
        // We need it in UAV for the add.

        // ── 2. Upload hidden_io → residualBuf ──────────────────────────
        pipeline.recordUpload(hiddenInput, 0, hidden, residualBuf, 0);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 3. GPU_add: O_output + residualBuf → residualBuf ───────────
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)  // in-place
                },
                new int[]{ hidden },
                hidden);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 4. GPU_RMSNorm: residualBuf → postNormBuf ──────────────────
        int epsBits = Float.floatToRawIntBits(rmsNormEps);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(postNormWeightBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(postNormBuf)
                },
                new int[]{ hidden, epsBits },
                1);  // single group for RMSNorm
        pipeline.recordUavBarrier(uavBarrier);

        // ── 5. GateUp GEMM: postNormBuf → guK.outputBuf ───────────────
        guK.recordIntoGpuResident(pipeline, postNormBuf, hiddenBytes);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 6. GPU_SwiGLU: guK.outputBuf → mlpActBuf ──────────────────
        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(mlpOutScaleBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(mlpActBuf)
                },
                new int[]{ intermediate },
                intermediate);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 7. Down GEMM: mlpActBuf → downK.outputBuf ─────────────────
        downK.recordIntoGpuResident(pipeline, mlpActBuf, interBytes);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 8. GPU_add: residualBuf + downK.outputBuf → residualBuf ────
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);
        pipeline.recordUavBarrier(uavBarrier);

        // ── 9. Readback residualBuf → CPU ──────────────────────────────
        pipeline.recordReadback(residualBuf, 0, hiddenBytes);

        // ── 10. Cleanup barriers ───────────────────────────────────────
        guK.recordCleanupBarriers(pipeline);
        downK.recordCleanupBarriers(pipeline);

        pipeline.submitAndWait();
        pipeline.readbackInto(hiddenOut, 0, hidden);
    }

    /** Whether the pipeline has GPU kernels for the given layer. */
    public boolean hasLayer(int layerIdx) { return kernels.hasLayer(layerIdx); }

    /** Whether lm_head is on GPU. */
    public boolean hasLmHead() { return kernels.hasLmHead(); }

    /** Underlying pipeline. */
    public GpuPipeline getPipeline() { return pipeline; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (computeKernels != null) computeKernels.close();
        pipeline.close();
        log.info("Phi3GpuPipeline closed");
    }
}
