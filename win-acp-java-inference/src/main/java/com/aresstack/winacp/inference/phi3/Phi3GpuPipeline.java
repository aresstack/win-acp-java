package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.windows.*;
import com.aresstack.winacp.windows.Phi3ComputeShaders.ComputeKernelSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;

/**
 * V2.0 batched GPU pipeline for Phi-3 decode.
 * <p>
 * Collapses 129 GPU submissions per token into ~65 by batching operations
 * that don't need CPU readback between them:
 * <ul>
 *   <li><b>QKV batch</b>: upload + QKV GEMM + readback [1 submission]</li>
 *   <li><b>MLP batch</b>: O GEMM → GPU_add(residual) → GPU_RMSNorm →
 *       GateUp GEMM → GPU_SwiGLU → Down GEMM → GPU_add(residual2) → readback
 *       [1 submission, 7 GPU ops]</li>
 * </ul>
 * Total: 2 submissions/layer × 32 + 1 lm_head = <b>65 submissions</b> (was 129).
 * <p>
 * <b>Resource State Strategy</b>: All GPU buffers decay to COMMON state after
 * each {@link GpuPipeline#submitAndWait()} (D3D12 buffer decay rule). Within a
 * batch, implicit promotion from COMMON → {COPY_DEST, UAV} is used, with explicit
 * transitions only where required (COPY_DEST → UAV for copy targets, UAV → COPY_SOURCE
 * for readback). No cleanup barriers needed at batch end — decay handles it.
 * <p>
 * <b>Zero-copy chaining</b>: Compute shaders (RMSNorm, SwiGLU) write their output
 * directly into the next GEMM kernel's input buffer, eliminating intermediate copies.
 */
public final class Phi3GpuPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3GpuPipeline.class);

    private final GpuPipeline pipeline;
    private final Phi3GpuKernels kernels;
    private ComputeKernelSet computeKernels;  // nullable if shader compilation fails

    // ── GPU-resident intermediate buffer ───────────────────────────────
    private MemorySegment residualBuf;     // [hidden] for residual add / running state

    // ── GPU-resident weight buffers (per-layer, uploaded once) ─────────
    private MemorySegment[] postNormWeightBufs;   // [layer] → GPU [hidden]
    private MemorySegment[] mlpOutScaleBufs;      // [layer] → GPU [intermediate]

    // ── Pre-allocated barriers ────────────────────────────────────────
    private MemorySegment uavBarrier;
    private MemorySegment barrierResidualCopyDestToUav;
    private MemorySegment barrierResidualUavToCopySource;

    private final int hidden;
    private final int intermediate;
    private final float rmsNormEps;
    private boolean mlpBatchEnabled = false;
    private boolean weightsUploaded = false;
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
        long qkvBytes = (long) hidden * 3 * Float.BYTES;
        long vocabBytes = (long) config.vocabSize() * Float.BYTES;

        long maxUpload = hiddenBytes;  // largest CPU→GPU upload per batch
        long maxReadback = Math.max(qkvBytes, Math.max(vocabBytes, hiddenBytes));

        this.pipeline = new GpuPipeline(wb, maxUpload, maxReadback);

        // ── Try to compile compute shaders for MLP batching ───────────
        try {
            computeKernels = Phi3ComputeShaders.createAll(wb, pipeline.getCommandList());

            // Allocate GPU residual buffer (only intermediate buffer needed)
            var dev = wb.getD3d12Device();
            var arena = pipeline.getArena();
            residualBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);

            // Pre-allocate barriers for residualBuf
            uavBarrier = pipeline.allocUavBarrier();
            barrierResidualCopyDestToUav = pipeline.allocTransitionBarrier(residualBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrierResidualUavToCopySource = pipeline.allocTransitionBarrier(residualBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);

            mlpBatchEnabled = true;
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

    /** Execute a single GEMM via the shared pipeline. */
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
    // MLP Batch: 7 GPU ops, 1 submission
    // ═══════════════════════════════════════════════════════════════════

    /** Whether MLP batching is available (compute shaders compiled + weights uploaded). */
    public boolean isMlpBatchEnabled() { return mlpBatchEnabled && weightsUploaded; }

    /**
     * Upload per-layer norm weights and MLP scales to GPU.
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
        weightsUploaded = true;
        log.info("Uploaded {} layer weights to GPU in {} ms", numLayers * 2,
                System.currentTimeMillis() - t0);
    }

    /**
     * Batched MLP: 7 GPU operations in ONE submission.
     * <p>
     * <b>Resource State Flow</b> (all buffers start in COMMON due to decay):
     * <pre>
     *   1. CPU→Upload: attnOutput → oK.uploadBuf → copy → oK.inputBuf  [COMMON→COPY_DEST]
     *   2. CPU→Upload: hiddenInput → pipeline.uploadBuf → copy → residualBuf  [COMMON→COPY_DEST]
     *   3. Barrier: oK.inputBuf COPY_DEST→UAV, residualBuf COPY_DEST→UAV
     *   4. DML O_proj: oK.inputBuf(UAV) → oK.outputBuf(COMMON→UAV)
     *   5. UAV barrier
     *   6. Compute ADD: oK.outputBuf + residualBuf → residualBuf  (in-place)
     *   7. UAV barrier
     *   8. Compute RMSNorm: residualBuf → guK.inputBuf(COMMON→UAV)  [direct write!]
     *   9. UAV barrier
     *  10. DML GateUp: guK.inputBuf(UAV) → guK.outputBuf(COMMON→UAV)
     *  11. UAV barrier
     *  12. Compute SwiGLU: guK.outputBuf → downK.inputBuf(COMMON→UAV)  [direct write!]
     *  13. UAV barrier
     *  14. DML Down: downK.inputBuf(UAV) → downK.outputBuf(COMMON→UAV)
     *  15. UAV barrier
     *  16. Compute ADD: residualBuf + downK.outputBuf → residualBuf
     *  17. Barrier: residualBuf UAV→COPY_SOURCE
     *  18. Copy: residualBuf → pipeline.readbackBuf
     *  19. Submit + Wait  (all buffers decay to COMMON automatically)
     *  20. Readback: pipeline.readbackBuf → hiddenOut
     * </pre>
     *
     * @param attnOutput  CPU [hidden] — attention output (after attn scale)
     * @param hiddenInput CPU [hidden] — for residual1 = hiddenInput + O_proj
     * @param hiddenOut   CPU [hidden] — output: residual2 = residual1 + Down
     * @param layerIdx    layer index
     */
    public void batchMlp(float[] attnOutput, float[] hiddenInput, float[] hiddenOut,
                          int layerIdx) {
        if (!isMlpBatchEnabled()) {
            throw new IllegalStateException("MLP batch not enabled");
        }

        long hiddenBytes = (long) hidden * Float.BYTES;

        MatMulNBitsKernel oK = kernels.oProj(layerIdx);
        MatMulNBitsKernel guK = kernels.gateUpProj(layerIdx);
        MatMulNBitsKernel downK = kernels.downProj(layerIdx);

        pipeline.begin();
        var cl = pipeline.getCommandList();

        // ── 1+2. Upload attnOutput → oK + hiddenInput → residualBuf ──
        // Uses separate upload buffers (oK.uploadBuf vs pipeline.uploadBuf)
        oK.recordBatchFromCpu(pipeline, attnOutput);
        // oK.inputBuf: COPY_DEST→UAV (done inside recordBatchFromCpu)
        // oK.outputBuf: COMMON→UAV (promoted by DML dispatch)

        // Upload hiddenInput → residualBuf (for residual add)
        pipeline.recordUpload(hiddenInput, 0, hidden, residualBuf, 0);
        // residualBuf: COMMON → promoted to COPY_DEST

        // ── 3. Barrier: residualBuf COPY_DEST → UAV ──────────────────
        pipeline.recordBarrier(barrierResidualCopyDestToUav);

        // ── 4. (O_proj DML already dispatched by recordBatchFromCpu) ──

        // ── 5. UAV barrier (sync DML O_proj write → compute add read) ─
        pipeline.recordUavBarrier(uavBarrier);

        // ── 6. Compute ADD: oK.outputBuf + residualBuf → residualBuf ──
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        // ── 7. UAV barrier (sync add write → RMSNorm read) ───────────
        pipeline.recordUavBarrier(uavBarrier);

        // ── 8. Compute RMSNorm: residualBuf → guK.inputBuf (direct!) ─
        // Writes directly into GateUp kernel's input buffer → zero-copy chain
        int epsBits = Float.floatToRawIntBits(rmsNormEps);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(postNormWeightBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(guK.getInputBuf())
                },
                new int[]{ hidden, epsBits },
                1);  // single thread group for RMSNorm

        // ── 9. UAV barrier (sync RMSNorm write → DML GateUp read) ────
        pipeline.recordUavBarrier(uavBarrier);

        // ── 10. DML GateUp: guK.inputBuf(UAV) → guK.outputBuf ────────
        guK.recordBatchDispatchOnly(pipeline);

        // ── 11. UAV barrier (sync GateUp write → SwiGLU read) ────────
        pipeline.recordUavBarrier(uavBarrier);

        // ── 12. Compute SwiGLU: guK.outputBuf → downK.inputBuf (direct!)
        // Writes directly into Down kernel's input buffer → zero-copy chain
        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(mlpOutScaleBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(downK.getInputBuf())
                },
                new int[]{ intermediate },
                intermediate);

        // ── 13. UAV barrier (sync SwiGLU write → DML Down read) ──────
        pipeline.recordUavBarrier(uavBarrier);

        // ── 14. DML Down: downK.inputBuf(UAV) → downK.outputBuf ──────
        downK.recordBatchDispatchOnly(pipeline);

        // ── 15. UAV barrier (sync Down write → add read) ─────────────
        pipeline.recordUavBarrier(uavBarrier);

        // ── 16. Compute ADD: residualBuf + downK.outputBuf → residualBuf
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        // ── 17. Barrier: residualBuf UAV → COPY_SOURCE (for readback) ─
        pipeline.recordBarrier(barrierResidualUavToCopySource);

        // ── 18. Copy residualBuf → readbackBuf ───────────────────────
        pipeline.recordReadback(residualBuf, 0, hiddenBytes);

        // ── 19. Submit + Wait (buffers decay to COMMON automatically) ─
        pipeline.submitAndWait();

        // ── 20. Readback → CPU ───────────────────────────────────────
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
