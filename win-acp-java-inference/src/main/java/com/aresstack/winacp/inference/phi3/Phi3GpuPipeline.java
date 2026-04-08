package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.windows.*;
import com.aresstack.winacp.windows.Phi3ComputeShaders.ComputeKernelSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;

/**
 * V2.0/V3.0 batched GPU pipeline for Phi-3 decode.
 * <p>
 * <b>V2.0</b>: Collapses 129 → 65 submissions (QKV batch + MLP batch per layer).
 * <p>
 * <b>V3.0 full GPU decode</b>: ALL 32 layers + lm_head in <b>ONE submission</b>.
 * <ul>
 *   <li>GPU-resident KV cache (no CPU roundtrip for K/V)</li>
 *   <li>GPU RoPE, attention (score → softmax → V-sum), norms, activations</li>
 *   <li>Hidden states stay on GPU between layers</li>
 *   <li>Only ONE fence wait per token decode</li>
 *   <li>Only readback = final logits [vocabSize]</li>
 * </ul>
 * <p>
 * <b>Submission count</b>: 129 (V1.x) → 65 (V2.0) → <b>1</b> (V3.0)
 */
public final class Phi3GpuPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3GpuPipeline.class);

    private final GpuPipeline pipeline;
    private final Phi3GpuKernels kernels;
    private ComputeKernelSet computeKernels;  // nullable if shader compilation fails

    // ── V2.0 GPU-resident intermediate buffer ─────────────────────────
    private MemorySegment residualBuf;

    // ── V2.0 GPU-resident weight buffers (per-layer, uploaded once) ───
    private MemorySegment[] postNormWeightBufs;
    private MemorySegment[] mlpOutScaleBufs;

    // ── V2.0 Pre-allocated barriers ──────────────────────────────────
    private MemorySegment uavBarrier;
    private MemorySegment barrierResidualCopyDestToUav;
    private MemorySegment barrierResidualUavToCopySource;

    // ── V3.0 GPU-resident intermediate buffers ────────────────────────
    private MemorySegment hiddenBufA;       // [hidden] — double-buffer A
    private MemorySegment hiddenBufB;       // [hidden] — double-buffer B
    private MemorySegment attnOutBuf;       // [hidden] — attention output
    private MemorySegment scoresBuf;        // [numHeads * maxPos] — attention scores

    // ── V3.0 GPU KV cache (per-layer) ─────────────────────────────────
    private MemorySegment[] gpuKCacheBufs;  // [layer] → GPU [maxGpuPos * hidden]
    private MemorySegment[] gpuVCacheBufs;  // [layer] → GPU [maxGpuPos * hidden]
    private int maxGpuPos;                  // max positions in GPU KV cache

    // ── V3.0 GPU weight buffers (per-layer, uploaded once) ────────────
    private MemorySegment[] inputNormWeightBufs;  // [layer] → GPU [hidden]
    private MemorySegment[] attnOutScaleBufs;     // [layer] → GPU [hidden]
    private MemorySegment cosCacheBuf;            // GPU [maxPos * halfDim]
    private MemorySegment sinCacheBuf;            // GPU [maxPos * halfDim]
    private MemorySegment finalNormWeightBuf;     // GPU [hidden]

    // ── V3.0 Pre-allocated barrier for hiddenBufA upload ──────────────
    private MemorySegment barrierHiddenACopyDestToUav;
    private MemorySegment barrierHiddenAUavToCommon;

    // ── V3.0 model config ─────────────────────────────────────────────
    private int numHeads;
    private int numKvHeads;
    private int headDim;
    private int halfDim;
    private int numLayers;
    private int vocabSize;

    private final int hidden;
    private final int intermediate;
    private final float rmsNormEps;
    private boolean mlpBatchEnabled = false;
    private boolean fullGpuEnabled = false;
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
        this.numHeads = config.numAttentionHeads();
        this.numKvHeads = config.numKeyValueHeads();
        this.headDim = config.headDim();
        this.halfDim = headDim / 2;
        this.numLayers = config.numHiddenLayers();
        this.vocabSize = config.vocabSize();

        long hiddenBytes = (long) hidden * Float.BYTES;
        long qkvBytes = (long) hidden * 3 * Float.BYTES;
        long vocabBytes = (long) config.vocabSize() * Float.BYTES;

        long maxUpload = hiddenBytes;
        long maxReadback = Math.max(qkvBytes, Math.max(vocabBytes, hiddenBytes));

        this.pipeline = new GpuPipeline(wb, maxUpload, maxReadback);

        // ── Try V3.0 full GPU shaders first ────────────────────────────
        try {
            computeKernels = Phi3ComputeShaders.createAllV3(wb, pipeline.getCommandList());
            initV3Buffers(wb, config);
            fullGpuEnabled = computeKernels.hasV3();
            mlpBatchEnabled = true;
            log.info("Phi3GpuPipeline V3.0: full GPU decode {} (1 submission/token)",
                    fullGpuEnabled ? "ENABLED" : "DISABLED (shader fallback)");
        } catch (Exception e) {
            log.warn("V3.0 shader compilation failed, trying V2.0: {}", e.getMessage());
            try {
                computeKernels = Phi3ComputeShaders.createAll(wb, pipeline.getCommandList());
                mlpBatchEnabled = true;
            } catch (Exception e2) {
                log.warn("V2.0 shader compilation also failed: {}", e2.getMessage());
                computeKernels = null;
                mlpBatchEnabled = false;
            }
        }

        // ── V2.0 MLP batch buffers (always needed as fallback) ─────────
        if (mlpBatchEnabled) {
            var dev = wb.getD3d12Device();
            var arena = pipeline.getArena();
            if (residualBuf == null) {
                residualBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            }
            uavBarrier = pipeline.allocUavBarrier();
            barrierResidualCopyDestToUav = pipeline.allocTransitionBarrier(residualBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            barrierResidualUavToCopySource = pipeline.allocTransitionBarrier(residualBuf,
                    D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
        }

        log.info("Phi3GpuPipeline ready: fullGpu={}, mlpBatch={}, upload={}KB readback={}KB",
                fullGpuEnabled, mlpBatchEnabled, maxUpload / 1024, maxReadback / 1024);
    }

    /**
     * Allocate V3.0 GPU intermediate buffers and KV cache.
     */
    private void initV3Buffers(WindowsBindings wb, Phi3Config config)
            throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var arena = pipeline.getArena();
        long hiddenBytes = (long) hidden * Float.BYTES;

        // Intermediate buffers
        hiddenBufA = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
        hiddenBufB = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
        attnOutBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
        residualBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);

        // Scores buffer: [numHeads * maxPos] — max size for full context
        maxGpuPos = Math.min(config.maxPositionEmbeddings(), 2048); // VRAM budget limit
        long scoresBytes = (long) numHeads * maxGpuPos * Float.BYTES;
        scoresBuf = D3D12Bindings.createDefaultBuffer(dev, scoresBytes, arena);

        // GPU KV cache: per-layer K and V buffers
        long kvBytes = (long) maxGpuPos * hidden * Float.BYTES;
        gpuKCacheBufs = new MemorySegment[numLayers];
        gpuVCacheBufs = new MemorySegment[numLayers];
        for (int l = 0; l < numLayers; l++) {
            gpuKCacheBufs[l] = D3D12Bindings.createDefaultBuffer(dev, kvBytes, arena);
            gpuVCacheBufs[l] = D3D12Bindings.createDefaultBuffer(dev, kvBytes, arena);
        }

        // Pre-allocate barriers for hiddenBufA upload
        barrierHiddenACopyDestToUav = pipeline.allocTransitionBarrier(hiddenBufA,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrierHiddenAUavToCommon = pipeline.allocTransitionBarrier(hiddenBufA,
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COMMON);

        long kvTotalMB = numLayers * 2 * kvBytes / (1024 * 1024);
        log.info("V3.0 buffers allocated: KV cache={}MB (maxPos={}), scores={}KB",
                kvTotalMB, maxGpuPos, scoresBytes / 1024);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Single-GEMM dispatch using shared pipeline (V2.0 fallback)
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
    // MLP Batch: 7 GPU ops, 1 submission (V2.0)
    // ═══════════════════════════════════════════════════════════════════

    /** Whether MLP batching is available (compute shaders compiled + weights uploaded). */
    public boolean isMlpBatchEnabled() { return mlpBatchEnabled && weightsUploaded; }

    /**
     * Upload per-layer weights to GPU (for V2.0 MLP batch + V3.0 full GPU).
     */
    public void uploadLayerWeights(WindowsBindings wb, Phi3Weights weights, Phi3Config config)
            throws WindowsNativeException {
        if (!mlpBatchEnabled) return;

        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var arena = pipeline.getArena();
        int nLayers = config.numHiddenLayers();
        long hiddenBytes = (long) hidden * Float.BYTES;
        long interBytes = (long) intermediate * Float.BYTES;

        // V2.0 weights
        postNormWeightBufs = new MemorySegment[nLayers];
        mlpOutScaleBufs = new MemorySegment[nLayers];

        // V3.0 weights
        if (fullGpuEnabled) {
            inputNormWeightBufs = new MemorySegment[nLayers];
            attnOutScaleBufs = new MemorySegment[nLayers];
        }

        long t0 = System.currentTimeMillis();
        for (int l = 0; l < nLayers; l++) {
            var lw = weights.layers[l];
            postNormWeightBufs[l] = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, postNormWeightBufs[l], lw.postNormWeight(), arena);
            mlpOutScaleBufs[l] = D3D12Bindings.createDefaultBuffer(dev, interBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, mlpOutScaleBufs[l], lw.mlpOutScale(), arena);

            if (fullGpuEnabled) {
                inputNormWeightBufs[l] = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
                D3D12Bindings.uploadFloats(dev, queue, inputNormWeightBufs[l], lw.inputNormWeight(), arena);
                attnOutScaleBufs[l] = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
                D3D12Bindings.uploadFloats(dev, queue, attnOutScaleBufs[l], lw.attnOutScale(), arena);
            }
        }

        // V3.0 global weights
        if (fullGpuEnabled) {
            long cosBytes = (long) weights.cosCache.length * Float.BYTES;
            long sinBytes = (long) weights.sinCache.length * Float.BYTES;
            cosCacheBuf = D3D12Bindings.createDefaultBuffer(dev, cosBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, cosCacheBuf, weights.cosCache, arena);
            sinCacheBuf = D3D12Bindings.createDefaultBuffer(dev, sinBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, sinCacheBuf, weights.sinCache, arena);
            finalNormWeightBuf = D3D12Bindings.createDefaultBuffer(dev, hiddenBytes, arena);
            D3D12Bindings.uploadFloats(dev, queue, finalNormWeightBuf, weights.finalNormWeight, arena);
        }

        weightsUploaded = true;
        long elapsed = System.currentTimeMillis() - t0;
        int totalBufs = nLayers * 2 + (fullGpuEnabled ? nLayers * 2 + 3 : 0);
        log.info("Uploaded {} weight buffers to GPU in {} ms (fullGpu={})",
                totalBufs, elapsed, fullGpuEnabled);
    }

    /**
     * Batched MLP: 7 GPU operations in ONE submission (V2.0).
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

        oK.recordBatchFromCpu(pipeline, attnOutput);
        pipeline.recordUpload(hiddenInput, 0, hidden, residualBuf, 0);
        pipeline.recordBarrier(barrierResidualCopyDestToUav);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        pipeline.recordUavBarrier(uavBarrier);

        int epsBits = Float.floatToRawIntBits(rmsNormEps);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(postNormWeightBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(guK.getInputBuf())
                },
                new int[]{ hidden, epsBits },
                1);

        pipeline.recordUavBarrier(uavBarrier);
        guK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(mlpOutScaleBufs[layerIdx]),
                        D3D12Bindings.getGpuVirtualAddress(downK.getInputBuf())
                },
                new int[]{ intermediate },
                intermediate);

        pipeline.recordUavBarrier(uavBarrier);
        downK.recordBatchDispatchOnly(pipeline);
        pipeline.recordUavBarrier(uavBarrier);

        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        pipeline.recordBarrier(barrierResidualUavToCopySource);
        pipeline.recordReadback(residualBuf, 0, hiddenBytes);
        pipeline.submitAndWait();
        pipeline.readbackInto(hiddenOut, 0, hidden);
    }

    // ═══════════════════════════════════════════════════════════════════
    // V3.0: Full GPU Decode — ALL layers + lm_head in 1 submission
    // ═══════════════════════════════════════════════════════════════════

    /** Whether V3.0 full GPU decode is available. */
    public boolean isFullGpuEnabled() {
        return fullGpuEnabled && weightsUploaded;
    }

    /** Max positions supported by GPU KV cache. */
    public int getMaxGpuPos() { return maxGpuPos; }

    /**
     * Decode ONE token entirely on GPU. All 32 layers + lm_head recorded
     * into a SINGLE command list with ONE fence wait.
     * <p>
     * <b>Total submissions: 1</b> (was 129 in V1.x, 65 in V2.0)
     * <p>
     * The embedding lookup is the only CPU operation. Everything else
     * (RMSNorm, QKV GEMM, RoPE, KV cache store, attention, activation scale,
     * O GEMM, residual add, post-norm, GateUp GEMM, SwiGLU, Down GEMM,
     * final norm, lm_head) runs on GPU.
     *
     * @param embedding  CPU [hidden] — token embedding (from weights.embedTokens)
     * @param pos        current KV cache position (0-based)
     * @param logitsOut  CPU [vocabSize] — output logits (filled after return)
     */
    public void decodeTokenFullGpu(float[] embedding, int pos, float[] logitsOut) {
        if (!isFullGpuEnabled()) {
            throw new IllegalStateException("Full GPU decode not enabled");
        }
        if (pos >= maxGpuPos) {
            throw new IllegalStateException("Position " + pos + " exceeds GPU KV cache limit " + maxGpuPos);
        }

        int seqLen = pos + 1;  // total positions to attend to
        int epsBits = Float.floatToRawIntBits(rmsNormEps);
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int scaleBits = Float.floatToRawIntBits(scale);
        int ropePairs = (numHeads + numKvHeads) * halfDim;

        pipeline.begin();
        var cl = pipeline.getCommandList();

        // ── Upload embedding → hiddenBufA ─────────────────────────────
        pipeline.recordUpload(embedding, 0, hidden, hiddenBufA, 0);
        pipeline.recordBarrier(barrierHiddenACopyDestToUav);

        // ── Process 32 layers ─────────────────────────────────────────
        MemorySegment currentHidden = hiddenBufA;
        MemorySegment nextHidden = hiddenBufB;

        for (int l = 0; l < numLayers; l++) {
            recordLayerFullGpu(cl, l, currentHidden, nextHidden, pos, seqLen,
                    epsBits, scaleBits, ropePairs);
            // Swap buffers
            var tmp = currentHidden;
            currentHidden = nextHidden;
            nextHidden = tmp;
        }

        // ── Final RMSNorm → lm_head input ──────────────────────────────
        MatMulNBitsKernel lmK = kernels.lmHead();
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(currentHidden),
                        D3D12Bindings.getGpuVirtualAddress(finalNormWeightBuf),
                        D3D12Bindings.getGpuVirtualAddress(lmK.getInputBuf())
                },
                new int[]{ hidden, epsBits },
                1);

        // ── LM head GEMM ──────────────────────────────────────────────
        pipeline.recordUavBarrier(uavBarrier);
        lmK.recordBatchDispatchOnly(pipeline);

        // ── Readback logits ───────────────────────────────────────────
        MemorySegment lmOutBarrier = pipeline.allocTransitionBarrier(lmK.getOutputBuf(),
                D3D12Bindings.D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12Bindings.D3D12_RESOURCE_STATE_COPY_SOURCE);
        pipeline.recordBarrier(lmOutBarrier);
        long vocabBytes = (long) vocabSize * Float.BYTES;
        pipeline.recordReadback(lmK.getOutputBuf(), 0, vocabBytes);

        // ── Cleanup: hiddenBufA back to COMMON (explicit transition) ──
        pipeline.recordBarrier(barrierHiddenAUavToCommon);

        // ── SINGLE submit + wait ──────────────────────────────────────
        pipeline.submitAndWait();

        // ── Readback logits to CPU ────────────────────────────────────
        pipeline.readbackInto(logitsOut, 0, vocabSize);
    }

    /**
     * Record one full decoder layer into the command list.
     * All operations (norm, QKV, RoPE, KV cache, attention, scale, O, residual,
     * post-norm, GateUp, SwiGLU, Down, residual) are GPU-only.
     */
    private void recordLayerFullGpu(MemorySegment cl, int l,
                                     MemorySegment currentHidden, MemorySegment nextHidden,
                                     int pos, int seqLen,
                                     int epsBits, int scaleBits, int ropePairs) {
        MatMulNBitsKernel qkvK = kernels.qkvFused(l);
        MatMulNBitsKernel oK = kernels.oProj(l);
        MatMulNBitsKernel guK = kernels.gateUpProj(l);
        MatMulNBitsKernel downK = kernels.downProj(l);

        // ── 1. Pre-attention RMSNorm: currentHidden → qkvK.inputBuf ──
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(currentHidden),
                        D3D12Bindings.getGpuVirtualAddress(inputNormWeightBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(qkvK.getInputBuf())
                },
                new int[]{ hidden, epsBits },
                1);

        // ── 2. QKV GEMM: qkvK.inputBuf → qkvK.outputBuf [3*hidden] ──
        pipeline.recordUavBarrier(uavBarrier);
        qkvK.recordBatchDispatchOnly(pipeline);

        // ── 3. RoPE: in-place on qkvK.outputBuf (Q and K portions) ───
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.rope().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(qkvK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(cosCacheBuf),
                        D3D12Bindings.getGpuVirtualAddress(sinCacheBuf)
                },
                new int[]{ headDim, numHeads, numKvHeads, pos },
                ropePairs);

        // ── 4. KV cache store: K/V from qkvK.outputBuf → GPU cache ───
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.kvCacheStore().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(qkvK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(gpuKCacheBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(gpuVCacheBufs[l])
                },
                new int[]{ hidden, pos },
                hidden);

        // ── 5. Attention scores: Q · K_cache → scores ────────────────
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.attnScore().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(qkvK.getOutputBuf()), // Q portion
                        D3D12Bindings.getGpuVirtualAddress(gpuKCacheBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(scoresBuf)
                },
                new int[]{ headDim, numHeads, seqLen, scaleBits },
                numHeads * seqLen);

        // ── 6. Attention softmax: scores (in-place) ──────────────────
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.attnSoftmax().recordDispatch(cl,
                new long[]{ D3D12Bindings.getGpuVirtualAddress(scoresBuf) },
                new int[]{ seqLen, numHeads },
                numHeads);  // 1 group per head

        // ── 7. Attention V-sum: scores · V_cache → attnOutBuf ─────────
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.attnVsum().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(scoresBuf),
                        D3D12Bindings.getGpuVirtualAddress(gpuVCacheBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(attnOutBuf)
                },
                new int[]{ headDim, numHeads, seqLen },
                numHeads);  // 1 group per head

        // ── 8. Attention scale: attnOutBuf → oK.inputBuf ─────────────
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.scale().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(attnOutBuf),
                        D3D12Bindings.getGpuVirtualAddress(attnOutScaleBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(oK.getInputBuf())
                },
                new int[]{ hidden },
                hidden);

        // ── 9. O projection GEMM ─────────────────────────────────────
        pipeline.recordUavBarrier(uavBarrier);
        oK.recordBatchDispatchOnly(pipeline);

        // ── 10. Residual1 add: currentHidden + oK.outputBuf → residualBuf ─
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(currentHidden),
                        D3D12Bindings.getGpuVirtualAddress(oK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(residualBuf)
                },
                new int[]{ hidden },
                hidden);

        // ── 11. Post-attention RMSNorm: residualBuf → guK.inputBuf ────
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.rmsNorm().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(postNormWeightBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(guK.getInputBuf())
                },
                new int[]{ hidden, epsBits },
                1);

        // ── 12. GateUp GEMM ──────────────────────────────────────────
        pipeline.recordUavBarrier(uavBarrier);
        guK.recordBatchDispatchOnly(pipeline);

        // ── 13. SwiGLU + Scale: guK.outputBuf → downK.inputBuf ───────
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.swiglu().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(guK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(mlpOutScaleBufs[l]),
                        D3D12Bindings.getGpuVirtualAddress(downK.getInputBuf())
                },
                new int[]{ intermediate },
                intermediate);

        // ── 14. Down GEMM ────────────────────────────────────────────
        pipeline.recordUavBarrier(uavBarrier);
        downK.recordBatchDispatchOnly(pipeline);

        // ── 15. Residual2 add: residualBuf + downK.outputBuf → nextHidden ─
        pipeline.recordUavBarrier(uavBarrier);
        computeKernels.add().recordDispatch(cl,
                new long[]{
                        D3D12Bindings.getGpuVirtualAddress(residualBuf),
                        D3D12Bindings.getGpuVirtualAddress(downK.getOutputBuf()),
                        D3D12Bindings.getGpuVirtualAddress(nextHidden)
                },
                new int[]{ hidden },
                hidden);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════════════

    public boolean hasLayer(int layerIdx) { return kernels.hasLayer(layerIdx); }
    public boolean hasLmHead() { return kernels.hasLmHead(); }
    public GpuPipeline getPipeline() { return pipeline; }

    /**
     * Enable/disable V3.0 full GPU decode at runtime.
     * Useful for A/B benchmarking (V2.0 vs V3.0).
     * Only effective if V3.0 shaders compiled successfully.
     */
    public void setFullGpuEnabled(boolean enabled) {
        if (enabled && computeKernels != null && computeKernels.hasV3()) {
            fullGpuEnabled = true;
            log.info("V3.0 full GPU decode re-enabled");
        } else {
            fullGpuEnabled = false;
            log.info("V3.0 full GPU decode disabled — V2.0 fallback active");
        }
    }

    /** Whether V3.0 shaders compiled (even if currently disabled). */
    public boolean hasV3Capability() {
        return computeKernels != null && computeKernels.hasV3();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (computeKernels != null) computeKernels.close();
        pipeline.close();
        log.info("Phi3GpuPipeline closed");
    }
}
