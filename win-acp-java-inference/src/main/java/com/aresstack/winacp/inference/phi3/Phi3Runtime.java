package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.winacp.windows.MatMulNBitsKernel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Decoder runtime for Phi-3-mini-4k-instruct.
 *
 * <p>Implements the full Phi-3 decoder stack:
 * <ol>
 *   <li>Token embedding lookup</li>
 *   <li>Per-layer: RMSNorm, Q/K/V projection, RoPE, causal attention with KV-cache,
 *       O projection, residual, RMSNorm, SwiGLU MLP, residual</li>
 *   <li>Final RMSNorm + LM head logits</li>
 * </ol>
 *
 * <p>V1 supports two execution modes:
 * <ul>
 *   <li><b>CPU-only</b>: all matrix multiplications on CPU (default)</li>
 *   <li><b>GPU-accelerated</b>: quantized projections dispatched to DirectML
 *       via {@link Phi3GpuKernels}; attention, norms, and activations remain on CPU</li>
 * </ul>
 *
 * <p>Constraints:
 * <ul>
 *   <li>Greedy decoding only (no sampling)</li>
 *   <li>Single-batch (batch_size=1)</li>
 * </ul>
 */
public final class Phi3Runtime {

    private static final Logger log = LoggerFactory.getLogger(Phi3Runtime.class);

    private final Phi3Config config;
    private final Phi3Weights weights;
    private final Phi3Tokenizer tokenizer;
    private final Phi3GpuKernels gpuKernels;

    private final float[][] kvCache;
    private int cachedSeqLen;

    // ── Decode profiling (accumulated per generation, reset on each call) ─
    private long profGpuProjNs;   // GPU Q/K/V/O + gate_up + down projections
    private long profCpuAttnNs;   // CPU causal self-attention (dot + softmax + weighted sum)
    private long profCpuNormNs;   // CPU RMSNorm + RoPE + residual + scales
    private long profCpuActNs;    // CPU SwiGLU activation
    private long profLmHeadNs;    // LM head (GPU or CPU)
    private long profTokenDecNs;  // Token decode (tokenizer)
    private long profPrefillNs;   // Prefill phase
    private int  profSteps;
    private String lastProfile;   // formatted profile string from last generation

    /**
     * CPU-only constructor (backward compatible).
     */
    public Phi3Runtime(Phi3Config config, Phi3Weights weights, Phi3Tokenizer tokenizer) {
        this(config, weights, tokenizer, null);
    }

    /**
     * @param gpuKernels optional GPU kernel pool — {@code null} for CPU-only mode
     */
    public Phi3Runtime(Phi3Config config, Phi3Weights weights, Phi3Tokenizer tokenizer,
                       Phi3GpuKernels gpuKernels) {
        this.config = config;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.gpuKernels = gpuKernels;
        this.kvCache = new float[config.numHiddenLayers() * 2][];
        this.cachedSeqLen = 0;

        if (gpuKernels != null) {
            log.info("Phi3Runtime: GPU mode — {}/{} layers on GPU, lmHead={}",
                    gpuKernels.getGpuLayers(), config.numHiddenLayers(), gpuKernels.hasLmHead());
        } else {
            log.info("Phi3Runtime: CPU-only mode");
        }
    }

    // ── Streaming callback ────────────────────────────────────────────────

    /**
     * Callback for token-by-token streaming during generation.
     */
    @FunctionalInterface
    public interface TokenConsumer {
        /**
         * Called after each generated token.
         *
         * @param tokenId   the token ID just generated
         * @param textSoFar full decoded text of all tokens generated so far
         * @param delta     new text fragment appended in this step
         */
        void onToken(int tokenId, String textSoFar, String delta);
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Generate tokens greedily from a prompt (non-streaming).
     *
     * @param prompt    text prompt
     * @param maxTokens maximum number of tokens to generate
     * @return generated text (excluding the prompt)
     */
    public String generate(String prompt, int maxTokens) {
        return generateStreaming(prompt, maxTokens, null);
    }

    /**
     * Returns the formatted profile string from the last generation, or null.
     */
    public String getLastProfile() {
        return lastProfile;
    }

    /**
     * Generate tokens greedily with a per-token streaming callback.
     * <p>
     * Token IDs are accumulated and decoded as a full sequence after each
     * step so that SentencePiece inter-token spaces are preserved correctly.
     *
     * @param prompt    text prompt
     * @param maxTokens maximum number of tokens to generate
     * @param consumer  optional callback invoked after each token (may be {@code null})
     * @return generated text (excluding the prompt)
     */
    public String generateStreaming(String prompt, int maxTokens, TokenConsumer consumer) {
        int[] inputIds = tokenizer.encode(prompt);
        log.info("Prompt: {} chars → {} tokens", prompt.length(), inputIds.length);

        // Reset KV cache and profiling
        resetCache();
        resetProfile();

        // Prefill
        long t0 = System.nanoTime();
        float[] logits = prefill(inputIds);
        profPrefillNs = System.nanoTime() - t0;

        // Accumulate generated token IDs for proper decoding (preserves spaces)
        List<Integer> generatedIds = new ArrayList<>();
        String previousText = "";

        for (int step = 0; step < maxTokens; step++) {
            int nextToken = argmax(logits);

            if (tokenizer.isEos(nextToken)) {
                break;
            }

            generatedIds.add(nextToken);

            // Decode full sequence to preserve SentencePiece spaces
            long td0 = System.nanoTime();
            String fullText = tokenizer.decode(
                    generatedIds.stream().mapToInt(Integer::intValue).toArray());
            String delta = fullText.substring(previousText.length());
            previousText = fullText;
            profTokenDecNs += System.nanoTime() - td0;

            if (consumer != null) {
                consumer.onToken(nextToken, fullText, delta);
            }

            // Decode: process single token
            logits = decode(nextToken);
            profSteps++;
        }

        // Build profile summary
        lastProfile = buildProfileSummary();
        log.debug("Profile: {}", lastProfile);

        return previousText;
    }

    // ── Profiling ────────────────────────────────────────────────────────

    private void resetProfile() {
        profGpuProjNs = profCpuAttnNs = profCpuNormNs = profCpuActNs = 0;
        profLmHeadNs = profTokenDecNs = profPrefillNs = 0;
        profSteps = 0;
        lastProfile = null;
    }

    private String buildProfileSummary() {
        if (profSteps == 0) return "[No tokens generated]";
        long totalDecode = profGpuProjNs + profCpuAttnNs + profCpuNormNs + profCpuActNs + profLmHeadNs;
        double totalMs = totalDecode / 1e6;
        double perToken = totalMs / profSteps;

        // Guard against division by zero if totalDecode is 0
        double pctDivisor = totalDecode > 0 ? totalDecode : 1;

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[Decode Profile] %d tokens, %.1f ms total, %.1f ms/token%n", profSteps, totalMs, perToken));
        sb.append(String.format("  Prefill:         %.1f ms (%d prompt tokens)%n", profPrefillNs / 1e6, cachedSeqLen - profSteps));
        sb.append(String.format("  GPU projections: %.1f ms avg (%.0f%%) [%d calls/tok × 32 layers]%n",
                profGpuProjNs / 1e6 / profSteps, 100.0 * profGpuProjNs / pctDivisor,
                gpuKernels != null ? 6 : 0));
        sb.append(String.format("  CPU attention:   %.1f ms avg (%.0f%%)%n",
                profCpuAttnNs / 1e6 / profSteps, 100.0 * profCpuAttnNs / pctDivisor));
        sb.append(String.format("  CPU norms+RoPE:  %.1f ms avg (%.0f%%)%n",
                profCpuNormNs / 1e6 / profSteps, 100.0 * profCpuNormNs / pctDivisor));
        sb.append(String.format("  CPU SwiGLU:      %.1f ms avg (%.0f%%)%n",
                profCpuActNs / 1e6 / profSteps, 100.0 * profCpuActNs / pctDivisor));
        sb.append(String.format("  LM head:         %.1f ms avg (%.0f%%)%n",
                profLmHeadNs / 1e6 / profSteps, 100.0 * profLmHeadNs / pctDivisor));
        sb.append(String.format("  Token decode:    %.1f ms avg%n", profTokenDecNs / 1e6 / profSteps));
        return sb.toString();
    }

    public void resetCache() {
        cachedSeqLen = 0;
        Arrays.fill(kvCache, null);
    }

    // ── Prefill ──────────────────────────────────────────────────────────

    /**
     * Process multiple tokens at once (prefill phase).
     * Returns logits for the last position.
     */
    private float[] prefill(int[] tokenIds) {
        int seqLen = tokenIds.length;
        int hidden = config.hiddenSize();

        // Embedding lookup
        float[] hidden_states = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            System.arraycopy(weights.embedTokens, tokenIds[s] * hidden, hidden_states, s * hidden, hidden);
        }

        // Process each layer
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            hidden_states = processLayer(l, hidden_states, seqLen, 0);
        }

        // Final norm + logits (only for last position)
        float[] lastHidden = new float[hidden];
        System.arraycopy(hidden_states, (seqLen - 1) * hidden, lastHidden, 0, hidden);
        rmsNorm(lastHidden, weights.finalNormWeight, config.rmsNormEps());

        float[] logits = new float[weights.lmHead.N()];
        lmHeadMatvec(lastHidden, logits);

        cachedSeqLen = seqLen;
        return logits;
    }

    // ── Single-token decode ──────────────────────────────────────────────

    /**
     * Process a single new token (autoregressive decode step).
     * Returns logits for the new position.
     */
    private float[] decode(int tokenId) {
        int hidden = config.hiddenSize();
        int pos = cachedSeqLen;

        // Embedding lookup
        float[] hidden_states = new float[hidden];
        System.arraycopy(weights.embedTokens, tokenId * hidden, hidden_states, 0, hidden);

        // Process each layer (seqLen=1)
        float[] buf = new float[hidden];
        System.arraycopy(hidden_states, 0, buf, 0, hidden);
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            buf = processLayer(l, buf, 1, pos);
        }
        System.arraycopy(buf, 0, hidden_states, 0, hidden);

        long t0 = System.nanoTime();
        rmsNorm(hidden_states, weights.finalNormWeight, config.rmsNormEps());
        profCpuNormNs += System.nanoTime() - t0;

        t0 = System.nanoTime();
        float[] logits = new float[weights.lmHead.N()];
        lmHeadMatvec(hidden_states, logits);
        profLmHeadNs += System.nanoTime() - t0;

        cachedSeqLen = pos + 1;
        return logits;
    }

    // ── LM head ──────────────────────────────────────────────────────────

    /**
     * Compute logits = x @ lm_head^T, using GPU if available.
     */
    private void lmHeadMatvec(float[] x, float[] logits) {
        if (gpuKernels != null && gpuKernels.hasLmHead()) {
            float[] result = gpuKernels.lmHead().matvec(x);
            System.arraycopy(result, 0, logits, 0, result.length);
        } else {
            weights.lmHead.matvec(x, logits);
        }
    }

    // ── Layer processing (with profiling) ────────────────────────────────

    /**
     * Process one decoder layer.
     *
     * @param layerIdx    layer index
     * @param input       input hidden states [seqLen, hidden]
     * @param seqLen      sequence length
     * @param startPos    position offset (0 for prefill, cachedSeqLen for decode)
     * @return output hidden states [seqLen, hidden]
     */
    private float[] processLayer(int layerIdx, float[] input, int seqLen, int startPos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        LayerWeights lw = weights.layers[layerIdx];
        boolean gpuLayer = gpuKernels != null && gpuKernels.hasLayer(layerIdx);
        long t0;

        // ── Pre-attention RMSNorm ────────────────────────────────────
        t0 = System.nanoTime();
        float[] normed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(input, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.inputNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, normed, s * hidden, hidden);
        }
        profCpuNormNs += System.nanoTime() - t0;

        // ── Q/K/V Projections (GPU or CPU) ────────────────────────────
        t0 = System.nanoTime();
        float[] q = new float[seqLen * hidden];
        float[] k = new float[seqLen * hidden];
        float[] v = new float[seqLen * hidden];

        if (gpuLayer) {
            gpuMatmul(gpuKernels.qProj(layerIdx), normed, q, seqLen, hidden, hidden);
            gpuMatmul(gpuKernels.kProj(layerIdx), normed, k, seqLen, hidden, hidden);
            gpuMatmul(gpuKernels.vProj(layerIdx), normed, v, seqLen, hidden, hidden);
        } else {
            lw.qProj().matmul(normed, q, seqLen);
            lw.kProj().matmul(normed, k, seqLen);
            lw.vProj().matmul(normed, v, seqLen);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── RoPE ─────────────────────────────────────────────────────
        t0 = System.nanoTime();
        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                applyRoPE(q, s * hidden + h * headDim, headDim, pos);
            }
            for (int h = 0; h < kvHeads; h++) {
                applyRoPE(k, s * hidden + h * headDim, headDim, pos);
            }
        }
        profCpuNormNs += System.nanoTime() - t0;

        // ── KV Cache update ──────────────────────────────────────────
        int kCacheIdx = layerIdx * 2;
        int vCacheIdx = layerIdx * 2 + 1;
        int totalSeq = startPos + seqLen;
        int cacheRowSize = kvHeads * headDim;

        if (kvCache[kCacheIdx] == null || kvCache[kCacheIdx].length < totalSeq * cacheRowSize) {
            int capacity = Math.max(totalSeq, 64) * cacheRowSize;
            float[] newK = new float[capacity];
            float[] newV = new float[capacity];
            if (kvCache[kCacheIdx] != null) {
                System.arraycopy(kvCache[kCacheIdx], 0, newK, 0,
                        Math.min(kvCache[kCacheIdx].length, startPos * cacheRowSize));
                System.arraycopy(kvCache[vCacheIdx], 0, newV, 0,
                        Math.min(kvCache[vCacheIdx].length, startPos * cacheRowSize));
            }
            kvCache[kCacheIdx] = newK;
            kvCache[vCacheIdx] = newV;
        }

        for (int s = 0; s < seqLen; s++) {
            System.arraycopy(k, s * hidden, kvCache[kCacheIdx],
                    (startPos + s) * cacheRowSize, cacheRowSize);
            System.arraycopy(v, s * hidden, kvCache[vCacheIdx],
                    (startPos + s) * cacheRowSize, cacheRowSize);
        }

        // ── Causal Self-Attention ────────────────────────────────────
        t0 = System.nanoTime();
        float[] attnOut = new float[seqLen * hidden];
        float scale = (float) (1.0 / Math.sqrt(headDim));

        for (int s = 0; s < seqLen; s++) {
            int queryPos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                int kvH = h % kvHeads;
                int qOff = s * hidden + h * headDim;

                float[] scores = new float[queryPos + 1];
                for (int p = 0; p <= queryPos; p++) {
                    int kOff = p * cacheRowSize + kvH * headDim;
                    float dot = 0;
                    for (int d = 0; d < headDim; d++) {
                        dot += q[qOff + d] * kvCache[kCacheIdx][kOff + d];
                    }
                    scores[p] = dot * scale;
                }

                softmax(scores);

                int outOff = s * hidden + h * headDim;
                for (int p = 0; p <= queryPos; p++) {
                    int vOff = p * cacheRowSize + kvH * headDim;
                    float w = scores[p];
                    for (int d = 0; d < headDim; d++) {
                        attnOut[outOff + d] += w * kvCache[vCacheIdx][vOff + d];
                    }
                }
            }
        }
        profCpuAttnNs += System.nanoTime() - t0;

        // ── Activation scale + O projection ──────────────────────────
        t0 = System.nanoTime();
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < hidden; i++) {
                attnOut[s * hidden + i] *= lw.attnOutScale()[i];
            }
        }
        profCpuNormNs += System.nanoTime() - t0;

        t0 = System.nanoTime();
        float[] oProjOut = new float[seqLen * hidden];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.oProj(layerIdx), attnOut, oProjOut, seqLen, hidden, hidden);
        } else {
            lw.oProj().matmul(attnOut, oProjOut, seqLen);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── Residual connection 1 ────────────────────────────────────
        t0 = System.nanoTime();
        float[] residual1 = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            residual1[i] = input[i] + oProjOut[i];
        }

        // ── Post-attention RMSNorm ───────────────────────────────────
        float[] postNormed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(residual1, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.postNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, postNormed, s * hidden, hidden);
        }
        profCpuNormNs += System.nanoTime() - t0;

        // ── MLP: gate_up_proj (GPU or CPU) ────────────────────────────
        t0 = System.nanoTime();
        int intermediateX2 = config.intermediateSize() * 2;
        float[] gateUp = new float[seqLen * intermediateX2];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.gateUpProj(layerIdx), postNormed, gateUp,
                    seqLen, hidden, intermediateX2);
        } else {
            lw.gateUpProj().matmul(postNormed, gateUp, seqLen);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── SwiGLU activation ────────────────────────────────────────
        t0 = System.nanoTime();
        int intermediate = config.intermediateSize();
        float[] mlpActivation = new float[seqLen * intermediate];
        for (int s = 0; s < seqLen; s++) {
            int guOff = s * intermediateX2;
            int outOff = s * intermediate;
            for (int i = 0; i < intermediate; i++) {
                float gate = gateUp[guOff + i];
                float up = gateUp[guOff + intermediate + i];
                float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
                mlpActivation[outOff + i] = up * gate * sigmoid;
            }
        }
        profCpuActNs += System.nanoTime() - t0;

        // ── Activation scale + down_proj (GPU or CPU) ─────────────────
        t0 = System.nanoTime();
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < intermediate; i++) {
                mlpActivation[s * intermediate + i] *= lw.mlpOutScale()[i];
            }
        }
        profCpuNormNs += System.nanoTime() - t0;

        t0 = System.nanoTime();
        float[] downOut = new float[seqLen * hidden];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.downProj(layerIdx), mlpActivation, downOut,
                    seqLen, intermediate, hidden);
        } else {
            lw.downProj().matmul(mlpActivation, downOut, seqLen);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── Residual connection 2 ────────────────────────────────────
        float[] output = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            output[i] = residual1[i] + downOut[i];
        }

        return output;
    }

    // ── GPU matmul helper ─────────────────────────────────────────────────

    /**
     * GPU-accelerated matrix multiplication: Y[seqLen, N] = X[seqLen, K] @ W^T.
     * Calls {@link MatMulNBitsKernel#matvec(float[])} once per row.
     */
    private static void gpuMatmul(MatMulNBitsKernel kernel,
                                   float[] input, float[] output,
                                   int seqLen, int K, int N) {
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[K];
            System.arraycopy(input, s * K, row, 0, K);
            float[] result = kernel.matvec(row);
            System.arraycopy(result, 0, output, s * N, N);
        }
    }

    // ── Math utilities ───────────────────────────────────────────────────

    /**
     * RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
     * Operates in-place on the input array.
     */
    static void rmsNorm(float[] x, float[] weight, float eps) {
        float sumSq = 0;
        for (float v : x) sumSq += v * v;
        float rms = (float) (1.0 / Math.sqrt(sumSq / x.length + eps));
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] * rms * weight[i];
        }
    }

    /**
     * Apply Rotary Position Embedding (RoPE) to a vector.
     *
     * @param vec    the vector array
     * @param offset offset into the array
     * @param dim    head dimension
     * @param pos    absolute position
     */
    private void applyRoPE(float[] vec, int offset, int dim, int pos) {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++) {
            // Use precomputed cos/sin cache
            float cos = weights.cosCache[pos * halfDim + i];
            float sin = weights.sinCache[pos * halfDim + i];

            float x0 = vec[offset + i];
            float x1 = vec[offset + halfDim + i];

            vec[offset + i] = x0 * cos - x1 * sin;
            vec[offset + halfDim + i] = x0 * sin + x1 * cos;
        }
    }

    /**
     * In-place softmax over a float array.
     */
    static void softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) if (v > max) max = v;

        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < x.length; i++) {
            x[i] *= invSum;
        }
    }

    /**
     * Greedy argmax over logits.
     */
    static int argmax(float[] logits) {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
