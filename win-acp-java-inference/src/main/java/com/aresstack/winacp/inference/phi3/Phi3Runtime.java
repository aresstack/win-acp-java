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

    // ── KV-cache token tracking for incremental prefill ──────────────────
    private int[] cachedTokenIds;  // token IDs currently represented in KV cache

    // ── Pre-allocated decode buffers (seqLen=1, reused across layers) ─────
    private final float[] decBuf;         // general-purpose [hidden]
    private final float[] decQ;           // [hidden]
    private final float[] decK;           // [hidden]
    private final float[] decV;           // [hidden]
    private final float[] decAttnOut;     // [hidden]
    private final float[] decOProj;       // [hidden]
    private final float[] decResidual;    // [hidden]
    private final float[] decPostNorm;    // [hidden]
    private final float[] decGateUp;      // [intermediateSize * 2]
    private final float[] decMlpAct;      // [intermediateSize]
    private final float[] decDown;        // [hidden]
    private final float[] decScores;      // [maxPositionEmbeddings] — attention scores
    private final float[] decLogits;      // [vocabSize]

    // ── Decode profiling (accumulated per generation, reset on each call) ─
    private long profGpuProjNs;
    private long profCpuAttnNs;
    private long profCpuNormNs;
    private long profCpuActNs;
    private long profLmHeadNs;
    private long profTokenDecNs;
    private long profPrefillNs;
    private int  profSteps;
    private int  profPrefillTokens;
    private String lastProfile;

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
        this.cachedTokenIds = new int[0];

        // Pre-allocate decode buffers (seqLen=1)
        int hidden = config.hiddenSize();
        int interX2 = config.intermediateSize() * 2;
        int inter   = config.intermediateSize();
        int maxPos  = config.maxPositionEmbeddings();

        decBuf      = new float[hidden];
        decQ        = new float[hidden];
        decK        = new float[hidden];
        decV        = new float[hidden];
        decAttnOut  = new float[hidden];
        decOProj    = new float[hidden];
        decResidual = new float[hidden];
        decPostNorm = new float[hidden];
        decGateUp   = new float[interX2];
        decMlpAct   = new float[inter];
        decDown     = new float[hidden];
        decScores   = new float[maxPos];
        decLogits   = new float[config.vocabSize()];

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

    // ── Generation quality parameters ────────────────────────────────────

    /**
     * Penalty factor for tokens already in the generated output (1.0 = off).
     * Positive logits are divided by the penalty, negative logits multiplied.
     * Typical values: 1.1 – 1.3.  This is a standard quality improvement for
     * greedy decoding — it reduces monotonous repetitions.
     */
    private float repetitionPenalty = 1.2f;

    /**
     * Maximum number of consecutive identical tokens before generation is
     * stopped early.  This is a safety brake for stuck loops, especially
     * important when {@code maxTokens = 0} (unlimited).  Use 0 to disable.
     */
    private int maxConsecutiveRepeats = 16;

    public void setRepetitionPenalty(float penalty) {
        this.repetitionPenalty = Math.max(1.0f, penalty);
    }

    public void setMaxConsecutiveRepeats(int max) {
        this.maxConsecutiveRepeats = max;
    }

    /**
     * Generate tokens greedily with a per-token streaming callback.
     * <p>
     * Token IDs are accumulated and decoded as a full sequence after each
     * step so that SentencePiece inter-token spaces are preserved correctly.
     * <p>
     * Includes a <b>repetition penalty</b> (standard quality improvement for
     * greedy decoding) and a <b>stuck-loop breaker</b> (safety net when the
     * model enters a degenerate loop).
     *
     * @param prompt    text prompt
     * @param maxTokens maximum number of tokens to generate
     * @param consumer  optional callback invoked after each token (may be {@code null})
     * @return generated text (excluding the prompt)
     */
    public String generateStreaming(String prompt, int maxTokens, TokenConsumer consumer) {
        int[] inputIds = tokenizer.encode(prompt);

        resetProfile();

        // ── Incremental prefill: find common prefix with cached tokens ──
        int commonPrefix = 0;
        if (cachedTokenIds != null) {
            int limit = Math.min(cachedTokenIds.length, inputIds.length);
            while (commonPrefix < limit && cachedTokenIds[commonPrefix] == inputIds[commonPrefix]) {
                commonPrefix++;
            }
        }

        // Trim cache to the common prefix (invalidate anything after it)
        cachedSeqLen = commonPrefix;

        int newTokenCount = inputIds.length - commonPrefix;
        log.info("Prompt: {} tokens (cached={}, new={})", inputIds.length, commonPrefix, newTokenCount);

        // Prefill only the NEW suffix tokens
        long t0 = System.nanoTime();
        float[] logits;
        if (newTokenCount > 0) {
            int[] suffixIds = new int[newTokenCount];
            System.arraycopy(inputIds, commonPrefix, suffixIds, 0, newTokenCount);
            logits = prefill(suffixIds, commonPrefix);
        } else {
            // Entire prompt is already cached — recompute logits for last position only
            logits = recomputeLastLogits(inputIds[inputIds.length - 1]);
        }
        profPrefillNs = System.nanoTime() - t0;
        profPrefillTokens = newTokenCount;

        // Track what's in the cache now (= the full prompt)
        cachedTokenIds = Arrays.copyOf(inputIds, inputIds.length);

        // ── Decode loop ──────────────────────────────────────────────
        List<Integer> generatedIds = new ArrayList<>();
        String previousText = "";
        int consecutiveCount = 0;
        int lastTokenId = -1;

        for (int step = 0; step < maxTokens; step++) {

            // ── Repetition penalty (quality improvement for greedy decoding) ─
            if (repetitionPenalty > 1.0f && !generatedIds.isEmpty()) {
                applyRepetitionPenalty(logits, generatedIds);
            }

            int nextToken = argmax(logits);

            if (tokenizer.isEos(nextToken)) {
                break;
            }

            // ── Stuck-loop detection (safety brake) ─────────────────
            if (nextToken == lastTokenId) {
                consecutiveCount++;
                if (maxConsecutiveRepeats > 0 && consecutiveCount >= maxConsecutiveRepeats) {
                    log.warn("Stuck loop: token {} repeated {} times — stopping",
                            nextToken, consecutiveCount);
                    break;
                }
            } else {
                consecutiveCount = 1;
                lastTokenId = nextToken;
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

            // Decode: process single token using pre-allocated buffers
            logits = decodeFast(nextToken);
            profSteps++;

            // Extend cached token tracking
            cachedTokenIds = Arrays.copyOf(cachedTokenIds, cachedTokenIds.length + 1);
            cachedTokenIds[cachedTokenIds.length - 1] = nextToken;
        }

        lastProfile = buildProfileSummary();
        log.debug("Profile: {}", lastProfile);

        return previousText;
    }

    /**
     * Apply repetition penalty to logits for all tokens that have already
     * been generated.  Positive logits are divided by the penalty, negative
     * logits are multiplied — so repeated tokens are always pushed down.
     */
    private void applyRepetitionPenalty(float[] logits, List<Integer> generatedIds) {
        boolean[] seen = new boolean[logits.length];
        for (int id : generatedIds) {
            if (id >= 0 && id < logits.length && !seen[id]) {
                seen[id] = true;
                if (logits[id] > 0) {
                    logits[id] /= repetitionPenalty;
                } else {
                    logits[id] *= repetitionPenalty;
                }
            }
        }
    }

    // ── Profiling ────────────────────────────────────────────────────────

    private void resetProfile() {
        profGpuProjNs = profCpuAttnNs = profCpuNormNs = profCpuActNs = 0;
        profLmHeadNs = profTokenDecNs = profPrefillNs = 0;
        profSteps = 0;
        profPrefillTokens = 0;
        lastProfile = null;
    }

    private String buildProfileSummary() {
        if (profSteps == 0) return "[No tokens generated]";
        long totalDecode = profGpuProjNs + profCpuAttnNs + profCpuNormNs + profCpuActNs + profLmHeadNs;
        double totalMs = totalDecode / 1e6;
        double perToken = totalMs / profSteps;
        double pctDivisor = totalDecode > 0 ? totalDecode : 1;

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[Decode Profile] %d tokens, %.1f ms total, %.1f ms/token%n", profSteps, totalMs, perToken));
        sb.append(String.format("  Prefill:         %.1f ms (%d new tokens, %d cached)%n",
                profPrefillNs / 1e6, profPrefillTokens, cachedSeqLen - profSteps - profPrefillTokens));
        sb.append(String.format("  GPU projections: %.1f ms avg (%.0f%%)%n",
                profGpuProjNs / 1e6 / profSteps, 100.0 * profGpuProjNs / pctDivisor));
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
        cachedTokenIds = new int[0];
        Arrays.fill(kvCache, null);
    }

    // ── Prefill (supports incremental via startPos) ──────────────────────

    /**
     * Process multiple tokens (prefill phase) starting at a given position.
     * Positions [0..startPos-1] are assumed already in the KV cache.
     */
    private float[] prefill(int[] tokenIds, int startPos) {
        int seqLen = tokenIds.length;
        int hidden = config.hiddenSize();

        // Embedding lookup
        float[] hidden_states = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            System.arraycopy(weights.embedTokens, tokenIds[s] * hidden, hidden_states, s * hidden, hidden);
        }

        // Process each layer
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            hidden_states = processLayer(l, hidden_states, seqLen, startPos);
        }

        // Final norm + logits (only for last position)
        float[] lastHidden = new float[hidden];
        System.arraycopy(hidden_states, (seqLen - 1) * hidden, lastHidden, 0, hidden);
        rmsNorm(lastHidden, weights.finalNormWeight, config.rmsNormEps());

        float[] logits = new float[config.vocabSize()];
        lmHeadMatvec(lastHidden, logits);

        cachedSeqLen = startPos + seqLen;
        return logits;
    }

    /**
     * Recompute logits for the last cached position (when full prompt is cached).
     * This is a lightweight single-token forward pass.
     */
    private float[] recomputeLastLogits(int lastTokenId) {
        return decodeFast(lastTokenId);
    }

    // ── Fast single-token decode (pre-allocated buffers) ─────────────────

    /**
     * Process a single new token using pre-allocated buffers.
     * Zero-allocation hot path (except GPU matvec returns).
     */
    private float[] decodeFast(int tokenId) {
        int hidden = config.hiddenSize();
        int pos = cachedSeqLen;

        // Embedding lookup → decBuf
        System.arraycopy(weights.embedTokens, tokenId * hidden, decBuf, 0, hidden);

        // Process each layer
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            processLayerDecode(l, decBuf, pos);
        }

        // Final norm
        long t0 = System.nanoTime();
        rmsNorm(decBuf, weights.finalNormWeight, config.rmsNormEps());
        profCpuNormNs += System.nanoTime() - t0;

        // LM head → decLogits
        t0 = System.nanoTime();
        lmHeadMatvec(decBuf, decLogits);
        profLmHeadNs += System.nanoTime() - t0;

        cachedSeqLen = pos + 1;
        return decLogits;
    }

    // ── LM head ──────────────────────────────────────────────────────────

    private void lmHeadMatvec(float[] x, float[] logits) {
        if (gpuKernels != null && gpuKernels.hasLmHead()) {
            float[] result = gpuKernels.lmHead().matvec(x);
            System.arraycopy(result, 0, logits, 0, result.length);
        } else {
            Arrays.fill(logits, 0);  // zero before matvec (accumulates via +=)
            weights.lmHead.matvec(x, logits);
        }
    }

    // ── Zero-alloc single-token layer processing ─────────────────────────

    /**
     * Process one decoder layer for a single token (decode phase).
     * Uses pre-allocated buffers — zero heap allocation in the hot path.
     * <p>
     * Buffer assignments:
     * <pre>
     *   decOProj  → normed input (temporary, reused later for O projection)
     *   decQ      → Q projection result
     *   decK      → K projection result
     *   decV      → V projection result (temporary, stored into KV cache)
     *   decScores → attention scores [pos+1]
     *   decAttnOut→ attention output
     *   decOProj  → O projection result (reuses normed buffer)
     *   decResidual → residual1
     *   decPostNorm → post-attention norm
     *   decGateUp → gate+up projection
     *   decMlpAct → SwiGLU activation
     *   decDown   → down projection
     * </pre>
     *
     * @param layerIdx  layer index
     * @param hidden_io input/output hidden state [hidden] — modified in-place
     * @param pos       absolute position for this token
     */
    private void processLayerDecode(int layerIdx, float[] hidden_io, int pos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        int cacheRowSize = kvHeads * headDim;
        LayerWeights lw = weights.layers[layerIdx];
        boolean gpuLayer = gpuKernels != null && gpuKernels.hasLayer(layerIdx);
        long t0;

        // ── Pre-attention RMSNorm → decOProj (as "normed" temp) ──────
        t0 = System.nanoTime();
        System.arraycopy(hidden_io, 0, decOProj, 0, hidden);
        rmsNorm(decOProj, lw.inputNormWeight(), config.rmsNormEps());
        // decOProj now holds the normed input for Q/K/V projections
        profCpuNormNs += System.nanoTime() - t0;

        // ── Q/K/V Projections (all read from decOProj = normed) ──────
        // IMPORTANT: QuantizedWeight.matvec() ACCUMULATES (y[n] += sum),
        // so output buffers must be zeroed before each call.
        t0 = System.nanoTime();
        Arrays.fill(decQ, 0);
        Arrays.fill(decK, 0);
        Arrays.fill(decV, 0);
        if (gpuLayer) {
            float[] qRes = gpuKernels.qProj(layerIdx).matvec(decOProj);
            System.arraycopy(qRes, 0, decQ, 0, hidden);
            float[] kRes = gpuKernels.kProj(layerIdx).matvec(decOProj);
            System.arraycopy(kRes, 0, decK, 0, hidden);
            float[] vRes = gpuKernels.vProj(layerIdx).matvec(decOProj);
            System.arraycopy(vRes, 0, decV, 0, hidden);
        } else {
            lw.qProj().matvec(decOProj, decQ);
            lw.kProj().matvec(decOProj, decK);
            lw.vProj().matvec(decOProj, decV);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── RoPE ─────────────────────────────────────────────────────
        t0 = System.nanoTime();
        for (int h = 0; h < numHeads; h++) {
            applyRoPE(decQ, h * headDim, headDim, pos);
        }
        for (int h = 0; h < kvHeads; h++) {
            applyRoPE(decK, h * headDim, headDim, pos);
        }
        profCpuNormNs += System.nanoTime() - t0;

        // ── KV Cache update ──────────────────────────────────────────
        int kCacheIdx = layerIdx * 2;
        int vCacheIdx = layerIdx * 2 + 1;
        int totalSeq = pos + 1;

        if (kvCache[kCacheIdx] == null || kvCache[kCacheIdx].length < totalSeq * cacheRowSize) {
            int capacity = Math.max(totalSeq, 64) * cacheRowSize;
            float[] newK = new float[capacity];
            float[] newV = new float[capacity];
            if (kvCache[kCacheIdx] != null) {
                System.arraycopy(kvCache[kCacheIdx], 0, newK, 0,
                        Math.min(kvCache[kCacheIdx].length, pos * cacheRowSize));
                System.arraycopy(kvCache[vCacheIdx], 0, newV, 0,
                        Math.min(kvCache[vCacheIdx].length, pos * cacheRowSize));
            }
            kvCache[kCacheIdx] = newK;
            kvCache[vCacheIdx] = newV;
        }

        System.arraycopy(decK, 0, kvCache[kCacheIdx], pos * cacheRowSize, cacheRowSize);
        System.arraycopy(decV, 0, kvCache[vCacheIdx], pos * cacheRowSize, cacheRowSize);

        // ── Causal Self-Attention (reuse decScores, write to decAttnOut) ─
        t0 = System.nanoTime();
        Arrays.fill(decAttnOut, 0, hidden, 0.0f);
        float scale = (float) (1.0 / Math.sqrt(headDim));

        for (int h = 0; h < numHeads; h++) {
            int kvH = h % kvHeads;
            int qOff = h * headDim;

            // Dot products: Q · K for all cached positions
            for (int p = 0; p <= pos; p++) {
                int kOff = p * cacheRowSize + kvH * headDim;
                float dot = 0;
                for (int d = 0; d < headDim; d++) {
                    dot += decQ[qOff + d] * kvCache[kCacheIdx][kOff + d];
                }
                decScores[p] = dot * scale;
            }

            // Softmax over [0..pos]
            softmax(decScores, pos + 1);

            // Weighted sum of V
            int outOff = h * headDim;
            for (int d = 0; d < headDim; d++) decAttnOut[outOff + d] = 0;
            for (int p = 0; p <= pos; p++) {
                int vOff = p * cacheRowSize + kvH * headDim;
                float w = decScores[p];
                for (int d = 0; d < headDim; d++) {
                    decAttnOut[outOff + d] += w * kvCache[vCacheIdx][vOff + d];
                }
            }
        }
        profCpuAttnNs += System.nanoTime() - t0;

        // ── Activation scale + O projection → decOProj ───────────────
        t0 = System.nanoTime();
        for (int i = 0; i < hidden; i++) {
            decAttnOut[i] *= lw.attnOutScale()[i];
        }
        profCpuNormNs += System.nanoTime() - t0;

        t0 = System.nanoTime();
        Arrays.fill(decOProj, 0);  // zero before matvec (accumulates via +=)
        if (gpuLayer) {
            float[] oRes = gpuKernels.oProj(layerIdx).matvec(decAttnOut);
            System.arraycopy(oRes, 0, decOProj, 0, hidden);
        } else {
            lw.oProj().matvec(decAttnOut, decOProj);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── Residual 1 → decResidual ─────────────────────────────────
        t0 = System.nanoTime();
        for (int i = 0; i < hidden; i++) {
            decResidual[i] = hidden_io[i] + decOProj[i];
        }

        // ── Post-attention RMSNorm → decPostNorm ─────────────────────
        System.arraycopy(decResidual, 0, decPostNorm, 0, hidden);
        rmsNorm(decPostNorm, lw.postNormWeight(), config.rmsNormEps());
        profCpuNormNs += System.nanoTime() - t0;

        // ── MLP: gate_up_proj → decGateUp ────────────────────────────
        t0 = System.nanoTime();
        Arrays.fill(decGateUp, 0);  // zero before matvec (accumulates via +=)
        if (gpuLayer) {
            float[] guRes = gpuKernels.gateUpProj(layerIdx).matvec(decPostNorm);
            System.arraycopy(guRes, 0, decGateUp, 0, guRes.length);
        } else {
            lw.gateUpProj().matvec(decPostNorm, decGateUp);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── SwiGLU activation → decMlpAct ────────────────────────────
        t0 = System.nanoTime();
        int intermediate = config.intermediateSize();
        for (int i = 0; i < intermediate; i++) {
            float gate = decGateUp[i];
            float up = decGateUp[intermediate + i];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-gate));
            decMlpAct[i] = up * gate * sigmoid;
        }
        profCpuActNs += System.nanoTime() - t0;

        // ── Activation scale + down_proj → decDown ───────────────────
        t0 = System.nanoTime();
        for (int i = 0; i < intermediate; i++) {
            decMlpAct[i] *= lw.mlpOutScale()[i];
        }
        profCpuNormNs += System.nanoTime() - t0;

        t0 = System.nanoTime();
        Arrays.fill(decDown, 0);  // zero before matvec (accumulates via +=)
        if (gpuLayer) {
            float[] dRes = gpuKernels.downProj(layerIdx).matvec(decMlpAct);
            System.arraycopy(dRes, 0, decDown, 0, hidden);
        } else {
            lw.downProj().matvec(decMlpAct, decDown);
        }
        profGpuProjNs += System.nanoTime() - t0;

        // ── Residual 2 → hidden_io (output, in-place) ───────────────
        for (int i = 0; i < hidden; i++) {
            hidden_io[i] = decResidual[i] + decDown[i];
        }
    }

    // ── Prefill layer processing (allocates per call, used only during prefill) ──

    /**
     * Process one decoder layer for multiple tokens (prefill phase).
     */
    private float[] processLayer(int layerIdx, float[] input, int seqLen, int startPos) {
        int hidden = config.hiddenSize();
        int numHeads = config.numAttentionHeads();
        int headDim = config.headDim();
        int kvHeads = config.numKeyValueHeads();
        LayerWeights lw = weights.layers[layerIdx];
        boolean gpuLayer = gpuKernels != null && gpuKernels.hasLayer(layerIdx);

        // ── Pre-attention RMSNorm ────────────────────────────────────
        float[] normed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(input, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.inputNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, normed, s * hidden, hidden);
        }

        // ── Q/K/V Projections ────────────────────────────────────────
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

        // ── RoPE ─────────────────────────────────────────────────────
        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numHeads; h++) {
                applyRoPE(q, s * hidden + h * headDim, headDim, pos);
            }
            for (int h = 0; h < kvHeads; h++) {
                applyRoPE(k, s * hidden + h * headDim, headDim, pos);
            }
        }

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

        // ── Activation scale + O projection ──────────────────────────
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < hidden; i++) {
                attnOut[s * hidden + i] *= lw.attnOutScale()[i];
            }
        }

        float[] oProjOut = new float[seqLen * hidden];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.oProj(layerIdx), attnOut, oProjOut, seqLen, hidden, hidden);
        } else {
            lw.oProj().matmul(attnOut, oProjOut, seqLen);
        }

        // ── Residual 1 + Post-attention RMSNorm ─────────────────────
        float[] residual1 = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            residual1[i] = input[i] + oProjOut[i];
        }

        float[] postNormed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(residual1, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.postNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, postNormed, s * hidden, hidden);
        }

        // ── MLP: gate_up_proj ────────────────────────────────────────
        int intermediateX2 = config.intermediateSize() * 2;
        float[] gateUp = new float[seqLen * intermediateX2];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.gateUpProj(layerIdx), postNormed, gateUp,
                    seqLen, hidden, intermediateX2);
        } else {
            lw.gateUpProj().matmul(postNormed, gateUp, seqLen);
        }

        // ── SwiGLU activation ────────────────────────────────────────
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

        // ── Activation scale + down_proj ─────────────────────────────
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < intermediate; i++) {
                mlpActivation[s * intermediate + i] *= lw.mlpOutScale()[i];
            }
        }

        float[] downOut = new float[seqLen * hidden];
        if (gpuLayer) {
            gpuMatmul(gpuKernels.downProj(layerIdx), mlpActivation, downOut,
                    seqLen, intermediate, hidden);
        } else {
            lw.downProj().matmul(mlpActivation, downOut, seqLen);
        }

        // ── Residual 2 ──────────────────────────────────────────────
        float[] output = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            output[i] = residual1[i] + downOut[i];
        }

        return output;
    }

    // ── GPU matmul helper ─────────────────────────────────────────────────

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

    static void rmsNorm(float[] x, float[] weight, float eps) {
        float sumSq = 0;
        for (float v : x) sumSq += v * v;
        float rms = (float) (1.0 / Math.sqrt(sumSq / x.length + eps));
        for (int i = 0; i < x.length; i++) {
            x[i] = x[i] * rms * weight[i];
        }
    }

    private void applyRoPE(float[] vec, int offset, int dim, int pos) {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++) {
            float cos = weights.cosCache[pos * halfDim + i];
            float sin = weights.sinCache[pos * halfDim + i];
            float x0 = vec[offset + i];
            float x1 = vec[offset + halfDim + i];
            vec[offset + i] = x0 * cos - x1 * sin;
            vec[offset + halfDim + i] = x0 * sin + x1 * cos;
        }
    }

    /**
     * In-place softmax over the first {@code len} elements of the array.
     */
    static void softmax(float[] x, int len) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) if (x[i] > max) max = x[i];
        float sum = 0;
        for (int i = 0; i < len; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < len; i++) x[i] *= invSum;
    }

    /**
     * In-place softmax over the entire array.
     */
    static void softmax(float[] x) {
        softmax(x, x.length);
    }

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
