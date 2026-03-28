package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.winacp.inference.phi3.Phi3Weights.QuantizedWeight;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * CPU-first decoder runtime for Phi-3-mini-4k-instruct.
 *
 * <p>Implements the full Phi-3 decoder stack:
 * <ol>
 *   <li>Token embedding lookup</li>
 *   <li>Per-layer: RMSNorm, Q/K/V projection, RoPE, causal attention with KV-cache,
 *       O projection, residual, RMSNorm, SwiGLU MLP, residual</li>
 *   <li>Final RMSNorm + LM head logits</li>
 * </ol>
 *
 * <p>V1 constraints:
 * <ul>
 *   <li>All computation on CPU (correctness first, GPU later)</li>
 *   <li>Greedy decoding only (no sampling)</li>
 *   <li>Single-batch (batch_size=1)</li>
 * </ul>
 */
public final class Phi3Runtime {

    private static final Logger log = LoggerFactory.getLogger(Phi3Runtime.class);

    private final Phi3Config config;
    private final Phi3Weights weights;
    private final Phi3Tokenizer tokenizer;

    // KV cache: [layer][2 (k/v)][head][position][headDim]
    // Stored flat as: [numLayers][2][maxPos * numHeads * headDim]
    private final float[][] kvCache; // [numLayers * 2][allocated]
    private int cachedSeqLen;        // number of positions in cache

    public Phi3Runtime(Phi3Config config, Phi3Weights weights, Phi3Tokenizer tokenizer) {
        this.config = config;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.kvCache = new float[config.numHiddenLayers() * 2][];
        this.cachedSeqLen = 0;
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Generate tokens greedily from a prompt.
     *
     * @param prompt    text prompt (will be formatted with chat template if system/user provided)
     * @param maxTokens maximum number of tokens to generate
     * @return generated text (excluding the prompt)
     */
    public String generate(String prompt, int maxTokens) {
        log.info("Encoding prompt ({} chars)", prompt.length());
        int[] inputIds = tokenizer.encode(prompt);
        log.info("Prompt tokens: {}", inputIds.length);

        // Reset KV cache
        resetCache();

        // Prefill: process all prompt tokens at once
        log.info("Prefill: {} tokens", inputIds.length);
        float[] logits = prefill(inputIds);

        // Decode loop
        StringBuilder result = new StringBuilder();
        for (int step = 0; step < maxTokens; step++) {
            int nextToken = argmax(logits);

            if (tokenizer.isEos(nextToken)) {
                log.info("EOS token {} at step {}", nextToken, step);
                break;
            }

            String tokenStr = tokenizer.decode(new int[]{nextToken});
            result.append(tokenStr);

            if (step < 10 || step % 50 == 0) {
                log.debug("Step {}: token={} '{}'", step, nextToken, tokenStr.trim());
            }

            // Decode: process single token
            logits = decode(nextToken);
        }

        return result.toString();
    }

    /**
     * Reset the KV cache (e.g. for a new conversation).
     */
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
            int tokenId = tokenIds[s];
            System.arraycopy(weights.embedTokens, tokenId * hidden, hidden_states, s * hidden, hidden);
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
        weights.lmHead.matvec(lastHidden, logits);

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
        float[] buf = new float[1 * hidden];
        System.arraycopy(hidden_states, 0, buf, 0, hidden);
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            buf = processLayer(l, buf, 1, pos);
        }
        System.arraycopy(buf, 0, hidden_states, 0, hidden);

        // Final norm + logits
        rmsNorm(hidden_states, weights.finalNormWeight, config.rmsNormEps());

        float[] logits = new float[weights.lmHead.N()];
        weights.lmHead.matvec(hidden_states, logits);

        cachedSeqLen = pos + 1;
        return logits;
    }

    // ── Layer processing ─────────────────────────────────────────────────

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

        // ── Pre-attention RMSNorm ────────────────────────────────────
        float[] normed = new float[seqLen * hidden];
        for (int s = 0; s < seqLen; s++) {
            float[] row = new float[hidden];
            System.arraycopy(input, s * hidden, row, 0, hidden);
            rmsNorm(row, lw.inputNormWeight(), config.rmsNormEps());
            System.arraycopy(row, 0, normed, s * hidden, hidden);
        }

        // ── Q/K/V Projections ────────────────────────────────────────
        float[] q = new float[seqLen * hidden]; // [seqLen, numHeads * headDim]
        float[] k = new float[seqLen * hidden]; // [seqLen, kvHeads * headDim]
        float[] v = new float[seqLen * hidden]; // [seqLen, kvHeads * headDim]

        lw.qProj().matmul(normed, q, seqLen);
        lw.kProj().matmul(normed, k, seqLen);
        lw.vProj().matmul(normed, v, seqLen);

        // ── RoPE ─────────────────────────────────────────────────────
        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            // Apply RoPE to all Q heads
            for (int h = 0; h < numHeads; h++) {
                int off = s * hidden + h * headDim;
                applyRoPE(q, off, headDim, pos);
            }
            // Apply RoPE to all K heads
            for (int h = 0; h < kvHeads; h++) {
                int off = s * hidden + h * headDim;
                applyRoPE(k, off, headDim, pos);
            }
        }

        // ── KV Cache update ──────────────────────────────────────────
        int kCacheIdx = layerIdx * 2;
        int vCacheIdx = layerIdx * 2 + 1;
        int totalSeq = startPos + seqLen;
        int cacheRowSize = kvHeads * headDim;

        // Ensure cache is allocated
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

        // Write new K/V into cache
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
                // Determine which KV head this query head uses
                int kvH = h % kvHeads; // For Phi-3-mini: kvHeads == numHeads, so kvH == h

                // Q vector for this head
                int qOff = s * hidden + h * headDim;

                // Compute attention scores against all cached K positions [0..queryPos]
                float[] scores = new float[queryPos + 1];
                for (int p = 0; p <= queryPos; p++) {
                    int kOff = p * cacheRowSize + kvH * headDim;
                    float dot = 0;
                    for (int d = 0; d < headDim; d++) {
                        dot += q[qOff + d] * kvCache[kCacheIdx][kOff + d];
                    }
                    scores[p] = dot * scale;
                }

                // Softmax
                softmax(scores);

                // Weighted sum of V vectors
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
        // Apply weight-only quantization input scale
        for (int s = 0; s < seqLen; s++) {
            for (int i = 0; i < hidden; i++) {
                attnOut[s * hidden + i] *= lw.attnOutScale()[i];
            }
        }

        float[] oProjOut = new float[seqLen * hidden];
        lw.oProj().matmul(attnOut, oProjOut, seqLen);

        // ── Residual connection 1 ────────────────────────────────────
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

        // ── MLP: gate_up_proj ────────────────────────────────────────
        int intermediateX2 = config.intermediateSize() * 2;
        float[] gateUp = new float[seqLen * intermediateX2];
        lw.gateUpProj().matmul(postNormed, gateUp, seqLen);

        // ── SwiGLU activation ────────────────────────────────────────
        // gate_up_proj output is [seqLen, 2*intermediate]
        // Split into gate[intermediate] and up[intermediate]
        // SwiGLU: output = up * SiLU(gate) = up * gate * sigmoid(gate)
        int intermediate = config.intermediateSize();
        float[] mlpActivation = new float[seqLen * intermediate];
        for (int s = 0; s < seqLen; s++) {
            int guOff = s * intermediateX2;
            int outOff = s * intermediate;
            for (int i = 0; i < intermediate; i++) {
                float gate = gateUp[guOff + i];
                float up = gateUp[guOff + intermediate + i];
                // SiLU(gate) = gate * sigmoid(gate)
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
        lw.downProj().matmul(mlpActivation, downOut, seqLen);

        // ── Residual connection 2 ────────────────────────────────────
        float[] output = new float[seqLen * hidden];
        for (int i = 0; i < seqLen * hidden; i++) {
            output[i] = residual1[i] + downOut[i];
        }

        return output;
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
