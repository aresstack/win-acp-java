package com.aresstack.winacp.inference;

import com.aresstack.winacp.config.InferenceConfiguration;
import com.aresstack.winacp.windows.OnnxRuntimeBridge;
import com.aresstack.winacp.windows.OnnxRuntimeBridgeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * Real inference engine backed by ONNX Runtime with DirectML (or CPU) provider.
 * <p>
 * V1 implementation: loads an ONNX model, runs a single forward pass,
 * and returns the output. Full auto-regressive LLM generation (tokenizer,
 * KV-cache, sampling loop) is <b>not yet included</b> – that is V2.
 * <p>
 * For V1, this engine proves the vertical slice:
 * <ol>
 *   <li>Model loaded ✓</li>
 *   <li>Session created ✓</li>
 *   <li>Input tensor submitted ✓</li>
 *   <li>Output tensor read ✓</li>
 *   <li>Result mapped to {@link InferenceResult} ✓</li>
 * </ol>
 */
public class DirectMlInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(DirectMlInferenceEngine.class);

    private final InferenceConfiguration config;
    private OnnxRuntimeBridge bridge;
    private boolean ready = false;

    public DirectMlInferenceEngine(InferenceConfiguration config) {
        this.config = config;
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("DirectMlInferenceEngine initializing (backend={}, model={})",
                config.getBackend(), config.getModelPath());

        try {
            bridge = new OnnxRuntimeBridge();
            bridge.loadModel(Path.of(config.getModelPath()), config.getBackend());
            ready = true;

            log.info("DirectMlInferenceEngine ready (inputs={}, outputs={})",
                    bridge.getInputNames(), bridge.getOutputNames());

        } catch (OnnxRuntimeBridgeException e) {
            throw new InferenceException("Failed to initialize DirectML engine: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) {
            throw new InferenceException("Engine not initialized – call initialize() first");
        }

        log.debug("DirectMlInferenceEngine.generate: {}", request);

        try {
            // V1: simple forward pass with the prompt as token IDs
            // In a full LLM pipeline, a tokenizer would convert text → tokens here.
            // For V1, we convert the prompt to a naive byte/char encoding
            // or use pre-tokenized input.
            String fullPrompt = request.toFullPrompt();
            long[] tokenIds = naiveTokenize(fullPrompt, request.getMaxTokens());
            long[] shape = {1, tokenIds.length};

            // Determine input tensor name
            String inputName = bridge.getInputNames().stream()
                    .findFirst()
                    .orElseThrow(() -> new InferenceException("Model has no inputs"));

            float[][] output = bridge.runForward(inputName, tokenIds, shape);

            // V1: convert output logits to text via naive decoding
            // A real tokenizer + sampling loop replaces this in V2.
            String resultText = naiveDecode(output);

            int completionTokens = resultText.split("\\s+").length;
            int promptTokens = tokenIds.length;

            return new InferenceResult(resultText, "end_turn",
                    new InferenceResult.Usage(promptTokens, completionTokens,
                            promptTokens + completionTokens));

        } catch (OnnxRuntimeBridgeException e) {
            throw new InferenceException("Inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        ready = false;
        if (bridge != null) {
            bridge.close();
            bridge = null;
        }
        log.info("DirectMlInferenceEngine shut down");
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    // ---- V1 naive tokenizer/decoder (replaced by real tokenizer in V2) ----

    /**
     * Naive tokenization: each character becomes a token ID.
     * Real implementation will use SentencePiece/BPE tokenizer.
     */
    private long[] naiveTokenize(String text, int maxTokens) {
        int len = Math.min(text.length(), maxTokens);
        long[] tokens = new long[len];
        for (int i = 0; i < len; i++) {
            tokens[i] = text.charAt(i);
        }
        return tokens;
    }

    /**
     * Naive decoding: interpret output logits as character codes.
     * Real implementation will use vocabulary-based decoding.
     */
    private String naiveDecode(float[][] output) {
        if (output == null || output.length == 0 || output[0].length == 0) {
            return "(empty model output)";
        }

        // For V1: return a description of the output shape
        // A full decoder would argmax over vocab dimensions
        StringBuilder sb = new StringBuilder();
        float[] logits = output[0];

        // Find top-5 indices as a diagnostic
        sb.append("[DirectML output: ").append(logits.length).append(" logit values, ");
        float max = Float.NEGATIVE_INFINITY;
        int maxIdx = 0;
        for (int i = 0; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
                maxIdx = i;
            }
        }
        sb.append("argmax=").append(maxIdx).append("]");

        return sb.toString();
    }
}

