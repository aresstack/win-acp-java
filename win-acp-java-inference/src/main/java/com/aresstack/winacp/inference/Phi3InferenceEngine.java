package com.aresstack.winacp.inference;

import com.aresstack.winacp.config.InferenceConfiguration;
import com.aresstack.winacp.inference.phi3.Phi3Config;
import com.aresstack.winacp.inference.phi3.Phi3Runtime;
import com.aresstack.winacp.inference.phi3.Phi3Tokenizer;
import com.aresstack.winacp.inference.phi3.Phi3Weights;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * {@link InferenceEngine} implementation for Phi-3-mini-4k-instruct.
 * <p>
 * Loads the Phi-3 model from the DirectML INT4 AWQ variant and runs
 * text generation using the CPU-first decoder runtime ({@link Phi3Runtime}).
 * <p>
 * <b>Model requirements:</b>
 * <ul>
 *   <li>{@code config.json} — HuggingFace model config</li>
 *   <li>{@code tokenizer.json} — HuggingFace tokenizer</li>
 *   <li>{@code model.onnx} — ONNX model proto (weight metadata)</li>
 *   <li>{@code model.onnx.data} — External weight data (~2.1 GB)</li>
 * </ul>
 * <p>
 * <b>V1 constraints:</b>
 * <ul>
 *   <li>CPU-only decode (correctness first, GPU kernel proven separately)</li>
 *   <li>Greedy decoding, no sampling</li>
 *   <li>Batch size 1</li>
 * </ul>
 */
public class Phi3InferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(Phi3InferenceEngine.class);

    private final Path modelDir;
    private final int defaultMaxTokens;

    private Phi3Config config;
    private Phi3Tokenizer tokenizer;
    private Phi3Weights weights;
    private Phi3Runtime runtime;
    private boolean ready = false;

    /**
     * @param modelDir path to the directory containing model files
     */
    public Phi3InferenceEngine(Path modelDir) {
        this(modelDir, 256);
    }

    /**
     * @param modelDir         path to the directory containing model files
     * @param defaultMaxTokens default maximum tokens to generate if not specified in request
     */
    public Phi3InferenceEngine(Path modelDir, int defaultMaxTokens) {
        this.modelDir = modelDir;
        this.defaultMaxTokens = defaultMaxTokens;
    }

    /**
     * Create from {@link InferenceConfiguration}.
     * The modelPath should point to the directory containing the Phi-3 model files.
     */
    public Phi3InferenceEngine(InferenceConfiguration config) {
        this(Path.of(config.getModelPath()),
                config.getMaxTokens() > 0 ? config.getMaxTokens() : 256);
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("Phi3InferenceEngine initializing from {}", modelDir);

        try {
            long t0 = System.currentTimeMillis();

            // Load config
            Path configPath = modelDir.resolve("config.json");
            config = Phi3Config.load(configPath);
            log.info("Config: hidden={}, layers={}, heads={}, vocab={}",
                    config.hiddenSize(), config.numHiddenLayers(),
                    config.numAttentionHeads(), config.vocabSize());

            // Load tokenizer
            Path tokenizerPath = modelDir.resolve("tokenizer.json");
            tokenizer = Phi3Tokenizer.load(tokenizerPath);
            log.info("Tokenizer loaded: vocabSize={}", tokenizer.vocabSize());

            // Load weights (memory-maps the 2.1GB external data file)
            weights = Phi3Weights.load(modelDir, config);
            log.info("Weights loaded in {} ms", System.currentTimeMillis() - t0);

            // Create runtime
            runtime = new Phi3Runtime(config, weights, tokenizer);

            ready = true;
            log.info("Phi3InferenceEngine ready ({}ms total)", System.currentTimeMillis() - t0);

        } catch (IOException e) {
            throw new InferenceException("Failed to load Phi-3 model from " + modelDir, e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) throw new InferenceException("Engine not initialized");

        try {
            // Format the prompt using Phi-3 chat template
            String systemPrompt = request.getSystemPrompt();
            String userPrompt = request.getUserPrompt();
            String formattedPrompt = tokenizer.formatChat(
                    systemPrompt.isBlank() ? null : systemPrompt,
                    userPrompt
            );

            int maxTokens = request.getMaxTokens() > 0 ? request.getMaxTokens() : defaultMaxTokens;
            log.info("Generating: maxTokens={}, promptLen={}", maxTokens, formattedPrompt.length());

            long t0 = System.currentTimeMillis();
            String generatedText = runtime.generate(formattedPrompt, maxTokens);
            long elapsed = System.currentTimeMillis() - t0;

            // Count tokens for usage
            int promptTokens = tokenizer.encode(formattedPrompt).length;
            int completionTokens = tokenizer.encode(generatedText).length;

            log.info("Generated {} chars ({} tokens) in {} ms ({} ms/token)",
                    generatedText.length(), completionTokens, elapsed,
                    completionTokens > 0 ? elapsed / completionTokens : 0);

            String finishReason = completionTokens >= maxTokens ? "max_tokens" : "end_turn";

            return new InferenceResult(
                    generatedText.strip(),
                    finishReason,
                    new InferenceResult.Usage(promptTokens, completionTokens,
                            promptTokens + completionTokens)
            );

        } catch (Exception e) {
            throw new InferenceException("Phi-3 generation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        ready = false;
        if (weights != null) {
            try {
                weights.close();
            } catch (Exception e) {
                log.warn("Error closing Phi3Weights: {}", e.getMessage());
            }
            weights = null;
        }
        runtime = null;
        tokenizer = null;
        config = null;
        log.info("Phi3InferenceEngine shut down");
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    /**
     * Check whether a directory contains a valid Phi-3 model.
     * Looks for the 4 required files.
     */
    public static boolean isValidModelDir(Path dir) {
        return dir != null
                && Files.isDirectory(dir)
                && Files.exists(dir.resolve("config.json"))
                && Files.exists(dir.resolve("tokenizer.json"))
                && Files.exists(dir.resolve("model.onnx"))
                && Files.exists(dir.resolve("model.onnx.data"));
    }
}

