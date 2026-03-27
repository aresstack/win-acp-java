package com.aresstack.winacp.inference;

import com.aresstack.winacp.config.InferenceConfiguration;
import com.aresstack.winacp.windows.MnistPipeline;
import com.aresstack.winacp.windows.WindowsBindings;
import com.aresstack.winacp.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;

/**
 * Inference engine backed by the Windows native stack (DXGI → D3D12 → DirectML).
 * <p>
 * Loads {@code mnist-8.onnx} via the {@link MnistPipeline} which executes
 * entirely on the GPU through DirectML operator dispatches.
 * <p>
 * <b>No third-party inference runtime</b> – pure FFM → Windows DLLs.
 */
public class DirectMlInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(DirectMlInferenceEngine.class);

    private final InferenceConfiguration config;
    private WindowsBindings bindings;
    private MnistPipeline pipeline;
    private boolean ready = false;

    public DirectMlInferenceEngine(InferenceConfiguration config) {
        this.config = config;
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("DirectMlInferenceEngine initializing (backend={}, model={})",
                config.getBackend(), config.getModelPath());

        try {
            // 1. Bring up the Windows native stack
            bindings = new WindowsBindings();
            bindings.init(config.getBackend());

            // 2. Load the MNIST model through the DirectML pipeline
            pipeline = new MnistPipeline(bindings);
            pipeline.loadModel(Path.of(config.getModelPath()));

            ready = true;
            log.info("DirectMlInferenceEngine ready – MNIST model loaded via DirectML");

        } catch (WindowsNativeException e) {
            throw new InferenceException("Failed to initialize DirectML engine: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new InferenceException("Failed to load model: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) throw new InferenceException("Engine not initialized");

        log.debug("DirectMlInferenceEngine.generate: {}", request);

        try {
            // For MNIST: create a test input (28x28 zeros or parse from prompt)
            float[] input = new float[784]; // default: zeros
            String userPrompt = request.getUserPrompt();

            // If the prompt contains comma-separated floats, parse them as pixel data
            if (userPrompt != null && userPrompt.contains(",")) {
                try {
                    String[] parts = userPrompt.trim().split("[,\\s]+");
                    if (parts.length == 784) {
                        for (int i = 0; i < 784; i++) input[i] = Float.parseFloat(parts[i]);
                    }
                } catch (NumberFormatException ignore) { /* use zeros */ }
            }

            // Run inference through DirectML
            float[] logits = pipeline.infer(input);
            int predicted = MnistPipeline.argmax(logits);

            String resultText = String.format(
                    "MNIST prediction: digit %d (logits: %s)",
                    predicted, Arrays.toString(logits));

            int promptTokens = request.toFullPrompt().split("\\s+").length;
            return new InferenceResult(resultText, "end_turn",
                    new InferenceResult.Usage(promptTokens, 1, promptTokens + 1));

        } catch (WindowsNativeException e) {
            throw new InferenceException("DirectML inference failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void shutdown() {
        ready = false;
        if (pipeline != null) { pipeline.close(); pipeline = null; }
        if (bindings != null) { bindings.close(); bindings = null; }
        log.info("DirectMlInferenceEngine shut down");
    }

    @Override
    public boolean isReady() {
        return ready;
    }
}
