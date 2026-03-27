package com.aresstack.winacp.windows;

import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Map;

/**
 * Internal bridge to ONNX Runtime.
 * <p>
 * Encapsulates all ORT-specific types so they do not leak into the
 * inference or graph layers. This is the single place where
 * {@code ai.onnxruntime.*} types are used.
 * <p>
 * V1 supports CPU and DirectML execution providers.
 * Future versions may add raw FFM / jextract-based DirectML bindings.
 */
public final class OnnxRuntimeBridge implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(OnnxRuntimeBridge.class);

    private final OrtEnvironment env;
    private OrtSession session;
    private String backend;

    public OnnxRuntimeBridge() {
        this.env = OrtEnvironment.getEnvironment("win-acp-java");
    }

    /**
     * Load an ONNX model and create an inference session.
     *
     * @param modelPath absolute path to the .onnx model file
     * @param backend   "directml", "cpu", or "auto"
     */
    public void loadModel(Path modelPath, String backend) throws OnnxRuntimeBridgeException {
        this.backend = backend;
        log.info("Loading ONNX model: {} (backend={})", modelPath, backend);

        try {
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            configureBackend(options, backend);

            session = env.createSession(modelPath.toString(), options);

            log.info("ONNX model loaded: {} input(s), {} output(s)",
                    session.getInputInfo().size(), session.getOutputInfo().size());

            // Log input/output shapes for debugging
            for (Map.Entry<String, NodeInfo> entry : session.getInputInfo().entrySet()) {
                log.debug("  Input '{}': {}", entry.getKey(), entry.getValue().getInfo());
            }
            for (Map.Entry<String, NodeInfo> entry : session.getOutputInfo().entrySet()) {
                log.debug("  Output '{}': {}", entry.getKey(), entry.getValue().getInfo());
            }

        } catch (OrtException e) {
            throw new OnnxRuntimeBridgeException("Failed to load model: " + modelPath, e);
        }
    }

    /**
     * Run a forward pass with the given token IDs.
     *
     * @param inputName  name of the input tensor (e.g. "input_ids")
     * @param tokenIds   input token IDs as long array
     * @param shape      tensor shape (e.g. {1, sequenceLength})
     * @return raw output values from the model
     */
    public float[][] runForward(String inputName, long[] tokenIds, long[] shape)
            throws OnnxRuntimeBridgeException {
        if (session == null) {
            throw new OnnxRuntimeBridgeException("No model loaded – call loadModel() first");
        }
        try {
            OnnxTensor inputTensor = OnnxTensor.createTensor(env,
                    LongBuffer.wrap(tokenIds), shape);

            try (OrtSession.Result result = session.run(Map.of(inputName, inputTensor))) {
                // Get first output tensor
                Object outputValue = result.get(0).getValue();

                if (outputValue instanceof float[][] f2d) {
                    return f2d;
                }
                if (outputValue instanceof float[][][] f3d) {
                    // For LLM logits: [batch, seq, vocab] → return last token's logits
                    return new float[][]{f3d[0][f3d[0].length - 1]};
                }

                throw new OnnxRuntimeBridgeException(
                        "Unexpected output type: " + outputValue.getClass().getName());
            } finally {
                inputTensor.close();
            }
        } catch (OrtException e) {
            throw new OnnxRuntimeBridgeException("Forward pass failed", e);
        }
    }

    /**
     * Run a forward pass with multiple named inputs.
     */
    public float[][] runForward(Map<String, OnnxTensor> inputs) throws OnnxRuntimeBridgeException {
        if (session == null) {
            throw new OnnxRuntimeBridgeException("No model loaded – call loadModel() first");
        }
        try (OrtSession.Result result = session.run(inputs)) {
            Object outputValue = result.get(0).getValue();

            if (outputValue instanceof float[][] f2d) {
                return f2d;
            }
            if (outputValue instanceof float[][][] f3d) {
                return new float[][]{f3d[0][f3d[0].length - 1]};
            }

            throw new OnnxRuntimeBridgeException(
                    "Unexpected output type: " + outputValue.getClass().getName());
        } catch (OrtException e) {
            throw new OnnxRuntimeBridgeException("Forward pass failed", e);
        }
    }

    /**
     * Create an ONNX tensor from a long array (for token IDs).
     */
    public OnnxTensor createLongTensor(long[] data, long[] shape) throws OnnxRuntimeBridgeException {
        try {
            return OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape);
        } catch (OrtException e) {
            throw new OnnxRuntimeBridgeException("Failed to create tensor", e);
        }
    }

    /** Get the input names of the loaded model. */
    public java.util.Set<String> getInputNames() throws OnnxRuntimeBridgeException {
        if (session == null) throw new OnnxRuntimeBridgeException("No model loaded");
        try {
            return session.getInputInfo().keySet();
        } catch (OrtException e) {
            throw new OnnxRuntimeBridgeException("Failed to get input names", e);
        }
    }

    /** Get the output names of the loaded model. */
    public java.util.Set<String> getOutputNames() throws OnnxRuntimeBridgeException {
        if (session == null) throw new OnnxRuntimeBridgeException("No model loaded");
        try {
            return session.getOutputInfo().keySet();
        } catch (OrtException e) {
            throw new OnnxRuntimeBridgeException("Failed to get output names", e);
        }
    }

    public boolean isLoaded() {
        return session != null;
    }

    public String getBackend() {
        return backend;
    }

    @Override
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                log.warn("Error closing ONNX session", e);
            }
            session = null;
        }
        log.info("OnnxRuntimeBridge closed");
    }

    // ---- internal ----

    private void configureBackend(OrtSession.SessionOptions options, String backend)
            throws OrtException {
        switch (backend.toLowerCase()) {
            case "directml" -> {
                if (WindowsBindings.isSupported()) {
                    try {
                        options.addDirectML(0);
                        log.info("DirectML execution provider enabled (device 0)");
                    } catch (OrtException e) {
                        log.warn("DirectML not available, falling back to CPU: {}", e.getMessage());
                    }
                } else {
                    log.warn("DirectML requested but not on Windows – using CPU");
                }
            }
            case "cpu" -> log.info("Using CPU execution provider");
            case "auto" -> {
                if (WindowsBindings.isSupported()) {
                    try {
                        options.addDirectML(0);
                        log.info("Auto: DirectML execution provider enabled");
                    } catch (OrtException e) {
                        log.info("Auto: DirectML not available, using CPU");
                    }
                } else {
                    log.info("Auto: not on Windows, using CPU");
                }
            }
            default -> log.warn("Unknown backend '{}' – using CPU", backend);
        }
    }
}

