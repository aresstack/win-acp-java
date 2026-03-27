package com.aresstack.winacp.windows;

/**
 * Exception thrown by the ONNX Runtime bridge layer.
 * <p>
 * Translates ORT-specific errors into a generic exception that
 * does not leak ONNX Runtime types into the inference layer.
 */
public class OnnxRuntimeBridgeException extends Exception {

    public OnnxRuntimeBridgeException(String message) {
        super(message);
    }

    public OnnxRuntimeBridgeException(String message, Throwable cause) {
        super(message, cause);
    }
}

