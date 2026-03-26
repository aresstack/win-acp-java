package com.aresstack.winacp.inference;

/**
 * Exception thrown by the inference layer.
 */
public class InferenceException extends Exception {

    public InferenceException(String message) {
        super(message);
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}

