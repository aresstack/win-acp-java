package com.aresstack.winacp.inference;

/**
 * Abstraction over the local inference backend.
 * <p>
 * The rest of the system consumes inference through this interface only.
 * The concrete implementation (DirectML, CPU, …) is hidden behind it (§8.5).
 */
public interface InferenceEngine {

    /** Initialize the engine (load model, allocate resources). */
    void initialize() throws InferenceException;

    /** Run inference on a given request. */
    InferenceResult infer(InferenceRequest request) throws InferenceException;

    /** Release all resources. */
    void shutdown();

    /** Whether the engine is ready to accept requests. */
    boolean isReady();
}

