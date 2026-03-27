package com.aresstack.winacp.inference;

/**
 * Abstraction over the local inference backend.
 * <p>
 * The rest of the system consumes inference through this interface only.
 * The concrete implementation (DirectML, CPU, Stub) is hidden behind it.
 * No ACP, graph, or MCP code may know about Windows/DirectML details.
 */
public interface InferenceEngine {

    /** Initialize the engine (load model, allocate resources). */
    void initialize() throws InferenceException;

    /** Run inference and produce a result. */
    InferenceResult generate(InferenceRequest request) throws InferenceException;

    /** Release all resources. */
    void shutdown();

    /** Whether the engine is ready to accept requests. */
    boolean isReady();
}
