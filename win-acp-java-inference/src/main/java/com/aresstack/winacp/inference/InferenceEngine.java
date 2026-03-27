package com.aresstack.winacp.inference;

/**
 * Abstraction over the local inference backend.
 * <p>
 * The rest of the system consumes inference through this interface only.
 * The concrete implementation is hidden behind it – no ACP, graph, or
 * MCP code may know about Windows/DirectML details.
 * <p>
 * <b>V1 scope:</b> The only real implementation is
 * {@link MnistDirectMlEngine} which classifies 28×28 grayscale digits
 * via DirectML on the GPU. Text generation or chat models are <em>not</em>
 * supported in V1. A {@link StubInferenceEngine} exists for development
 * and testing without a GPU.
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
