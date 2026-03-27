package com.aresstack.winacp.inference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Stub implementation of {@link InferenceEngine} for development and testing.
 * <p>
 * Returns deterministic canned responses. This is <b>not</b> a real
 * inference backend – it exists so the rest of the stack can run
 * end-to-end without a model or GPU.
 * <p>
 * The real implementation (DirectML/D3D12 via FFM + jextract) will
 * replace this once the Windows bindings module is ready.
 */
public class StubInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(StubInferenceEngine.class);

    private boolean ready = false;

    @Override
    public void initialize() throws InferenceException {
        log.info("StubInferenceEngine initializing (no real model)");
        ready = true;
        log.info("StubInferenceEngine ready");
    }

    @Override
    public InferenceResult infer(InferenceRequest request) throws InferenceException {
        if (!ready) throw new InferenceException("Engine not initialized");

        log.debug("StubInferenceEngine: prompt length={} chars", request.getPrompt().length());

        // Deterministic stub response
        String response = "[stub] I received your request but I am a stub inference engine. " +
                "A real DirectML-based engine will replace me.";

        return new InferenceResult(response, true, response.split("\\s+").length);
    }

    @Override
    public void shutdown() {
        ready = false;
        log.info("StubInferenceEngine shut down");
    }

    @Override
    public boolean isReady() {
        return ready;
    }
}

