package com.aresstack.winacp.inference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Stub implementation of {@link InferenceEngine} for development and testing.
 * <p>
 * Returns deterministic canned responses. This is <b>not</b> a real
 * inference backend – it exists so the rest of the stack can run
 * end-to-end without a model or GPU.
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
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) throw new InferenceException("Engine not initialized");

        log.debug("StubInferenceEngine: prompt length={} chars", request.getUserPrompt().length());

        // Deterministic stub response
        String response = "[stub] I received your request but I am a stub inference engine. " +
                "A real DirectML-based engine will replace me.";

        int wordCount = response.split("\\s+").length;
        return new InferenceResult(response, "end_turn",
                new InferenceResult.Usage(10, wordCount, 10 + wordCount));
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
