package com.aresstack.winacp.inference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class StubInferenceEngineTest {

    private StubInferenceEngine engine;

    @BeforeEach
    void setUp() throws InferenceException {
        engine = new StubInferenceEngine();
        engine.initialize();
    }

    @AfterEach
    void tearDown() {
        engine.shutdown();
    }

    @Test
    void isReadyAfterInit() {
        assertTrue(engine.isReady());
    }

    @Test
    void generateReturnsResult() throws InferenceException {
        InferenceRequest req = InferenceRequest.builder()
                .userPrompt("What is 2+2?")
                .maxTokens(100)
                .temperature(0.7f)
                .build();
        InferenceResult result = engine.generate(req);

        assertNotNull(result);
        assertEquals("end_turn", result.getFinishReason());
        assertFalse(result.getText().isEmpty());
        assertNotNull(result.getUsage());
        assertTrue(result.getUsage().totalTokens() > 0);
    }

    @Test
    void generateBeforeInitThrows() {
        StubInferenceEngine cold = new StubInferenceEngine();
        assertThrows(InferenceException.class, () ->
                cold.generate(InferenceRequest.builder().userPrompt("test").build()));
    }

    @Test
    void shutdownClearsReady() {
        engine.shutdown();
        assertFalse(engine.isReady());
    }
}
