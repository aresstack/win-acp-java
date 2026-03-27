package com.aresstack.winacp.inference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

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
    void inferReturnsResult() throws InferenceException {
        InferenceRequest req = new InferenceRequest("What is 2+2?", List.of(), 100, 0.7f);
        InferenceResult result = engine.infer(req);

        assertNotNull(result);
        assertTrue(result.isComplete());
        assertFalse(result.getText().isEmpty());
        assertTrue(result.getTokenCount() > 0);
    }

    @Test
    void inferBeforeInitThrows() {
        StubInferenceEngine cold = new StubInferenceEngine();
        assertThrows(InferenceException.class, () ->
                cold.infer(new InferenceRequest("test", List.of(), 100, 0.7f)));
    }

    @Test
    void shutdownClearsReady() {
        engine.shutdown();
        assertFalse(engine.isReady());
    }
}

