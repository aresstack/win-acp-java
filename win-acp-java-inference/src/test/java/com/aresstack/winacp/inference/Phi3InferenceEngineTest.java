package com.aresstack.winacp.inference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration test for {@link Phi3InferenceEngine}.
 * Validates the full InferenceEngine contract: initialize → generate → shutdown.
 */
class Phi3InferenceEngineTest {

    private static final Path MODEL_DIR = Path.of(
            System.getProperty("user.dir")).getParent()
            .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");

    static boolean modelAvailable() {
        return Phi3InferenceEngine.isValidModelDir(MODEL_DIR);
    }

    @Test
    @EnabledIf("modelAvailable")
    void full_lifecycle() throws InferenceException {
        Phi3InferenceEngine engine = new Phi3InferenceEngine(MODEL_DIR, 5);

        try {
            // Initialize
            assertFalse(engine.isReady());
            engine.initialize();
            assertTrue(engine.isReady());

            // Generate
            InferenceRequest request = InferenceRequest.builder()
                    .systemPrompt("You are a helpful assistant.")
                    .userPrompt("What is 2+2?")
                    .maxTokens(5)
                    .build();

            InferenceResult result = engine.generate(request);

            assertNotNull(result);
            assertNotNull(result.getText());
            assertFalse(result.getText().isEmpty(), "Should generate at least some text");
            assertNotNull(result.getFinishReason());
            assertNotNull(result.getUsage());
            assertTrue(result.getUsage().promptTokens() > 0);
            assertTrue(result.getUsage().completionTokens() > 0);

            System.out.println("Generated: '" + result.getText() + "'");
            System.out.println("Usage: " + result.getUsage());
            System.out.println("Finish: " + result.getFinishReason());

        } finally {
            engine.shutdown();
            assertFalse(engine.isReady());
        }
    }

    @Test
    @EnabledIf("modelAvailable")
    void generate_without_system_prompt() throws InferenceException {
        Phi3InferenceEngine engine = new Phi3InferenceEngine(MODEL_DIR, 3);

        try {
            engine.initialize();

            InferenceRequest request = InferenceRequest.builder()
                    .userPrompt("Say hello.")
                    .maxTokens(3)
                    .build();

            InferenceResult result = engine.generate(request);
            assertNotNull(result);
            assertNotNull(result.getText());
            System.out.println("No-system-prompt result: '" + result.getText() + "'");

        } finally {
            engine.shutdown();
        }
    }

    @Test
    void isValidModelDir_nonexistent() {
        assertFalse(Phi3InferenceEngine.isValidModelDir(Path.of("/nonexistent/path")));
    }

    @Test
    void isValidModelDir_null() {
        assertFalse(Phi3InferenceEngine.isValidModelDir(null));
    }

    @Test
    void generate_before_init_throws() {
        Phi3InferenceEngine engine = new Phi3InferenceEngine(MODEL_DIR, 5);
        InferenceRequest request = InferenceRequest.builder()
                .userPrompt("test")
                .build();
        assertThrows(InferenceException.class, () -> engine.generate(request));
    }
}

