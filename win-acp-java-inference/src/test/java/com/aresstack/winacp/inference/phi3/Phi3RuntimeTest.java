package com.aresstack.winacp.inference.phi3;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for {@link Phi3Runtime} - end-to-end text generation.
 * Requires the full Phi-3 model + tokenizer files.
 */
class Phi3RuntimeTest {

    /** Resolve model path relative to project root (parent of this submodule). */
    private static final Path MODEL_DIR = Path.of(
            System.getProperty("user.dir")).getParent()
            .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");

    static boolean modelAvailable() {
        return Files.exists(MODEL_DIR.resolve("model.onnx"))
                && Files.exists(MODEL_DIR.resolve("model.onnx.data"))
                && Files.exists(MODEL_DIR.resolve("config.json"))
                && Files.exists(MODEL_DIR.resolve("tokenizer.json"));
    }

    @Test
    void rmsNorm_basic() {
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] w = {1.0f, 1.0f, 1.0f, 1.0f};
        Phi3Runtime.rmsNorm(x, w, 1e-5f);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ~ 2.7386
        // Each element should be divided by RMS
        float rms = (float) Math.sqrt(30.0f / 4.0f + 1e-5f);
        assertEquals(1.0f / rms, x[0], 1e-4f);
        assertEquals(2.0f / rms, x[1], 1e-4f);
    }

    @Test
    void softmax_basic() {
        float[] x = {1.0f, 2.0f, 3.0f};
        Phi3Runtime.softmax(x);

        // Sum should be 1.0
        float sum = 0;
        for (float v : x) sum += v;
        assertEquals(1.0f, sum, 1e-5f);

        // Should be monotonically increasing
        assertTrue(x[0] < x[1]);
        assertTrue(x[1] < x[2]);
    }

    @Test
    void argmax_basic() {
        assertEquals(2, Phi3Runtime.argmax(new float[]{1.0f, 2.0f, 5.0f, 3.0f}));
        assertEquals(0, Phi3Runtime.argmax(new float[]{10.0f, 2.0f, 5.0f}));
    }

    @Test
    @EnabledIf("modelAvailable")
    void generate_single_token() throws IOException {
        Phi3Config config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
        Phi3Tokenizer tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));

        try (Phi3Weights weights = Phi3Weights.load(MODEL_DIR, config)) {
            Phi3Runtime runtime = new Phi3Runtime(config, weights, tokenizer);

            String prompt = tokenizer.formatChat(
                    "You are a helpful assistant.",
                    "What is 2+2?"
            );

            // Generate just 1 token to verify the pipeline works
            String output = runtime.generate(prompt, 1);
            assertNotNull(output);
            assertFalse(output.isEmpty(), "Should generate at least 1 token");
            System.out.println("Generated: '" + output + "'");
        }
    }

    @Test
    @EnabledIf("modelAvailable")
    void generate_short_response() throws IOException {
        Phi3Config config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
        Phi3Tokenizer tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));

        try (Phi3Weights weights = Phi3Weights.load(MODEL_DIR, config)) {
            Phi3Runtime runtime = new Phi3Runtime(config, weights, tokenizer);

            String prompt = tokenizer.formatChat(null, "Say hello.");

            String output = runtime.generate(prompt, 20);
            assertNotNull(output);
            System.out.println("Generated: '" + output + "'");
        }
    }
}
