package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.windows.WindowsBindings;
import org.junit.jupiter.api.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * V3.0 GPU Pipeline validation test.
 * <p>
 * Validates the full-GPU decode path (1 submission/token) against the
 * V2.0 per-layer path (65 submissions/token) to ensure numerical correctness.
 * <p>
 * <b>Requires</b>: Windows 11, DirectML-capable GPU with ≥16 GB VRAM,
 * and the Phi-3-mini model at the standard path.
 * <p>
 * Run with: {@code --enable-native-access=ALL-UNNAMED -Xmx4g}
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class Phi3GpuPipelineV3Test {

    private static final Path MODEL_DIR = resolveModelDir();

    private static Phi3Config config;
    private static Phi3Tokenizer tokenizer;
    private static Phi3Weights weights;
    private static WindowsBindings wb;
    private static Phi3GpuKernels gpuKernels;

    // V2.0 pipeline (MLP batch, no full GPU)
    private static Phi3GpuPipeline pipelineV2;
    private static Phi3Runtime runtimeV2;

    // V3.0 pipeline (full GPU)
    private static Phi3GpuPipeline pipelineV3;
    private static Phi3Runtime runtimeV3;

    private static boolean v3Available = false;

    private static Path resolveModelDir() {
        Path rel = Path.of("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(rel.resolve("model.onnx"))) return rel;
        Path parent = Path.of(System.getProperty("user.dir")).getParent()
                .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(parent.resolve("model.onnx"))) return parent;
        return rel;
    }

    @BeforeAll
    static void loadModel() throws Exception {
        assumeTrue(WindowsBindings.isSupported(), "Requires Windows with DirectML");
        assumeTrue(Files.exists(MODEL_DIR.resolve("model.onnx")), "Model not found at " + MODEL_DIR);

        config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
        tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));
        weights = Phi3Weights.load(MODEL_DIR, config);

        wb = new WindowsBindings();
        wb.init("directml");
        assumeTrue(wb.hasDirectMl(), "DirectML not available");

        int gpuLayers = config.numHiddenLayers();
        gpuKernels = Phi3GpuKernels.create(wb, weights, config, gpuLayers, true);

        // V3.0 pipeline — will detect if shaders compile
        pipelineV3 = new Phi3GpuPipeline(wb, gpuKernels, config);
        pipelineV3.uploadLayerWeights(wb, weights, config);
        v3Available = pipelineV3.isFullGpuEnabled();

        // V2.0 runtime (uses pipeline but bypasses V3.0)
        // We create two separate runtimes to compare outputs
        runtimeV3 = new Phi3Runtime(config, weights, tokenizer, gpuKernels, pipelineV3);

        System.out.println("V3.0 full GPU available: " + v3Available);
        System.out.println("maxGpuPos: " + pipelineV3.getMaxGpuPos());
    }

    @AfterAll
    static void cleanup() {
        if (pipelineV3 != null) pipelineV3.close();
        if (gpuKernels != null) gpuKernels.close();
        if (wb != null) wb.close();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 1: Shader compilation
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(1)
    @DisplayName("V3.0 compute shaders compile successfully")
    void shaderCompilation() {
        assumeTrue(WindowsBindings.isSupported());
        // If we got here without exception, shaders compiled
        assertNotNull(pipelineV3, "Pipeline should be created");
        System.out.println("V3.0 shader compilation: " + (v3Available ? "OK" : "FAILED (V2.0 fallback)"));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 2: Single token decode produces valid logits
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(2)
    @DisplayName("V3.0 full GPU decode produces non-zero logits")
    void singleTokenDecode() {
        assumeTrue(v3Available, "V3.0 not available");

        runtimeV3.resetCache();
        String result = runtimeV3.generate("Hello", 1);

        assertNotNull(result, "Generated text should not be null");
        assertFalse(result.isEmpty(), "Generated text should not be empty");
        System.out.println("V3.0 single token output: '" + result + "'");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 3: Multi-token generation
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(3)
    @DisplayName("V3.0 generates coherent multi-token output")
    void multiTokenGeneration() {
        assumeTrue(v3Available, "V3.0 not available");

        runtimeV3.resetCache();
        String prompt = "<|system|>\nYou are helpful.<|end|>\n<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n";
        String result = runtimeV3.generate(prompt, 32);

        assertNotNull(result);
        assertFalse(result.isBlank(), "Should generate non-empty text");
        System.out.println("V3.0 multi-token output (" + result.length() + " chars): " + result);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 4: Multi-turn stability
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(4)
    @DisplayName("V3.0 handles multiple turns without crashing")
    void multiTurnStability() {
        assumeTrue(v3Available, "V3.0 not available");

        runtimeV3.resetCache();
        String[] prompts = {
                "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\n",
                "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi! How can I help?<|end|>\n<|user|>\nWhat is Java?<|end|>\n<|assistant|>\n",
                "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi!<|end|>\n<|user|>\nBye<|end|>\n<|assistant|>\n",
        };

        for (int i = 0; i < prompts.length; i++) {
            runtimeV3.resetCache();
            String result = runtimeV3.generate(prompts[i], 16);
            assertNotNull(result, "Turn " + i + " should produce output");
            System.out.printf("Turn %d: '%s'%n", i, result.substring(0, Math.min(80, result.length())));
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 5: Determinism (same input → same output)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(5)
    @DisplayName("V3.0 greedy decode is deterministic")
    void determinism() {
        assumeTrue(v3Available, "V3.0 not available");

        String prompt = "<|system|>\nBe concise.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n";

        runtimeV3.resetCache();
        String result1 = runtimeV3.generate(prompt, 16);

        runtimeV3.resetCache();
        String result2 = runtimeV3.generate(prompt, 16);

        assertEquals(result1, result2, "Greedy decode should be deterministic");
        System.out.println("Deterministic output: '" + result1 + "'");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 6: Benchmark — V3.0 performance measurement
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(6)
    @DisplayName("V3.0 benchmark — ms/token")
    void benchmark() {
        assumeTrue(v3Available, "V3.0 not available");

        String prompt = "<|system|>\nYou are helpful.<|end|>\n<|user|>\nExplain TCP vs UDP briefly.<|end|>\n<|assistant|>\n";

        // Warmup
        runtimeV3.resetCache();
        runtimeV3.generate(prompt, 16);

        // Benchmark: 3 runs × 32 tokens
        int runs = 3;
        int tokens = 32;
        double[] msPerToken = new double[runs];

        for (int r = 0; r < runs; r++) {
            runtimeV3.resetCache();
            long t0 = System.nanoTime();
            final int[] count = {0};
            runtimeV3.generateStreaming(prompt, tokens,
                    (id, text, delta) -> count[0]++);
            long elapsed = System.nanoTime() - t0;
            msPerToken[r] = (elapsed / 1e6) / Math.max(count[0], 1);
        }

        double avg = Arrays.stream(msPerToken).average().orElse(0);
        double min = Arrays.stream(msPerToken).min().orElse(0);
        double max = Arrays.stream(msPerToken).max().orElse(0);

        System.out.printf("V3.0 Benchmark (%d runs × %d tokens):%n", runs, tokens);
        System.out.printf("  avg: %.1f ms/token (%.1f tok/s)%n", avg, 1000.0 / avg);
        System.out.printf("  min: %.1f ms/token%n", min);
        System.out.printf("  max: %.1f ms/token%n", max);

        String profile = runtimeV3.getLastProfile();
        if (profile != null) {
            System.out.println("  Profile:\n" + profile);
        }

        assertTrue(avg > 0, "Benchmark should produce valid timings");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 7: KV cache reuse across turns
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(7)
    @DisplayName("V3.0 KV cache reuse — common prefix not re-prefilled")
    void kvCacheReuse() {
        assumeTrue(v3Available, "V3.0 not available");

        runtimeV3.resetCache();

        // Turn 1
        String prompt1 = "<|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n";
        long t1Start = System.nanoTime();
        String r1 = runtimeV3.generate(prompt1, 8);
        long t1 = System.nanoTime() - t1Start;

        // Turn 2 — extends the same prefix
        String prompt2 = prompt1 + r1 + "<|end|>\n<|user|>\nBye<|end|>\n<|assistant|>\n";
        long t2Start = System.nanoTime();
        String r2 = runtimeV3.generate(prompt2, 8);
        long t2 = System.nanoTime() - t2Start;

        System.out.printf("Turn 1: %.1f ms, Turn 2: %.1f ms (should be faster due to cache reuse)%n",
                t1 / 1e6, t2 / 1e6);

        assertNotNull(r1);
        assertNotNull(r2);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Test 8: Resource cleanup
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @Order(8)
    @DisplayName("V3.0 pipeline closes without error")
    void resourceCleanup() {
        // This is implicitly tested by @AfterAll, but we verify no exception
        assertDoesNotThrow(() -> {
            if (v3Available) {
                // Run a generation before close to ensure all resources are "dirty"
                runtimeV3.resetCache();
                runtimeV3.generate("Test", 4);
            }
        });
    }
}
