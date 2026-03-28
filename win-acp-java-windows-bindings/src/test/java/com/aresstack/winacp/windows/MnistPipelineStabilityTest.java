package com.aresstack.winacp.windows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Stability and multi-run tests for the MNIST DirectML pipeline.
 * <p>
 * V1 scope: MNIST-family CNN vertical slice, currently validated with
 * {@code mnist-12.onnx} (opset 12).
 * <p>
 * These tests verify that the native COM lifecycle, GPU sync, and
 * descriptor/buffer resources behave correctly across multiple
 * inference runs and load/close cycles.
 */
class MnistPipelineStabilityTest {

    private static final Path MODEL_PATH = findModelPath("mnist-12.onnx");
    private static final Path INT8_MODEL_PATH = findModelPath("mnist-12-int8.onnx");

    private static Path findModelPath(String filename) {
        for (String prefix : new String[]{"model/", "../model/", "../../model/"}) {
            Path p = Path.of(prefix + filename);
            if (Files.exists(p)) return p;
        }
        return Path.of("model/" + filename);
    }

    static boolean modelExists() {
        return Files.exists(MODEL_PATH);
    }

    static boolean int8ModelExists() {
        return Files.exists(INT8_MODEL_PATH);
    }

    // ── Multi-run: N consecutive inferences on same pipeline ─────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void multipleInferences_tenRuns_allConsistent() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(MODEL_PATH);

                float[] input = new float[784]; // all-zeros
                float[] reference = pipeline.infer(input);
                assertNotNull(reference);
                assertEquals(10, reference.length);

                // Run 9 more times – all must match the first result exactly
                for (int i = 1; i < 10; i++) {
                    float[] result = pipeline.infer(input);
                    assertArrayEquals(reference, result, 0f,
                            "Run " + i + " diverged from reference");
                }
                System.out.println("10-run consistency: ✓");
            }
        }
    }

    // ── Load-infer-close cycle: repeated pipeline lifecycle ──────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void loadInferClose_threeCycles_noLeaks() throws Exception {
        float[] referenceOutput = null;

        for (int cycle = 0; cycle < 3; cycle++) {
            try (WindowsBindings wb = new WindowsBindings()) {
                wb.init("auto");
                if (!wb.hasDirectMl()) return;

                try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                    pipeline.loadModel(MODEL_PATH);

                    float[] input = new float[784];
                    float[] output = pipeline.infer(input);
                    assertNotNull(output);
                    assertEquals(10, output.length);

                    if (referenceOutput == null) {
                        referenceOutput = output;
                    } else {
                        assertArrayEquals(referenceOutput, output, 1e-5f,
                                "Cycle " + cycle + " produced different output");
                    }
                }
            }
            System.out.println("Cycle " + cycle + ": ✓");
        }
        System.out.println("3 load-infer-close cycles: ✓");
    }

    // ── Double-close safety ──────────────────────────────────────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void pipeline_doubleClose_noError() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            MnistPipeline pipeline = new MnistPipeline(wb);
            pipeline.loadModel(MODEL_PATH);
            pipeline.close();
            pipeline.close(); // must not throw
            System.out.println("Double-close: ✓");
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void windowsBindings_doubleClose_noError() throws Exception {
        WindowsBindings wb = new WindowsBindings();
        wb.init("auto");
        wb.close();
        wb.close(); // must not throw
        System.out.println("WindowsBindings double-close: ✓");
    }

    // ── Use-after-close guard ────────────────────────────────────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void pipeline_inferAfterClose_throws() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            MnistPipeline pipeline = new MnistPipeline(wb);
            pipeline.loadModel(MODEL_PATH);
            pipeline.close();

            assertThrows(WindowsNativeException.class, () -> {
                pipeline.infer(new float[784]);
            }, "infer() after close() should throw");
            System.out.println("Use-after-close guard: ✓");
        }
    }

    // ── Double-load guard ─────────────────────────────────────────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void pipeline_doubleLoad_throws() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(MODEL_PATH);

                assertThrows(IllegalStateException.class, () -> {
                    pipeline.loadModel(MODEL_PATH);
                }, "loadModel() on already-loaded pipeline should throw");
                System.out.println("Double-load guard: ✓");
            }
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void windowsBindings_initAfterClose_throws() throws Exception {
        WindowsBindings wb = new WindowsBindings();
        wb.init("auto");
        wb.close();

        assertThrows(IllegalStateException.class, () -> {
            wb.init("auto");
        }, "init() after close() should throw");
        System.out.println("WindowsBindings init-after-close guard: ✓");
    }

    // ── Varied input: different pixel patterns ───────────────────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void variedInputs_differentOutputs() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(MODEL_PATH);

                // All zeros
                float[] zeros = new float[784];
                float[] outZeros = pipeline.infer(zeros);

                // All ones
                float[] ones = new float[784];
                java.util.Arrays.fill(ones, 1.0f);
                float[] outOnes = pipeline.infer(ones);

                // Center pixel pattern (rough "1" shape)
                float[] centerLine = new float[784];
                for (int row = 5; row < 23; row++) {
                    centerLine[row * 28 + 14] = 1.0f;
                }
                float[] outCenter = pipeline.infer(centerLine);

                // Outputs should differ
                assertNotNull(outZeros);
                assertNotNull(outOnes);
                assertNotNull(outCenter);

                boolean allSame = java.util.Arrays.equals(outZeros, outOnes)
                        && java.util.Arrays.equals(outOnes, outCenter);
                assertFalse(allSame, "Different inputs should produce different logit distributions");

                System.out.printf("Zeros → digit %d, Ones → digit %d, Center → digit %d%n",
                        MnistPipeline.argmax(outZeros),
                        MnistPipeline.argmax(outOnes),
                        MnistPipeline.argmax(outCenter));
            }
        }
    }

    // ── Regression: mnist-12 end-to-end ──────────────────────────────────

    /**
     * Full regression test for mnist-12.onnx: parse → load → infer → argmax → consistency.
     * This is the primary validation that the V1 vertical slice works with opset 12.
     */
    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void mnist12_regression_fullEndToEnd() throws Exception {
        // 1. Parse: verify graph structure
        var graph = OnnxModelReader.parse(MODEL_PATH);
        assertNotNull(graph, "Graph must parse");
        assertFalse(graph.nodes().isEmpty(), "Graph must have nodes");
        assertFalse(graph.initializers().isEmpty(), "Graph must have initializers");

        // Verify expected MNIST operators are present
        long convCount = graph.nodes().stream().filter(n -> "Conv".equals(n.opType())).count();
        long poolCount = graph.nodes().stream().filter(n -> "MaxPool".equals(n.opType())).count();
        long matMulCount = graph.nodes().stream().filter(n -> "MatMul".equals(n.opType())).count();
        assertEquals(2, convCount, "MNIST should have 2 Conv nodes");
        assertEquals(2, poolCount, "MNIST should have 2 MaxPool nodes");
        assertEquals(1, matMulCount, "MNIST should have 1 MatMul node");

        System.out.printf("Graph: %d nodes, %d inits, Conv=%d, MaxPool=%d, MatMul=%d%n",
                graph.nodes().size(), graph.initializers().size(), convCount, poolCount, matMulCount);

        // 2. Full inference pipeline
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) {
                System.out.println("SKIP: DirectML not available");
                return;
            }

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(MODEL_PATH);

                // 3. Inference with zeros input
                float[] zeros = new float[784];
                float[] outZeros = pipeline.infer(zeros);
                assertNotNull(outZeros);
                assertEquals(10, outZeros.length, "MNIST output must be 10 logits");

                int digitZeros = MnistPipeline.argmax(outZeros);
                assertTrue(digitZeros >= 0 && digitZeros <= 9, "argmax must be a digit 0-9");

                // 4. Multi-run consistency (5 runs)
                for (int i = 0; i < 5; i++) {
                    float[] repeat = pipeline.infer(zeros);
                    assertArrayEquals(outZeros, repeat, 0f,
                            "Run " + i + " must match reference (deterministic)");
                }

                // 5. Different input produces different output
                float[] ones = new float[784];
                java.util.Arrays.fill(ones, 1.0f);
                float[] outOnes = pipeline.infer(ones);
                assertFalse(java.util.Arrays.equals(outZeros, outOnes),
                        "Different inputs must produce different logits");

                System.out.printf("mnist-12 regression: zeros→%d, ones→%d, 5-run consistent ✓%n",
                        digitZeros, MnistPipeline.argmax(outOnes));
            }
        }
    }

    // ── Regression: mnist-12-int8 end-to-end ─────────────────────────────

    /**
     * Full regression test for mnist-12-int8.onnx: parse → dequantize → load → infer → consistency.
     * Validates that the int8 quantized model produces reasonable outputs through the
     * dequantize-first pipeline.
     */
    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("int8ModelExists")
    void mnist12int8_regression_fullEndToEnd() throws Exception {
        // 1. Parse: verify int8 graph structure
        var graph = OnnxModelReader.parse(INT8_MODEL_PATH);
        assertNotNull(graph, "Graph must parse");

        long qConvCount = graph.nodes().stream()
                .filter(n -> "QLinearConv".equals(n.opType())).count();
        assertEquals(2, qConvCount, "Int8 MNIST should have 2 QLinearConv nodes");

        assertTrue(graph.nodes().stream().anyMatch(n -> "QLinearMatMul".equals(n.opType())),
                "Int8 MNIST should have QLinearMatMul");

        // 2. Full inference pipeline (dequantize-first)
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) {
                System.out.println("SKIP: DirectML not available");
                return;
            }

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(INT8_MODEL_PATH);

                // 3. Inference with zeros input
                float[] zeros = new float[784];
                float[] outZeros = pipeline.infer(zeros);
                assertNotNull(outZeros);
                assertEquals(10, outZeros.length, "MNIST output must be 10 logits");

                int digitZeros = MnistPipeline.argmax(outZeros);
                assertTrue(digitZeros >= 0 && digitZeros <= 9);

                // 4. Multi-run consistency (3 runs)
                for (int i = 0; i < 3; i++) {
                    float[] repeat = pipeline.infer(zeros);
                    assertArrayEquals(outZeros, repeat, 0f,
                            "Int8 run " + i + " must match reference (deterministic)");
                }

                // 5. Different input, different output
                float[] ones = new float[784];
                java.util.Arrays.fill(ones, 1.0f);
                float[] outOnes = pipeline.infer(ones);
                assertFalse(java.util.Arrays.equals(outZeros, outOnes),
                        "Different inputs must produce different logits (int8)");

                System.out.printf("mnist-12-int8 regression: zeros→%d, ones→%d, 3-run consistent ✓%n",
                        digitZeros, MnistPipeline.argmax(outOnes));
            }
        }
    }

    // ── Cross-model: float32 vs int8 produce same argmax ─────────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void float32VsInt8_sameArgmax_forZerosInput() throws Exception {
        if (!Files.exists(INT8_MODEL_PATH)) {
            System.out.println("SKIP: int8 model not available");
            return;
        }

        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            float[] input = new float[784]; // zeros

            int f32Digit;
            try (MnistPipeline f32 = new MnistPipeline(wb)) {
                f32.loadModel(MODEL_PATH);
                f32Digit = MnistPipeline.argmax(f32.infer(input));
            }

            int i8Digit;
            try (MnistPipeline i8 = new MnistPipeline(wb)) {
                i8.loadModel(INT8_MODEL_PATH);
                i8Digit = MnistPipeline.argmax(i8.infer(input));
            }

            assertEquals(f32Digit, i8Digit,
                    "Float32 and int8 should agree on argmax for zeros input");
            System.out.printf("Cross-model: float32→%d, int8→%d ✓%n", f32Digit, i8Digit);
        }
    }
}

