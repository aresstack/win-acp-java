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
 * These tests verify that the native COM lifecycle, GPU sync, and
 * descriptor/buffer resources behave correctly across multiple
 * inference runs and load/close cycles.
 */
class MnistPipelineStabilityTest {

    private static final Path MODEL_PATH = findModelPath();

    private static Path findModelPath() {
        for (String candidate : new String[]{
                "model/mnist-8.onnx",
                "../model/mnist-8.onnx",
                "../../model/mnist-8.onnx"
        }) {
            Path p = Path.of(candidate);
            if (Files.exists(p)) return p;
        }
        return Path.of("model/mnist-8.onnx");
    }

    static boolean modelExists() {
        return Files.exists(MODEL_PATH);
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
}

