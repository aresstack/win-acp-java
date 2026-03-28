package com.aresstack.winacp.windows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the minimal ONNX protobuf parser and the MNIST DirectML pipeline.
 * <p>
 * V1 scope: MNIST-family CNN vertical slice, validated with
 * {@code mnist-12.onnx} (float32) and {@code mnist-12-int8.onnx} (quantized).
 * Tests that require a model are skipped if the file is not present.
 */
class MnistPipelineTest {

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

    // ── ONNX Parser Tests ────────────────────────────────────────────────

    @Test
    @EnabledIf("modelExists")
    void onnxParser_canParseMnistModel() throws Exception {
        var graph = OnnxModelReader.parse(MODEL_PATH);
        assertNotNull(graph);
        assertFalse(graph.nodes().isEmpty(), "Graph should have nodes");
        assertFalse(graph.initializers().isEmpty(), "Graph should have initializers (weights)");

        System.out.println("Graph name: " + graph.name());
        System.out.println("Nodes: " + graph.nodes().size());
        System.out.println("Initializers: " + graph.initializers().size());
        System.out.println("Inputs: " + graph.inputNames());
        System.out.println("Outputs: " + graph.outputNames());

        // Print all node op types
        for (var node : graph.nodes()) {
            System.out.printf("  Node: %s, inputs=%s, outputs=%s%n",
                    node.opType(), node.inputs(), node.outputs());
        }
    }

    @Test
    @EnabledIf("modelExists")
    void onnxParser_extractsConvWeights() throws Exception {
        var graph = OnnxModelReader.parse(MODEL_PATH);

        // Find first Conv node and check its weights
        var convNode = graph.nodes().stream()
                .filter(n -> "Conv".equals(n.opType()))
                .findFirst().orElse(null);
        assertNotNull(convNode, "No Conv node found");

        String filterName = convNode.inputs().get(1);
        var filterTensor = graph.initializers().get(filterName);
        assertNotNull(filterTensor, "Conv filter tensor should exist: " + filterName);
        assertTrue(filterTensor.data().length > 0, "Filter should have data");
        assertEquals(1, filterTensor.dataType(), "Data type should be FLOAT (1)");

        System.out.printf("Conv1 filter: name=%s, dims=%s, elements=%d%n",
                filterTensor.name(), java.util.Arrays.toString(filterTensor.dims()),
                filterTensor.data().length);
    }

    @Test
    @EnabledIf("modelExists")
    void onnxParser_extractsFCWeights() throws Exception {
        var graph = OnnxModelReader.parse(MODEL_PATH);

        // Find MatMul node
        var matMulNode = graph.nodes().stream()
                .filter(n -> "MatMul".equals(n.opType()))
                .findFirst().orElse(null);
        assertNotNull(matMulNode, "No MatMul node found");

        // The weight may be a Reshape output – trace through Reshape nodes
        String weightName = matMulNode.inputs().get(1);
        var weightTensor = graph.initializers().get(weightName);

        if (weightTensor == null) {
            // Trace through Reshape nodes
            var reshapeNode = graph.nodes().stream()
                    .filter(n -> "Reshape".equals(n.opType()) && n.outputs().contains(weightName))
                    .findFirst().orElse(null);
            if (reshapeNode != null) {
                weightTensor = graph.initializers().get(reshapeNode.inputs().get(0));
                System.out.println("FC weight traced through Reshape: " + reshapeNode.inputs().get(0));
            }
        }

        assertNotNull(weightTensor, "MatMul weight tensor should exist (direct or via Reshape)");
        assertTrue(weightTensor.data().length > 0, "Weight should have data");

        System.out.printf("FC weight: name=%s, dims=%s, elements=%d%n",
                weightTensor.name(), java.util.Arrays.toString(weightTensor.dims()),
                weightTensor.data().length);
    }

    // ── Full MNIST Pipeline Test ─────────────────────────────────────────

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void mnistPipeline_fullInference_producesOutput() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            assertTrue(wb.isInitialised(), "Windows bindings must initialize");

            if (!wb.hasDirectMl()) {
                System.out.println("SKIP: DirectML not available on this system");
                return;
            }

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(MODEL_PATH);

                // Run inference with all-zeros input (should still produce valid logits)
                float[] input = new float[784];
                float[] output = pipeline.infer(input);

                assertNotNull(output);
                assertEquals(10, output.length, "MNIST output should have 10 logits");

                int predicted = MnistPipeline.argmax(output);
                assertTrue(predicted >= 0 && predicted <= 9);

                System.out.println("MNIST output logits: " + java.util.Arrays.toString(output));
                System.out.println("Predicted digit (zeros input): " + predicted);
                System.out.println("argmax is readable: ✓");
            }
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("modelExists")
    void mnistPipeline_multipleInferences_consistent() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(MODEL_PATH);

                // Two inferences with the same input should give the same result
                float[] input = new float[784];
                float[] out1 = pipeline.infer(input);
                float[] out2 = pipeline.infer(input);

                assertArrayEquals(out1, out2, 1e-6f, "Same input should produce same output");

                System.out.println("Consistency check: ✓");
            }
        }
    }

    // ── Argmax Test ──────────────────────────────────────────────────────

    @Test
    void argmax_findsMaxIndex() {
        float[] logits = {0.1f, 0.05f, 0.02f, 0.0f, 0.01f, 0.0f, 0.0f, 0.8f, 0.01f, 0.01f};
        assertEquals(7, MnistPipeline.argmax(logits));

        float[] equal = {1.0f, 1.0f, 1.0f};
        assertEquals(0, MnistPipeline.argmax(equal)); // first max
    }

    // ── Int8 Model Tests ─────────────────────────────────────────────────

    @Test
    @EnabledIf("int8ModelExists")
    void onnxParser_canParseInt8Model() throws Exception {
        var graph = OnnxModelReader.parse(INT8_MODEL_PATH);
        assertNotNull(graph);
        assertFalse(graph.nodes().isEmpty(), "Graph should have nodes");
        assertFalse(graph.initializers().isEmpty(), "Graph should have initializers");

        // Int8 graph should have quantized operators
        assertTrue(graph.nodes().stream().anyMatch(n -> "QLinearConv".equals(n.opType())),
                "Int8 model should have QLinearConv nodes");

        // Should have quantized initializers (INT8 tensors with rawBytes)
        boolean hasInt8Tensor = graph.initializers().values().stream()
                .anyMatch(t -> t.dataType() == OnnxModelReader.ONNX_INT8 && t.rawBytes().length > 0);
        assertTrue(hasInt8Tensor, "Int8 model should have INT8 tensors with raw data");

        System.out.println("Int8 graph: " + graph.nodes().size() + " nodes, "
                + graph.initializers().size() + " initializers");
        for (var node : graph.nodes()) {
            System.out.printf("  %s: in=%d, out=%d%n",
                    node.opType(), node.inputs().size(), node.outputs().size());
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("int8ModelExists")
    void mnistInt8Pipeline_fullInference_producesOutput() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) {
                System.out.println("SKIP: DirectML not available");
                return;
            }

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(INT8_MODEL_PATH);

                float[] input = new float[784];
                float[] output = pipeline.infer(input);

                assertNotNull(output);
                assertEquals(10, output.length, "MNIST output should have 10 logits");

                int predicted = MnistPipeline.argmax(output);
                assertTrue(predicted >= 0 && predicted <= 9);

                System.out.println("Int8 MNIST output: " + java.util.Arrays.toString(output));
                System.out.println("Int8 predicted digit (zeros): " + predicted);
            }
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    @EnabledIf("int8ModelExists")
    void mnistInt8Pipeline_multipleInferences_consistent() throws Exception {
        try (WindowsBindings wb = new WindowsBindings()) {
            wb.init("auto");
            if (!wb.hasDirectMl()) return;

            try (MnistPipeline pipeline = new MnistPipeline(wb)) {
                pipeline.loadModel(INT8_MODEL_PATH);

                float[] input = new float[784];
                float[] out1 = pipeline.infer(input);
                float[] out2 = pipeline.infer(input);

                assertArrayEquals(out1, out2, 1e-6f, "Same input should produce same output (int8)");
                System.out.println("Int8 consistency check: ✓");
            }
        }
    }
}

