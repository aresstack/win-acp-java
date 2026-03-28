package com.aresstack.winacp.windows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Inspect the EMNIST blank model architecture for pipeline integration.
 */
class EmnistModelInspectionTest {

    private static final Path MODEL_PATH = findModelPath("mnist_emnist_blank_cnn_v1.onnx");

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

    @Test
    @EnabledIf("modelExists")
    void inspectEmnistBlankModel() throws Exception {
        var graph = OnnxModelReader.parse(MODEL_PATH);
        assertNotNull(graph);

        System.out.println("=== EMNIST Blank Model Architecture ===");
        System.out.println("Graph name: " + graph.name());
        System.out.println("Nodes: " + graph.nodes().size());
        System.out.println("Initializers: " + graph.initializers().size());
        System.out.println("Inputs: " + graph.inputNames());
        System.out.println("Outputs: " + graph.outputNames());
        System.out.println();

        System.out.println("=== Nodes ===");
        for (int i = 0; i < graph.nodes().size(); i++) {
            var n = graph.nodes().get(i);
            System.out.printf("[%2d] %-20s in=%s out=%s%n", i, n.opType(), n.inputs(), n.outputs());
            for (var a : n.attrs().entrySet()) {
                System.out.printf("       attr: %s = %s%n", a.getKey(), a.getValue());
            }
        }

        System.out.println();
        System.out.println("=== Initializers (Weights) ===");
        for (var e : graph.initializers().entrySet()) {
            var t = e.getValue();
            System.out.printf("  %-40s dims=%-25s dtype=%d floats=%d raw=%d%n",
                    t.name(), Arrays.toString(t.dims()), t.dataType(), t.data().length, t.rawBytes().length);
        }

        // Basic assertions
        assertFalse(graph.nodes().isEmpty());
        assertFalse(graph.initializers().isEmpty());

        // Count op types
        long convCount = graph.nodes().stream().filter(n -> "Conv".equals(n.opType())).count();
        long bnCount = graph.nodes().stream().filter(n -> "BatchNormalization".equals(n.opType())).count();
        long reluCount = graph.nodes().stream().filter(n -> "Relu".equals(n.opType())).count();
        long poolCount = graph.nodes().stream().filter(n -> "MaxPool".equals(n.opType())).count();
        long dropoutCount = graph.nodes().stream().filter(n -> "Dropout".equals(n.opType())).count();
        long flattenCount = graph.nodes().stream().filter(n -> "Flatten".equals(n.opType())).count();
        long gemmCount = graph.nodes().stream().filter(n -> "Gemm".equals(n.opType())).count();
        long matMulCount = graph.nodes().stream().filter(n -> "MatMul".equals(n.opType())).count();
        long linearCount = graph.nodes().stream().filter(n -> "Linear".equals(n.opType())).count();

        System.out.println();
        System.out.println("=== Op Summary ===");
        System.out.println("Conv: " + convCount);
        System.out.println("BatchNormalization: " + bnCount);
        System.out.println("Relu: " + reluCount);
        System.out.println("MaxPool: " + poolCount);
        System.out.println("Dropout: " + dropoutCount);
        System.out.println("Flatten: " + flattenCount);
        System.out.println("Gemm: " + gemmCount);
        System.out.println("MatMul: " + matMulCount);
        System.out.println("Linear: " + linearCount);
    }
}

