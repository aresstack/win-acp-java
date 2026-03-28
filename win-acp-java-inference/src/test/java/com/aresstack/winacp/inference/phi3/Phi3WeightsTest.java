package com.aresstack.winacp.inference.phi3;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Phi3Weights} - ONNX weight loading.
 * Requires the full Phi-3 model directory with model.onnx and model.onnx.data.
 */
class Phi3WeightsTest {

    /** Resolve model path relative to project root (parent of this submodule). */
    private static final Path MODEL_DIR = Path.of(
            System.getProperty("user.dir")).getParent()
            .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");

    static boolean modelAvailable() {
        return Files.exists(MODEL_DIR.resolve("model.onnx"))
                && Files.exists(MODEL_DIR.resolve("model.onnx.data"))
                && Files.exists(MODEL_DIR.resolve("config.json"));
    }

    @Test
    @EnabledIf("modelAvailable")
    void load_weights() throws IOException {
        Phi3Config config = Phi3Config.load(MODEL_DIR.resolve("config.json"));

        try (Phi3Weights weights = Phi3Weights.load(MODEL_DIR, config)) {
            // Embedding
            assertNotNull(weights.embedTokens);
            assertEquals(config.vocabSize() * config.hiddenSize(), weights.embedTokens.length);
            // Verify embedding is not all zeros
            boolean hasNonZero = false;
            for (int i = 0; i < 100; i++) {
                if (weights.embedTokens[i] != 0) { hasNonZero = true; break; }
            }
            assertTrue(hasNonZero, "Embedding should contain non-zero values");

            // Cos/Sin cache
            assertNotNull(weights.cosCache);
            assertEquals(config.maxPositionEmbeddings() * (config.headDim() / 2),
                    weights.cosCache.length);
            assertNotNull(weights.sinCache);
            assertEquals(weights.cosCache.length, weights.sinCache.length);

            // Layers
            assertEquals(config.numHiddenLayers(), weights.layers.length);
            for (int l = 0; l < config.numHiddenLayers(); l++) {
                Phi3Weights.LayerWeights lw = weights.layers[l];
                assertNotNull(lw, "Layer " + l + " should not be null");
                assertNotNull(lw.inputNormWeight());
                assertEquals(config.hiddenSize(), lw.inputNormWeight().length);
                assertNotNull(lw.qProj());
                assertEquals(config.hiddenSize(), lw.qProj().N());
                assertEquals(config.hiddenSize(), lw.qProj().K());
                assertNotNull(lw.kProj());
                assertNotNull(lw.vProj());
                assertNotNull(lw.oProj());
                assertNotNull(lw.gateUpProj());
                assertEquals(config.intermediateSize() * 2, lw.gateUpProj().N());
                assertNotNull(lw.downProj());
                assertEquals(config.hiddenSize(), lw.downProj().N());
            }

            // Final norm
            assertNotNull(weights.finalNormWeight);
            assertEquals(config.hiddenSize(), weights.finalNormWeight.length);

            // LM head
            assertNotNull(weights.lmHead);
            assertEquals(config.vocabSize(), weights.lmHead.N());
            assertEquals(config.hiddenSize(), weights.lmHead.K());
        }
    }

    @Test
    @EnabledIf("modelAvailable")
    void fp16_conversion_correctness() {
        // Test known fp16 values
        assertEquals(1.0f, Phi3Weights.fp16ToFp32((short) 0x3C00), 1e-6f);   // 1.0
        assertEquals(-1.0f, Phi3Weights.fp16ToFp32((short) 0xBC00), 1e-6f);  // -1.0
        assertEquals(0.5f, Phi3Weights.fp16ToFp32((short) 0x3800), 1e-6f);   // 0.5
        assertEquals(0.0f, Phi3Weights.fp16ToFp32((short) 0x0000), 1e-6f);   // 0.0
        // Inf
        assertTrue(Float.isInfinite(Phi3Weights.fp16ToFp32((short) 0x7C00)));
        // NaN
        assertTrue(Float.isNaN(Phi3Weights.fp16ToFp32((short) 0x7E00)));
    }

    @Test
    @EnabledIf("modelAvailable")
    void quantized_matvec_basic() throws IOException {
        Phi3Config config = Phi3Config.load(MODEL_DIR.resolve("config.json"));

        try (Phi3Weights weights = Phi3Weights.load(MODEL_DIR, config)) {
            // Test that matvec produces non-zero output
            Phi3Weights.QuantizedWeight qProj = weights.layers[0].qProj();
            float[] input = new float[qProj.K()];
            // Use first embedding row as test input
            System.arraycopy(weights.embedTokens, 0, input, 0, qProj.K());

            float[] output = new float[qProj.N()];
            qProj.matvec(input, output);

            boolean anyNonZero = false;
            for (float v : output) {
                if (v != 0f) { anyNonZero = true; break; }
            }
            assertTrue(anyNonZero, "Quantized matvec should produce non-zero output");
        }
    }
}
