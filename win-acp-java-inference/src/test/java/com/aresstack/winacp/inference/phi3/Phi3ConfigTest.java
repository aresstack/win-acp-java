package com.aresstack.winacp.inference.phi3;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Phi3Config}.
 */
class Phi3ConfigTest {

    /** Resolve model path relative to project root (parent of this submodule). */
    private static final Path CONFIG_PATH = Path.of(
            System.getProperty("user.dir")).getParent()
            .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128/config.json");

    static boolean configAvailable() {
        return Files.exists(CONFIG_PATH);
    }

    @Test
    @EnabledIf("configAvailable")
    void load_phi3_config() throws IOException {
        Phi3Config config = Phi3Config.load(CONFIG_PATH);

        assertEquals(3072, config.hiddenSize());
        assertEquals(32, config.numAttentionHeads());
        assertEquals(32, config.numHiddenLayers());
        assertEquals(32, config.numKeyValueHeads());
        assertEquals(32064, config.vocabSize());
        assertEquals(4096, config.maxPositionEmbeddings());
        assertEquals(8192, config.intermediateSize());
        assertEquals(1e-5f, config.rmsNormEps(), 1e-8f);
        assertEquals(10000.0f, config.ropeTheta(), 0.1f);
        assertEquals(96, config.headDim());
    }
}
