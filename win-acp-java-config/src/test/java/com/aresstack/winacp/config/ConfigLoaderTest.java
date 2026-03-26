package com.aresstack.winacp.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ConfigLoaderTest {

    private final ConfigLoader loader = new ConfigLoader();

    @Test
    void loadValidYaml(@TempDir Path tmp) throws IOException {
        Path yml = tmp.resolve("application.yml");
        Files.writeString(yml, """
                profile:
                  id: test
                  name: Test Agent
                  systemRole: You are a test agent.
                behavior:
                  startNode: analyze
                  nodes:
                    - id: analyze
                      type: ANALYZE_INPUT
                    - id: respond
                      type: FORMULATE_RESPONSE
                  edges:
                    - { from: analyze, to: respond, condition: ALWAYS }
                """);

        RuntimeConfiguration config = loader.load(yml);

        assertNotNull(config);
        assertEquals("test", config.getProfile().getId());
        assertEquals("Test Agent", config.getProfile().getName());
        assertEquals("analyze", config.getBehavior().getStartNode());
        assertEquals(2, config.getBehavior().getNodes().size());
        assertEquals(1, config.getBehavior().getEdges().size());
    }

    @Test
    void loadMissingFileThrowsWithHelpfulMessage() {
        Path missing = Path.of("does-not-exist.yml");

        IOException ex = assertThrows(IOException.class, () -> loader.load(missing));

        String msg = ex.getMessage();
        assertTrue(msg.contains("not found"), "should mention 'not found'");
        assertTrue(msg.contains("--config"), "should suggest --config flag");
        assertTrue(msg.contains("WIN_ACP_CONFIG"), "should suggest env variable");
        assertTrue(msg.contains("agent.example.yaml"), "should point to example file");
    }

    @Test
    void loadValidJson(@TempDir Path tmp) throws IOException {
        Path json = tmp.resolve("config.json");
        Files.writeString(json, """
                {
                  "profile": { "id": "json-test", "name": "JSON Agent" },
                  "behavior": {
                    "startNode": "analyze",
                    "nodes": [{ "id": "analyze", "type": "ANALYZE_INPUT" }],
                    "edges": []
                  }
                }
                """);

        RuntimeConfiguration config = loader.load(json);

        assertNotNull(config);
        assertEquals("json-test", config.getProfile().getId());
    }

    @Test
    void unknownFieldsAreIgnored(@TempDir Path tmp) throws IOException {
        Path yml = tmp.resolve("application.yml");
        Files.writeString(yml, """
                profile:
                  id: test
                  name: Test
                behavior:
                  startNode: a
                  nodes:
                    - id: a
                      type: ANALYZE_INPUT
                  edges: []
                someFutureField: value
                """);

        // Must not throw – forward compatibility
        assertDoesNotThrow(() -> loader.load(yml));
    }
}

