package com.aresstack.winacp.runtime;

import com.aresstack.winacp.config.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Main} – verifies that the agent runtime starts, loads
 * configuration, wires all layers, and enters the ACP message loop.
 */
class MainTest {

    /**
     * When started with a valid config and stdin is immediately closed,
     * the agent should initialize all layers and then exit cleanly
     * (not crash or hang).
     */
    @Test
    void startWithValidConfig_andImmediateEof_exitsCleanly(@TempDir Path tmp) throws Exception {
        Path config = tmp.resolve("application.yml");
        Files.writeString(config, """
                profile:
                  id: test
                  name: Test Agent
                  description: Smoke test agent
                  systemRole: You are a test agent.

                behavior:
                  startNode: analyze
                  maxIterations: 10
                  abortMessage: Aborted.
                  nodes:
                    - id: analyze
                      type: ANALYZE_INPUT
                    - id: goal
                      type: DETERMINE_GOAL
                    - id: infer
                      type: INFER
                    - id: respond
                      type: FORMULATE_RESPONSE
                    - id: finalize
                      type: FINALIZE
                  edges:
                    - { from: analyze,  to: goal,     condition: ALWAYS }
                    - { from: goal,     to: infer,    condition: ALWAYS }
                    - { from: infer,    to: respond,  condition: ALWAYS }
                    - { from: respond,  to: finalize, condition: ALWAYS }

                mcpServers: []

                toolPolicy:
                  allowedTools: []
                  blockedTools: []
                  approvalRequiredTools: []
                  maxToolCallsPerTurn: 10

                approvalPolicy:
                  enabled: false
                  approvalRequiredByDefault: false

                inference:
                  modelPath: /dummy/model.onnx
                  backend: cpu
                  maxTokens: 100
                  temperature: 0.5

                logLevel: INFO
                debugGraphTracing: false
                """);

        // Capture original stdin/stdout
        InputStream originalIn = System.in;
        PrintStream originalOut = System.out;

        try {
            // Provide empty stdin (EOF immediately)
            System.setIn(new ByteArrayInputStream(new byte[0]));

            // Capture stdout
            ByteArrayOutputStream captured = new ByteArrayOutputStream();
            System.setOut(new PrintStream(captured));

            // Run main – should not throw
            assertDoesNotThrow(() ->
                    Main.main(new String[]{"--config", config.toString()}));

        } finally {
            System.setIn(originalIn);
            System.setOut(originalOut);
        }
    }

    /**
     * When no config file exists, the agent should fail with a helpful
     * error message and exit code 2.
     */
    @Test
    void startWithMissingConfig_logsErrorAndExits(@TempDir Path tmp) {
        Path missing = tmp.resolve("nonexistent.yml");

        // The agent calls System.exit() on fatal errors. We can't easily
        // test that in-process without a SecurityManager (removed in Java 21).
        // Instead, verify that ConfigLoader throws the right error.
        var loader = new ConfigLoader();
        IOException ex = assertThrows(IOException.class,
                () -> loader.load(missing));

        assertTrue(ex.getMessage().contains("not found"));
        assertTrue(ex.getMessage().contains("--config"));
        assertTrue(ex.getMessage().contains("WIN_ACP_CONFIG"));
    }

    /**
     * InferenceEngine selection: stub is used when model path does not exist.
     */
    @Test
    void createInferenceEngine_noModelFile_returnsStub() {
        var cfg = new InferenceConfiguration();
        cfg.setModelPath("/nonexistent/model.onnx");
        cfg.setBackend("cpu");

        var engine = Main.createInferenceEngine(cfg);
        assertTrue(engine instanceof com.aresstack.winacp.inference.StubInferenceEngine);
    }

    /**
     * InferenceEngine selection: null config returns stub.
     */
    @Test
    void createInferenceEngine_nullConfig_returnsStub() {
        var engine = Main.createInferenceEngine(null);
        assertTrue(engine instanceof com.aresstack.winacp.inference.StubInferenceEngine);
    }
}
