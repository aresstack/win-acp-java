package com.aresstack.winacp.runtime;

import com.aresstack.winacp.acp.AcpAgentServer;
import com.aresstack.winacp.config.*;
import com.aresstack.winacp.graph.LangGraphAgentRunner;
import com.aresstack.winacp.inference.MnistDirectMlEngine;
import com.aresstack.winacp.inference.Phi3InferenceEngine;
import com.aresstack.winacp.inference.InferenceEngine;
import com.aresstack.winacp.inference.StubInferenceEngine;
import com.aresstack.winacp.mcp.McpClientManager;
import com.aresstack.winacp.mcp.ToolRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Main entry point for the win-acp-java agent process.
 * <p>
 * Started as a subprocess by an ACP client. Loads configuration,
 * validates it, wires all layers, and enters the ACP message loop.
 *
 * <pre>
 * java --enable-native-access=ALL-UNNAMED -jar win-acp-java-runtime.jar --config agent.example.yaml
 * </pre>
 */
public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    /** User-specific local config (gitignored). */
    private static final String LOCAL_CONFIG = "application.yml";
    /** Shipped example config (always present in the repo). */
    private static final String EXAMPLE_CONFIG = "agent.example.yaml";

    public static void main(String[] args) {
        log.info("win-acp-java 0.1.0-SNAPSHOT starting");

        try {
            // 1. Load + validate configuration
            String configPath = resolveConfigPath(args);
            log.info("Configuration: {}", configPath);

            ConfigLoader loader = new ConfigLoader();
            RuntimeConfiguration config = loader.load(Path.of(configPath));

            ConfigValidator validator = new ConfigValidator();
            List<String> errors = validator.validate(config);
            if (!errors.isEmpty()) {
                log.error("Configuration invalid ({} error(s)) – aborting", errors.size());
                System.exit(1);
            }

            // 2. Initialize inference engine – real or stub based on config
            InferenceEngine inferenceEngine = createInferenceEngine(config.getInference());
            inferenceEngine.initialize();
            log.info("Inference engine: {} (ready={})",
                    inferenceEngine.getClass().getSimpleName(), inferenceEngine.isReady());

            // 3. Initialize MCP tool registry
            ToolRegistry toolRegistry = new ToolRegistry();
            for (McpServerDefinition server : config.getMcpServers()) {
                toolRegistry.discoverTools(server);
            }
            McpClientManager mcpClient = new McpClientManager(toolRegistry);
            log.info("MCP: {} tool(s) from {} server(s)",
                    toolRegistry.getAllTools().size(), config.getMcpServers().size());

            // 4. Build + wire behavior graph with InferenceEngine
            LangGraphAgentRunner graphRunner = new LangGraphAgentRunner(config.getBehavior());
            graphRunner.setInferenceEngine(inferenceEngine);
            graphRunner.registerDefaults();
            log.info("Behavior graph: startNode='{}', {} node(s), {} edge(s)",
                    config.getBehavior().getStartNode(),
                    config.getBehavior().getNodes().size(),
                    config.getBehavior().getEdges().size());

            // 5. Start ACP server (blocks on stdin)
            String systemRole = config.getProfile() != null ? config.getProfile().getSystemRole() : null;
            AcpAgentServer acpServer = new AcpAgentServer(graphRunner, systemRole);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                log.info("Shutdown signal received");
                acpServer.shutdown();
                inferenceEngine.shutdown();
                toolRegistry.closeAll();
            }));

            log.info("win-acp-java ready – listening for JSON-RPC on stdin");
            acpServer.start(); // blocks until stdin closes or shutdown

            log.info("win-acp-java stopped");

        } catch (Exception e) {
            log.error("Fatal error during startup", e);
            System.exit(2);
        }
    }

    /**
     * Select the inference engine based on configuration.
     * <p>
     * Detection order:
     * <ol>
     *   <li><b>Phi-3</b>: If modelPath is a directory containing
     *       {@code config.json}, {@code tokenizer.json}, {@code model.onnx},
     *       and {@code model.onnx.data} → {@link Phi3InferenceEngine}</li>
     *   <li><b>MNIST</b>: If modelPath is an existing {@code .onnx} file →
     *       {@link MnistDirectMlEngine}</li>
     *   <li><b>Stub</b>: Fallback → {@link StubInferenceEngine}</li>
     * </ol>
     */
    static InferenceEngine createInferenceEngine(InferenceConfiguration inferenceConfig) {
        if (inferenceConfig == null
                || inferenceConfig.getModelPath() == null
                || inferenceConfig.getModelPath().isBlank()) {
            log.info("No model path configured – using StubInferenceEngine");
            return new StubInferenceEngine();
        }

        Path modelPath = Path.of(inferenceConfig.getModelPath());

        // 1. Phi-3 model directory?
        if (Phi3InferenceEngine.isValidModelDir(modelPath)) {
            log.info("Detected Phi-3 model directory: {}", modelPath);
            return new Phi3InferenceEngine(inferenceConfig);
        }

        // 2. MNIST .onnx file?
        if (Files.exists(modelPath) && modelPath.toString().endsWith(".onnx")) {
            log.info("Using MnistDirectMlEngine (model={})", modelPath);
            return new MnistDirectMlEngine(inferenceConfig);
        }

        log.info("Model path '{}' not recognized – using StubInferenceEngine", modelPath);
        return new StubInferenceEngine();
    }

    /**
     * Resolve configuration path.
     * <p>
     * Fallback chain:
     * <ol>
     *   <li>{@code --config <path>} CLI argument</li>
     *   <li>{@code WIN_ACP_CONFIG} environment variable</li>
     *   <li>{@code application.yml} in working directory (local override, gitignored)</li>
     *   <li>{@code agent.example.yaml} in working directory (shipped with repo)</li>
     * </ol>
     */
    static String resolveConfigPath(String[] args) {
        // 1. Explicit --config argument
        for (int i = 0; i < args.length - 1; i++) {
            if ("--config".equals(args[i])) return args[i + 1];
        }
        // 2. Environment variable
        String env = System.getenv("WIN_ACP_CONFIG");
        if (env != null && !env.isBlank()) return env;
        // 3. Local config (gitignored)
        if (Files.exists(Path.of(LOCAL_CONFIG))) return LOCAL_CONFIG;
        // 4. Shipped example
        return EXAMPLE_CONFIG;
    }
}
