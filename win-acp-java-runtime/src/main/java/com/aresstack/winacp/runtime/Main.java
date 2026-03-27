package com.aresstack.winacp.runtime;

import com.aresstack.winacp.acp.AcpAgentServer;
import com.aresstack.winacp.config.*;
import com.aresstack.winacp.graph.LangGraphAgentRunner;
import com.aresstack.winacp.inference.InferenceEngine;
import com.aresstack.winacp.inference.StubInferenceEngine;
import com.aresstack.winacp.mcp.McpClientManager;
import com.aresstack.winacp.mcp.ToolRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.List;

/**
 * Main entry point for the win-acp-java agent process.
 * <p>
 * Started as a subprocess by an ACP client. Loads configuration,
 * validates it, wires all layers, and enters the ACP message loop.
 *
 * <pre>
 * java --enable-native-access=ALL-UNNAMED -jar win-acp-java-runtime.jar --config application.yml
 * </pre>
 */
public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);
    private static final String DEFAULT_CONFIG = "application.yml";

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

            // 2. Initialize inference engine
            InferenceEngine inferenceEngine = new StubInferenceEngine();
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

            // 4. Build + wire behavior graph
            LangGraphAgentRunner graphRunner = new LangGraphAgentRunner(config.getBehavior());
            graphRunner.registerDefaults();
            log.info("Behavior graph: startNode='{}', {} node(s), {} edge(s)",
                    config.getBehavior().getStartNode(),
                    config.getBehavior().getNodes().size(),
                    config.getBehavior().getEdges().size());

            // 5. Start ACP server (blocks on stdin)
            AcpAgentServer acpServer = new AcpAgentServer(graphRunner);
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

    private static String resolveConfigPath(String[] args) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--config".equals(args[i])) return args[i + 1];
        }
        String env = System.getenv("WIN_ACP_CONFIG");
        if (env != null && !env.isBlank()) return env;
        return DEFAULT_CONFIG;
    }
}
