package com.aresstack.winacp.runtime;

import com.aresstack.winacp.acp.AcpAgentServer;
import com.aresstack.winacp.config.*;
import com.aresstack.winacp.graph.AgentGraphRunner;
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
        log.info("win-acp-java starting");

        try {
            // 1. Determine config path
            String configPath = resolveConfigPath(args);
            log.info("Configuration path: {}", configPath);

            // 2. Load configuration (fails fast if missing)
            ConfigLoader loader = new ConfigLoader();
            RuntimeConfiguration config = loader.load(Path.of(configPath));

            // 3. Validate configuration
            ConfigValidator validator = new ConfigValidator();
            List<String> errors = validator.validate(config);
            if (!errors.isEmpty()) {
                log.error("Configuration validation failed with {} error(s) – aborting", errors.size());
                System.exit(1);
            }

            // 4. Initialize MCP tool registry
            ToolRegistry toolRegistry = new ToolRegistry();
            for (McpServerDefinition server : config.getMcpServers()) {
                toolRegistry.discoverTools(server);
            }
            McpClientManager mcpClient = new McpClientManager(toolRegistry);
            log.info("MCP tool registry: {} tool(s) discovered", toolRegistry.getAllTools().size());

            // 5. Build behavior graph
            AgentGraphRunner graphRunner = new AgentGraphRunner(config.getBehavior());
            // TODO: register node implementations based on config.getBehavior().getNodes()

            // 6. Start ACP server (blocks until shutdown)
            AcpAgentServer server = new AcpAgentServer(graphRunner);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                log.info("Shutdown signal received");
                server.shutdown();
            }));
            server.start();

            log.info("win-acp-java agent is ready – waiting for ACP requests on stdin");

            // Keep the process alive until interrupted
            Thread.currentThread().join();

        } catch (InterruptedException e) {
            log.info("Agent interrupted – shutting down");
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            log.error("Fatal error during startup", e);
            System.exit(2);
        }
    }

    private static String resolveConfigPath(String[] args) {
        // --config <path>
        for (int i = 0; i < args.length - 1; i++) {
            if ("--config".equals(args[i])) {
                return args[i + 1];
            }
        }

        // Environment variable
        String envPath = System.getenv("WIN_ACP_CONFIG");
        if (envPath != null && !envPath.isBlank()) {
            return envPath;
        }

        return DEFAULT_CONFIG;
    }
}
