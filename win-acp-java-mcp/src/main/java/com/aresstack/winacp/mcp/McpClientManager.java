package com.aresstack.winacp.mcp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Manages MCP tool execution across connected servers.
 */
public class McpClientManager {

    private static final Logger log = LoggerFactory.getLogger(McpClientManager.class);

    private final ToolRegistry toolRegistry;

    public McpClientManager(ToolRegistry toolRegistry) {
        this.toolRegistry = toolRegistry;
    }

    /**
     * Execute a tool call, routing to the appropriate MCP server.
     */
    public ToolExecutionResult execute(ToolExecutionRequest request) {
        log.info("Executing tool: {}/{}", request.getServerId(), request.getToolName());

        McpStdioClient client = toolRegistry.getClient(request.getServerId());
        if (client == null) {
            return ToolExecutionResult.failure(request.getToolName(),
                    "No connection to MCP server: " + request.getServerId());
        }

        try {
            return client.callTool(request.getToolName(), request.getArguments());
        } catch (IOException e) {
            log.error("Tool execution failed: {}", request, e);
            return ToolExecutionResult.failure(request.getToolName(), e.getMessage());
        }
    }

    public ToolRegistry getToolRegistry() {
        return toolRegistry;
    }
}
