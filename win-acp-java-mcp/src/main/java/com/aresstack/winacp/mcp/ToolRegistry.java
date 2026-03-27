package com.aresstack.winacp.mcp;

import com.aresstack.winacp.config.McpServerDefinition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of all available MCP tools across all connected MCP servers.
 * Performs real tool discovery via {@link McpStdioClient}.
 */
public class ToolRegistry {

    private static final Logger log = LoggerFactory.getLogger(ToolRegistry.class);

    private final Map<String, List<ToolDescriptor>> toolsByServer = new ConcurrentHashMap<>();
    private final Map<String, McpStdioClient> clients = new ConcurrentHashMap<>();

    /**
     * Connect to an MCP server and discover its tools.
     */
    public void discoverTools(McpServerDefinition server) {
        log.info("Connecting to MCP server '{}' ({})", server.getName(), server.getTransport());
        try {
            McpStdioClient client = new McpStdioClient(server);
            client.connect();
            List<ToolDescriptor> tools = client.discoverTools();
            clients.put(server.getId(), client);
            toolsByServer.put(server.getId(), tools);
            log.info("MCP server '{}': {} tool(s) available", server.getName(), tools.size());
        } catch (IOException e) {
            log.error("Failed to connect to MCP server '{}': {}", server.getName(), e.getMessage());
            toolsByServer.put(server.getId(), List.of());
        }
    }

    /** Get the client for a given server. */
    public McpStdioClient getClient(String serverId) {
        return clients.get(serverId);
    }

    /** All known tools across all servers. */
    public List<ToolDescriptor> getAllTools() {
        return toolsByServer.values().stream().flatMap(List::stream).toList();
    }

    /** Tools from a specific server. */
    public List<ToolDescriptor> getToolsByServer(String serverId) {
        return toolsByServer.getOrDefault(serverId, List.of());
    }

    /** Find tool by qualified name (serverId/toolName). */
    public ToolDescriptor findTool(String qualifiedName) {
        return getAllTools().stream()
                .filter(t -> t.getQualifiedName().equals(qualifiedName))
                .findFirst().orElse(null);
    }

    /** Shut down all MCP server connections. */
    public void closeAll() {
        clients.values().forEach(McpStdioClient::close);
        clients.clear();
    }
}
