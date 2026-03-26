package com.aresstack.winacp.mcp;

import com.aresstack.winacp.config.McpServerDefinition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of all available MCP tools across all connected MCP servers.
 * <p>
 * Discovers tools at startup and exposes them as {@link ToolDescriptor} instances.
 */
public class ToolRegistry {

    private static final Logger log = LoggerFactory.getLogger(ToolRegistry.class);

    private final Map<String, List<ToolDescriptor>> toolsByServer = new ConcurrentHashMap<>();

    /**
     * Register tools from a given MCP server definition.
     * TODO: actual MCP SDK discovery call
     */
    public void discoverTools(McpServerDefinition server) {
        log.info("Discovering tools from MCP server: {} ({})", server.getName(), server.getTransport());
        // TODO: connect to MCP server and list tools
        toolsByServer.put(server.getId(), new ArrayList<>());
    }

    /** All known tools across all servers. */
    public List<ToolDescriptor> getAllTools() {
        return toolsByServer.values().stream()
                .flatMap(List::stream)
                .toList();
    }

    /** Tools from a specific server. */
    public List<ToolDescriptor> getToolsByServer(String serverId) {
        return toolsByServer.getOrDefault(serverId, Collections.emptyList());
    }

    /** Find a tool by its qualified name (serverId/toolName). */
    public ToolDescriptor findTool(String qualifiedName) {
        return getAllTools().stream()
                .filter(t -> t.getQualifiedName().equals(qualifiedName))
                .findFirst()
                .orElse(null);
    }
}

