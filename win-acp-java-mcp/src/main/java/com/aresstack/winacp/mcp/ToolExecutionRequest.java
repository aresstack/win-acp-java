package com.aresstack.winacp.mcp;

import java.util.Map;

/**
 * A request to execute a specific MCP tool.
 */
public class ToolExecutionRequest {

    private final String serverId;
    private final String toolName;
    private final Map<String, Object> arguments;

    public ToolExecutionRequest(String serverId, String toolName, Map<String, Object> arguments) {
        this.serverId = serverId;
        this.toolName = toolName;
        this.arguments = arguments;
    }

    public String getServerId() { return serverId; }
    public String getToolName() { return toolName; }
    public Map<String, Object> getArguments() { return arguments; }

    @Override
    public String toString() {
        return "ToolExecutionRequest{server='" + serverId + "', tool='" + toolName + "'}";
    }
}

