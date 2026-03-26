package com.aresstack.winacp.mcp;

import java.util.List;

/**
 * Describes a single tool exposed by an MCP server.
 */
public class ToolDescriptor {

    private final String serverId;
    private final String name;
    private final String description;
    private final List<String> parameterNames;

    public ToolDescriptor(String serverId, String name, String description, List<String> parameterNames) {
        this.serverId = serverId;
        this.name = name;
        this.description = description;
        this.parameterNames = parameterNames;
    }

    public String getServerId() { return serverId; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public List<String> getParameterNames() { return parameterNames; }

    /** Fully qualified tool identifier: serverId/toolName */
    public String getQualifiedName() {
        return serverId + "/" + name;
    }
}

