package com.aresstack.winacp.config;

import java.util.HashMap;
import java.util.Map;

/**
 * Connection details for a single MCP server.
 */
public class McpServerDefinition {

    private String id;
    private String name;
    private String transport;           // "stdio" | "sse" | "streamable-http"
    private String command;             // for stdio transport
    private String url;                 // for HTTP-based transports
    private Map<String, String> env = new HashMap<>();
    private int timeoutSeconds = 30;
    private Map<String, String> authentication = new HashMap<>();

    public McpServerDefinition() {}

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getTransport() { return transport; }
    public void setTransport(String transport) { this.transport = transport; }

    public String getCommand() { return command; }
    public void setCommand(String command) { this.command = command; }

    public String getUrl() { return url; }
    public void setUrl(String url) { this.url = url; }

    public Map<String, String> getEnv() { return env; }
    public void setEnv(Map<String, String> env) { this.env = env; }

    public int getTimeoutSeconds() { return timeoutSeconds; }
    public void setTimeoutSeconds(int timeoutSeconds) { this.timeoutSeconds = timeoutSeconds; }

    public Map<String, String> getAuthentication() { return authentication; }
    public void setAuthentication(Map<String, String> authentication) { this.authentication = authentication; }
}

