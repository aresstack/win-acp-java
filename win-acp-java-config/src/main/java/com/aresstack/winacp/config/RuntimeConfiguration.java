package com.aresstack.winacp.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.ArrayList;
import java.util.List;

/**
 * Top-level runtime configuration loaded from an external YAML/JSON file.
 * <p>
 * This is the root of the entire agent configuration and contains all
 * configurable sections defined in the product requirements (§8.6).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class RuntimeConfiguration {

    // --- Agent metadata ---
    private AgentProfile profile;

    // --- Behavior ---
    private AgentBehaviorDefinition behavior;

    // --- MCP servers & tool rules ---
    private List<McpServerDefinition> mcpServers = new ArrayList<>();
    private ToolPolicy toolPolicy;
    private ApprovalPolicy approvalPolicy;

    // --- Inference ---
    private InferenceConfiguration inference;

    // --- Logging ---
    private String logLevel = "INFO";
    private boolean debugGraphTracing = false;

    public RuntimeConfiguration() {}

    // --- Getters / setters ---

    public AgentProfile getProfile() { return profile; }
    public void setProfile(AgentProfile profile) { this.profile = profile; }

    public AgentBehaviorDefinition getBehavior() { return behavior; }
    public void setBehavior(AgentBehaviorDefinition behavior) { this.behavior = behavior; }

    public List<McpServerDefinition> getMcpServers() { return mcpServers; }
    public void setMcpServers(List<McpServerDefinition> mcpServers) { this.mcpServers = mcpServers; }

    public ToolPolicy getToolPolicy() { return toolPolicy; }
    public void setToolPolicy(ToolPolicy toolPolicy) { this.toolPolicy = toolPolicy; }

    public ApprovalPolicy getApprovalPolicy() { return approvalPolicy; }
    public void setApprovalPolicy(ApprovalPolicy approvalPolicy) { this.approvalPolicy = approvalPolicy; }

    public InferenceConfiguration getInference() { return inference; }
    public void setInference(InferenceConfiguration inference) { this.inference = inference; }

    public String getLogLevel() { return logLevel; }
    public void setLogLevel(String logLevel) { this.logLevel = logLevel; }

    public boolean isDebugGraphTracing() { return debugGraphTracing; }
    public void setDebugGraphTracing(boolean debugGraphTracing) { this.debugGraphTracing = debugGraphTracing; }
}

