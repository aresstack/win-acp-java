package com.aresstack.winacp.config;

import java.util.ArrayList;
import java.util.List;

/**
 * Policy controlling which tools the agent may use and under what conditions.
 */
public class ToolPolicy {

    private List<String> allowedTools = new ArrayList<>();
    private List<String> blockedTools = new ArrayList<>();
    private List<String> approvalRequiredTools = new ArrayList<>();
    private int maxToolCallsPerTurn = 10;

    public ToolPolicy() {}

    public List<String> getAllowedTools() { return allowedTools; }
    public void setAllowedTools(List<String> allowedTools) { this.allowedTools = allowedTools; }

    public List<String> getBlockedTools() { return blockedTools; }
    public void setBlockedTools(List<String> blockedTools) { this.blockedTools = blockedTools; }

    public List<String> getApprovalRequiredTools() { return approvalRequiredTools; }
    public void setApprovalRequiredTools(List<String> approvalRequiredTools) { this.approvalRequiredTools = approvalRequiredTools; }

    public int getMaxToolCallsPerTurn() { return maxToolCallsPerTurn; }
    public void setMaxToolCallsPerTurn(int maxToolCallsPerTurn) { this.maxToolCallsPerTurn = maxToolCallsPerTurn; }
}

