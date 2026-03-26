package com.aresstack.winacp.graph;

import com.aresstack.winacp.mcp.ToolExecutionResult;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Mutable state object carried through the agent behavior graph.
 * <p>
 * Holds the conversation context, tool results, iteration counter,
 * and all data needed by nodes to make routing decisions.
 */
public class AgentState {

    private String userInput;
    private String currentGoal;
    private String selectedTool;
    private String pendingResponse;
    private boolean approvalGranted;
    private int iterationCount;
    private final List<ToolExecutionResult> toolResults = new ArrayList<>();
    private final Map<String, Object> context = new HashMap<>();

    // --- core accessors ---

    public String getUserInput() { return userInput; }
    public void setUserInput(String userInput) { this.userInput = userInput; }

    public String getCurrentGoal() { return currentGoal; }
    public void setCurrentGoal(String currentGoal) { this.currentGoal = currentGoal; }

    public String getSelectedTool() { return selectedTool; }
    public void setSelectedTool(String selectedTool) { this.selectedTool = selectedTool; }

    public String getPendingResponse() { return pendingResponse; }
    public void setPendingResponse(String pendingResponse) { this.pendingResponse = pendingResponse; }

    public boolean isApprovalGranted() { return approvalGranted; }
    public void setApprovalGranted(boolean approvalGranted) { this.approvalGranted = approvalGranted; }

    public int getIterationCount() { return iterationCount; }
    public void incrementIteration() { this.iterationCount++; }

    public List<ToolExecutionResult> getToolResults() { return toolResults; }

    public ToolExecutionResult getLastToolResult() {
        return toolResults.isEmpty() ? null : toolResults.getLast();
    }

    public Map<String, Object> getContext() { return context; }
}

