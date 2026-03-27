package com.aresstack.winacp.graph;

import com.aresstack.winacp.mcp.ToolExecutionResult;

import java.util.*;

/**
 * Mutable state object carried through the agent behavior graph.
 * <p>
 * Also provides conversion to/from {@code Map<String, Object>}
 * for LangGraph4j integration.
 */
public class AgentState {

    public static final String KEY_USER_INPUT = "userInput";
    public static final String KEY_GOAL = "currentGoal";
    public static final String KEY_SELECTED_TOOL = "selectedTool";
    public static final String KEY_RESPONSE = "pendingResponse";
    public static final String KEY_APPROVAL = "approvalGranted";
    public static final String KEY_ITERATION = "iterationCount";
    public static final String KEY_TOOL_RESULTS = "toolResults";
    public static final String KEY_NEEDS_TOOL = "needsTool";
    public static final String KEY_TOOL_ALLOWED = "toolAllowed";
    public static final String KEY_NEEDS_APPROVAL = "needsApproval";
    public static final String KEY_RESULT_SUFFICIENT = "resultSufficient";
    public static final String KEY_NEEDS_CLARIFICATION = "needsClarification";
    public static final String KEY_ERROR = "error";

    private String userInput;
    private String currentGoal;
    private String selectedTool;
    private String pendingResponse;
    private boolean approvalGranted;
    private int iterationCount;
    private boolean needsTool;
    private boolean toolAllowed = true;
    private boolean needsApproval;
    private boolean resultSufficient;
    private boolean needsClarification;
    private String error;
    private final List<ToolExecutionResult> toolResults = new ArrayList<>();

    // --- accessors ---

    public String getUserInput() { return userInput; }
    public void setUserInput(String userInput) { this.userInput = userInput; }

    public String getCurrentGoal() { return currentGoal; }
    public void setCurrentGoal(String currentGoal) { this.currentGoal = currentGoal; }

    public String getSelectedTool() { return selectedTool; }
    public void setSelectedTool(String selectedTool) { this.selectedTool = selectedTool; }

    public String getPendingResponse() { return pendingResponse; }
    public void setPendingResponse(String pendingResponse) { this.pendingResponse = pendingResponse; }

    public boolean isApprovalGranted() { return approvalGranted; }
    public void setApprovalGranted(boolean v) { this.approvalGranted = v; }

    public int getIterationCount() { return iterationCount; }
    public void incrementIteration() { this.iterationCount++; }

    public boolean isNeedsTool() { return needsTool; }
    public void setNeedsTool(boolean v) { this.needsTool = v; }

    public boolean isToolAllowed() { return toolAllowed; }
    public void setToolAllowed(boolean v) { this.toolAllowed = v; }

    public boolean isNeedsApproval() { return needsApproval; }
    public void setNeedsApproval(boolean v) { this.needsApproval = v; }

    public boolean isResultSufficient() { return resultSufficient; }
    public void setResultSufficient(boolean v) { this.resultSufficient = v; }

    public boolean isNeedsClarification() { return needsClarification; }
    public void setNeedsClarification(boolean v) { this.needsClarification = v; }

    public String getError() { return error; }
    public void setError(String error) { this.error = error; }

    public List<ToolExecutionResult> getToolResults() { return toolResults; }
    public ToolExecutionResult getLastToolResult() {
        return toolResults.isEmpty() ? null : toolResults.getLast();
    }
}
