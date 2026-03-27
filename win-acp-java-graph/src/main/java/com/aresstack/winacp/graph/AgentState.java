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
    public static final String KEY_SYSTEM_ROLE = "systemRole";
    public static final String KEY_INFERENCE_TEXT = "inferenceText";

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
    private String systemRole;
    private String inferenceText;
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

    public String getSystemRole() { return systemRole; }
    public void setSystemRole(String systemRole) { this.systemRole = systemRole; }

    public String getInferenceText() { return inferenceText; }
    public void setInferenceText(String inferenceText) { this.inferenceText = inferenceText; }

    public List<ToolExecutionResult> getToolResults() { return toolResults; }
    public ToolExecutionResult getLastToolResult() {
        return toolResults.isEmpty() ? null : toolResults.getLast();
    }

    /**
     * Serialize this state to a Map for LangGraph4j interop.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put(KEY_USER_INPUT, userInput);
        map.put(KEY_GOAL, currentGoal);
        map.put(KEY_SELECTED_TOOL, selectedTool);
        map.put(KEY_RESPONSE, pendingResponse);
        map.put(KEY_APPROVAL, approvalGranted);
        map.put(KEY_ITERATION, iterationCount);
        map.put(KEY_NEEDS_TOOL, needsTool);
        map.put(KEY_TOOL_ALLOWED, toolAllowed);
        map.put(KEY_NEEDS_APPROVAL, needsApproval);
        map.put(KEY_RESULT_SUFFICIENT, resultSufficient);
        map.put(KEY_NEEDS_CLARIFICATION, needsClarification);
        map.put(KEY_ERROR, error);
        map.put(KEY_SYSTEM_ROLE, systemRole);
        map.put(KEY_INFERENCE_TEXT, inferenceText);
        map.put(KEY_TOOL_RESULTS, new ArrayList<>(toolResults));
        return map;
    }

    /**
     * Restore state from a LangGraph4j state map.
     */
    @SuppressWarnings("unchecked")
    public static AgentState fromMap(Map<String, Object> map) {
        AgentState state = new AgentState();
        state.userInput = (String) map.get(KEY_USER_INPUT);
        state.currentGoal = (String) map.get(KEY_GOAL);
        state.selectedTool = (String) map.get(KEY_SELECTED_TOOL);
        state.pendingResponse = (String) map.get(KEY_RESPONSE);
        state.approvalGranted = Boolean.TRUE.equals(map.get(KEY_APPROVAL));
        state.iterationCount = map.get(KEY_ITERATION) instanceof Number n ? n.intValue() : 0;
        state.needsTool = Boolean.TRUE.equals(map.get(KEY_NEEDS_TOOL));
        state.toolAllowed = map.containsKey(KEY_TOOL_ALLOWED) ? Boolean.TRUE.equals(map.get(KEY_TOOL_ALLOWED)) : true;
        state.needsApproval = Boolean.TRUE.equals(map.get(KEY_NEEDS_APPROVAL));
        state.resultSufficient = Boolean.TRUE.equals(map.get(KEY_RESULT_SUFFICIENT));
        state.needsClarification = Boolean.TRUE.equals(map.get(KEY_NEEDS_CLARIFICATION));
        state.error = (String) map.get(KEY_ERROR);
        state.systemRole = (String) map.get(KEY_SYSTEM_ROLE);
        state.inferenceText = (String) map.get(KEY_INFERENCE_TEXT);
        Object results = map.get(KEY_TOOL_RESULTS);
        if (results instanceof List<?> list) {
            for (Object item : list) {
                if (item instanceof ToolExecutionResult r) {
                    state.toolResults.add(r);
                }
            }
        }
        return state;
    }
}
