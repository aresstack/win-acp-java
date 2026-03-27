package com.aresstack.winacp.graph;

import com.aresstack.winacp.inference.InferenceEngine;
import com.aresstack.winacp.inference.InferenceRequest;
import com.aresstack.winacp.inference.InferenceResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementations for all {@link com.aresstack.winacp.config.NodeType}s.
 * <p>
 * These provide baseline behavior. Each can be replaced via
 * {@link AgentGraphRunner#registerNode}.
 */
public final class DefaultNodes {

    private static final Logger log = LoggerFactory.getLogger(DefaultNodes.class);

    private DefaultNodes() {}

    public static AgentState analyzeInput(AgentState state) {
        log.debug("ANALYZE_INPUT: '{}'", state.getUserInput());
        // Baseline: pass through – real implementation would classify input
        return state;
    }

    public static AgentState determineGoal(AgentState state) {
        String input = state.getUserInput();
        // Simple heuristic: if the input contains "search", "find", "lookup" → needs tool
        boolean needsTool = input != null &&
                (input.contains("search") || input.contains("find") || input.contains("lookup"));
        state.setNeedsTool(needsTool);
        state.setCurrentGoal(needsTool ? "tool-assisted-answer" : "direct-answer");
        log.debug("DETERMINE_GOAL: needsTool={}, goal='{}'", needsTool, state.getCurrentGoal());
        return state;
    }

    public static AgentState loadContext(AgentState state) {
        log.debug("LOAD_CONTEXT");
        return state;
    }

    public static AgentState selectTool(AgentState state) {
        log.debug("SELECT_TOOL");
        // Baseline: no tool selection logic yet → mark not allowed
        state.setToolAllowed(false);
        state.setNeedsApproval(false);
        return state;
    }

    public static AgentState executeTool(AgentState state) {
        log.debug("EXECUTE_TOOL: {}", state.getSelectedTool());
        // Baseline: no real tool execution
        state.setResultSufficient(false);
        return state;
    }

    public static AgentState evaluateToolResult(AgentState state) {
        var last = state.getLastToolResult();
        boolean sufficient = last != null && last.isSuccess();
        state.setResultSufficient(sufficient);
        log.debug("EVALUATE_TOOL_RESULT: sufficient={}", sufficient);
        return state;
    }

    public static AgentState generateClarification(AgentState state) {
        state.setPendingResponse("I could not find a sufficient answer. Could you provide more details?");
        log.debug("GENERATE_CLARIFICATION");
        return state;
    }

    public static AgentState formulateResponse(AgentState state) {
        if (state.getPendingResponse() == null) {
            // If inference produced text, use that as the response
            if (state.getInferenceText() != null && !state.getInferenceText().isBlank()) {
                state.setPendingResponse(state.getInferenceText());
            } else {
                String input = state.getUserInput() != null ? state.getUserInput() : "";
                state.setPendingResponse("Received: " + input);
            }
        }
        log.debug("FORMULATE_RESPONSE: '{}'", state.getPendingResponse());
        return state;
    }

    public static AgentState finalize(AgentState state) {
        log.debug("FINALIZE");
        return state;
    }

    public static AgentState handleError(AgentState state) {
        String err = state.getError() != null ? state.getError() : "unknown error";
        state.setPendingResponse("An error occurred: " + err);
        log.debug("HANDLE_ERROR: {}", err);
        return state;
    }

    public static AgentState requestApproval(AgentState state) {
        log.debug("REQUEST_APPROVAL");
        state.setPendingResponse("This action requires approval. Please confirm.");
        return state;
    }

    /**
     * Create an INFER node that calls the given {@link InferenceEngine}.
     * <p>
     * This is the critical node type: it takes the current state,
     * builds an {@link InferenceRequest}, runs real inference, and
     * stores the result text in {@code AgentState.inferenceText}.
     */
    public static AgentNode inferNode(InferenceEngine engine) {
        return state -> {
            log.debug("INFER: userInput='{}'",
                    state.getUserInput() != null && state.getUserInput().length() > 60
                            ? state.getUserInput().substring(0, 60) + "…"
                            : state.getUserInput());

            try {
                InferenceRequest request = InferenceRequest.builder()
                        .systemPrompt(state.getSystemRole() != null ? state.getSystemRole() : "")
                        .userPrompt(state.getUserInput() != null ? state.getUserInput() : "")
                        .maxTokens(256)
                        .temperature(0.7f)
                        .build();

                InferenceResult result = engine.generate(request);

                state.setInferenceText(result.getText());
                state.setPendingResponse(result.getText());

                log.info("INFER: finishReason={}, text length={}",
                        result.getFinishReason(),
                        result.getText() != null ? result.getText().length() : 0);

            } catch (Exception e) {
                log.error("INFER failed", e);
                state.setError("Inference failed: " + e.getMessage());
            }

            return state;
        };
    }
}
