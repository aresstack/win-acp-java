package com.aresstack.winacp.graph;

import com.aresstack.winacp.config.*;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AgentGraphRunnerTest {

    @Test
    void directAnswerPath_noTool() {
        AgentBehaviorDefinition behavior = minimalBehavior();
        AgentGraphRunner runner = new AgentGraphRunner(behavior);
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("Hello, how are you?");

        AgentState result = runner.run(state);

        assertNotNull(result.getPendingResponse(), "should produce a response");
        assertFalse(result.isNeedsTool(), "simple greeting should not need tool");
        assertTrue(result.getIterationCount() > 0, "should have executed at least one node");
    }

    @Test
    void toolPath_triggersTool() {
        AgentBehaviorDefinition behavior = minimalBehavior();
        AgentGraphRunner runner = new AgentGraphRunner(behavior);
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("Please search for documents about Java");

        AgentState result = runner.run(state);

        assertTrue(result.isNeedsTool(), "'search' keyword should trigger tool path");
        assertNotNull(result.getPendingResponse());
    }

    @Test
    void maxIterationsEnforced() {
        AgentBehaviorDefinition behavior = loopingBehavior();
        behavior.setMaxIterations(5);
        behavior.setAbortMessage("Aborted.");

        AgentGraphRunner runner = new AgentGraphRunner(behavior);
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("test");

        AgentState result = runner.run(state);

        assertTrue(result.getIterationCount() <= 5);
        assertEquals("Aborted.", result.getPendingResponse());
    }

    @Test
    void unknownNodeProducesError() {
        AgentBehaviorDefinition behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("nonexistent");
        behavior.setNodes(new ArrayList<>());
        behavior.setEdges(new ArrayList<>());

        AgentGraphRunner runner = new AgentGraphRunner(behavior);
        AgentState result = runner.run(new AgentState());

        assertNotNull(result.getError());
        assertTrue(result.getError().contains("nonexistent"));
    }

    @Test
    void customNodeImplementationOverridesDefault() {
        AgentBehaviorDefinition behavior = minimalBehavior();
        AgentGraphRunner runner = new AgentGraphRunner(behavior);
        runner.registerDefaults();

        // Override FORMULATE_RESPONSE with custom implementation
        runner.registerNode(NodeType.FORMULATE_RESPONSE, state -> {
            state.setPendingResponse("CUSTOM RESPONSE");
            return state;
        });

        AgentState state = new AgentState();
        state.setUserInput("hello");

        AgentState result = runner.run(state);
        assertEquals("CUSTOM RESPONSE", result.getPendingResponse());
    }

    // ---- helpers ----

    private AgentBehaviorDefinition minimalBehavior() {
        var analyze = node("analyze", NodeType.ANALYZE_INPUT);
        var goal = node("goal", NodeType.DETERMINE_GOAL);
        var selectTool = node("select_tool", NodeType.SELECT_TOOL);
        var error = node("error", NodeType.HANDLE_ERROR);
        var respond = node("respond", NodeType.FORMULATE_RESPONSE);
        var fin = node("finalize", NodeType.FINALIZE);

        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setMaxIterations(20);
        behavior.setAbortMessage("Max iterations.");
        behavior.setNodes(new ArrayList<>(List.of(analyze, goal, selectTool, error, respond, fin)));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "goal", RoutingCondition.ALWAYS),
                edge("goal", "select_tool", RoutingCondition.TOOL_REQUIRED),
                edge("goal", "respond", RoutingCondition.TOOL_NOT_REQUIRED),
                edge("select_tool", "error", RoutingCondition.TOOL_NOT_ALLOWED),
                edge("respond", "finalize", RoutingCondition.ALWAYS),
                edge("error", "finalize", RoutingCondition.ALWAYS)
        )));
        return behavior;
    }

    private AgentBehaviorDefinition loopingBehavior() {
        var analyze = node("analyze", NodeType.ANALYZE_INPUT);

        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setNodes(new ArrayList<>(List.of(analyze)));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "analyze", RoutingCondition.ALWAYS)
        )));
        return behavior;
    }

    private NodeDefinition node(String id, NodeType type) {
        var n = new NodeDefinition();
        n.setId(id);
        n.setType(type);
        return n;
    }

    private EdgeDefinition edge(String from, String to, RoutingCondition cond) {
        var e = new EdgeDefinition();
        e.setFrom(from);
        e.setTo(to);
        e.setCondition(cond);
        return e;
    }
}

