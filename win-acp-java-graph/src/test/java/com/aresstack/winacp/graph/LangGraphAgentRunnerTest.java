package com.aresstack.winacp.graph;

import com.aresstack.winacp.config.*;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LangGraphAgentRunner} – verifies that the agent behavior
 * graph is built and executed through the real LangGraph4j runtime.
 */
class LangGraphAgentRunnerTest {

    @Test
    void directAnswerPath_throughLangGraph4j() {
        LangGraphAgentRunner runner = new LangGraphAgentRunner(minimalBehavior());
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("Hello, how are you?");

        AgentState result = runner.run(state);

        assertNotNull(result.getPendingResponse(), "should produce a response via LangGraph4j");
        assertFalse(result.isNeedsTool(), "simple greeting should not need tool");
        assertTrue(result.getIterationCount() > 0, "should have executed at least one node");
    }

    @Test
    void toolPath_triggersToolRoute() {
        LangGraphAgentRunner runner = new LangGraphAgentRunner(behaviorWithToolPath());
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("Please search for documents about Java");

        AgentState result = runner.run(state);

        assertTrue(result.isNeedsTool(), "'search' keyword should trigger tool path");
        assertNotNull(result.getPendingResponse());
    }

    @Test
    void customNodeOverridesDefault() {
        LangGraphAgentRunner runner = new LangGraphAgentRunner(minimalBehavior());
        runner.registerDefaults();

        runner.registerNode(NodeType.FORMULATE_RESPONSE, s -> {
            s.setPendingResponse("LG4J_CUSTOM_RESPONSE");
            return s;
        });

        AgentState state = new AgentState();
        state.setUserInput("test");

        AgentState result = runner.run(state);
        assertEquals("LG4J_CUSTOM_RESPONSE", result.getPendingResponse());
    }

    @Test
    void errorOnMissingNodeImplementation() {
        var unknown = new NodeDefinition();
        unknown.setId("mystery");
        unknown.setType(NodeType.LOAD_CONTEXT); // registered but not for this test

        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("mystery");
        behavior.setMaxIterations(5);
        behavior.setNodes(new ArrayList<>(List.of(unknown)));
        behavior.setEdges(new ArrayList<>());

        // Runner without registering defaults → LOAD_CONTEXT has no implementation
        LangGraphAgentRunner runner = new LangGraphAgentRunner(behavior);

        AgentState state = new AgentState();
        state.setUserInput("test");

        AgentState result = runner.run(state);
        // Should handle error gracefully
        assertNotNull(result.getError(), "should set error when node implementation missing");
    }

    // ---- helpers ----

    private AgentBehaviorDefinition minimalBehavior() {
        var analyze = node("analyze", NodeType.ANALYZE_INPUT);
        var goal = node("goal", NodeType.DETERMINE_GOAL);
        var respond = node("respond", NodeType.FORMULATE_RESPONSE);
        var fin = node("finalize", NodeType.FINALIZE);

        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setMaxIterations(10);
        behavior.setNodes(new ArrayList<>(List.of(analyze, goal, respond, fin)));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "goal", RoutingCondition.ALWAYS),
                edge("goal", "respond", RoutingCondition.ALWAYS),
                edge("respond", "finalize", RoutingCondition.ALWAYS)
        )));
        return behavior;
    }

    private AgentBehaviorDefinition behaviorWithToolPath() {
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

