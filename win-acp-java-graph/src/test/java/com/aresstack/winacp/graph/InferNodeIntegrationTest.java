package com.aresstack.winacp.graph;

import com.aresstack.winacp.config.*;
import com.aresstack.winacp.inference.InferenceEngine;
import com.aresstack.winacp.inference.InferenceRequest;
import com.aresstack.winacp.inference.InferenceResult;
import com.aresstack.winacp.inference.StubInferenceEngine;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that the INFER node type works end-to-end through the
 * LangGraph4j graph runner, calling the InferenceEngine and
 * producing a real response in AgentState.
 */
class InferNodeIntegrationTest {

    @Test
    void inferNode_producesResponseFromEngine() throws Exception {
        // Use stub engine (deterministic output)
        StubInferenceEngine engine = new StubInferenceEngine();
        engine.initialize();

        LangGraphAgentRunner runner = new LangGraphAgentRunner(inferBehavior());
        runner.setInferenceEngine(engine);
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("What is the capital of France?");
        state.setSystemRole("You are a helpful assistant.");

        AgentState result = runner.run(state);

        assertNotNull(result.getPendingResponse(), "should have a response");
        assertFalse(result.getPendingResponse().isEmpty(), "response should not be empty");
        assertNotNull(result.getInferenceText(), "inference text should be set");
        assertTrue(result.getIterationCount() > 0, "should have executed nodes");

        engine.shutdown();
    }

    @Test
    void inferNode_withCustomEngine_getsCalledCorrectly() throws Exception {
        // Custom engine that returns a predictable response
        InferenceEngine customEngine = new InferenceEngine() {
            private boolean ready = false;

            @Override public void initialize() { ready = true; }
            @Override public void shutdown() { ready = false; }
            @Override public boolean isReady() { return ready; }

            @Override
            public InferenceResult generate(InferenceRequest request) {
                assertEquals("You are a test bot.", request.getSystemPrompt());
                return new InferenceResult("Paris is the capital of France.", "end_turn",
                        new InferenceResult.Usage(5, 7, 12));
            }
        };
        customEngine.initialize();

        LangGraphAgentRunner runner = new LangGraphAgentRunner(inferBehavior());
        runner.setInferenceEngine(customEngine);
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("What is the capital of France?");
        state.setSystemRole("You are a test bot.");

        AgentState result = runner.run(state);

        assertEquals("Paris is the capital of France.", result.getPendingResponse());
        assertEquals("Paris is the capital of France.", result.getInferenceText());
    }

    @Test
    void inferNode_errorDoesNotCrashGraph() throws Exception {
        InferenceEngine failingEngine = new InferenceEngine() {
            @Override public void initialize() {}
            @Override public void shutdown() {}
            @Override public boolean isReady() { return true; }

            @Override
            public InferenceResult generate(InferenceRequest request) {
                throw new RuntimeException("GPU exploded");
            }
        };

        LangGraphAgentRunner runner = new LangGraphAgentRunner(inferWithErrorHandling());
        runner.setInferenceEngine(failingEngine);
        runner.registerDefaults();

        AgentState state = new AgentState();
        state.setUserInput("test");

        AgentState result = runner.run(state);

        // The graph should handle the error gracefully
        assertNotNull(result.getPendingResponse(), "should still produce a response");
    }

    // ---- helpers ----

    /**
     * Minimal graph: analyze → goal → infer → respond → finalize
     */
    private AgentBehaviorDefinition inferBehavior() {
        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setMaxIterations(10);
        behavior.setNodes(new ArrayList<>(List.of(
                node("analyze", NodeType.ANALYZE_INPUT),
                node("goal", NodeType.DETERMINE_GOAL),
                node("infer", NodeType.INFER),
                node("respond", NodeType.FORMULATE_RESPONSE),
                node("finalize", NodeType.FINALIZE)
        )));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "goal", RoutingCondition.ALWAYS),
                edge("goal", "infer", RoutingCondition.ALWAYS),
                edge("infer", "respond", RoutingCondition.ALWAYS),
                edge("respond", "finalize", RoutingCondition.ALWAYS)
        )));
        return behavior;
    }

    /**
     * Graph with error handling: analyze → infer → respond → finalize
     * If infer sets error, respond still runs and finalize completes.
     */
    private AgentBehaviorDefinition inferWithErrorHandling() {
        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setMaxIterations(10);
        behavior.setNodes(new ArrayList<>(List.of(
                node("analyze", NodeType.ANALYZE_INPUT),
                node("infer", NodeType.INFER),
                node("respond", NodeType.FORMULATE_RESPONSE),
                node("finalize", NodeType.FINALIZE)
        )));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "infer", RoutingCondition.ALWAYS),
                edge("infer", "respond", RoutingCondition.ALWAYS),
                edge("respond", "finalize", RoutingCondition.ALWAYS)
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

