package com.aresstack.winacp.graph;

import com.aresstack.winacp.config.*;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.GraphStateException;
import org.bsc.langgraph4j.action.AsyncNodeAction;
import org.bsc.langgraph4j.action.NodeAction;
import org.bsc.langgraph4j.action.AsyncEdgeAction;
import org.bsc.langgraph4j.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Agent behavior graph runner powered by <b>LangGraph4j</b>.
 * <p>
 * Builds a real {@link StateGraph} from an {@link AgentBehaviorDefinition},
 * compiles it into a {@link CompiledGraph}, and executes it through the
 * LangGraph4j runtime. This replaces the hand-rolled walker in
 * {@link AgentGraphRunner} with the library's native graph engine.
 * <p>
 * State is stored as a {@code Map<String, Object>} inside LangGraph4j's
 * {@link org.bsc.langgraph4j.state.AgentState} and converted to/from
 * our {@link AgentState} POJO at each node boundary via
 * {@link AgentState#toMap()} / {@link AgentState#fromMap(Map)}.
 */
public class LangGraphAgentRunner {

    private static final Logger log = LoggerFactory.getLogger(LangGraphAgentRunner.class);

    private final AgentBehaviorDefinition behavior;
    private final Map<NodeType, AgentNode> nodeImplementations = new EnumMap<>(NodeType.class);

    public LangGraphAgentRunner(AgentBehaviorDefinition behavior) {
        this.behavior = Objects.requireNonNull(behavior, "behavior");
    }

    /** Register a concrete implementation for a given node type. */
    public void registerNode(NodeType type, AgentNode node) {
        nodeImplementations.put(type, node);
    }

    /** Register all default node implementations. */
    public void registerDefaults() {
        registerNode(NodeType.ANALYZE_INPUT,          DefaultNodes::analyzeInput);
        registerNode(NodeType.DETERMINE_GOAL,         DefaultNodes::determineGoal);
        registerNode(NodeType.LOAD_CONTEXT,           DefaultNodes::loadContext);
        registerNode(NodeType.SELECT_TOOL,            DefaultNodes::selectTool);
        registerNode(NodeType.EXECUTE_TOOL,           DefaultNodes::executeTool);
        registerNode(NodeType.EVALUATE_TOOL_RESULT,   DefaultNodes::evaluateToolResult);
        registerNode(NodeType.GENERATE_CLARIFICATION, DefaultNodes::generateClarification);
        registerNode(NodeType.FORMULATE_RESPONSE,     DefaultNodes::formulateResponse);
        registerNode(NodeType.FINALIZE,               DefaultNodes::finalize);
        registerNode(NodeType.HANDLE_ERROR,           DefaultNodes::handleError);
        registerNode(NodeType.REQUEST_APPROVAL,       DefaultNodes::requestApproval);
    }

    /**
     * Build, compile and execute the LangGraph4j state graph.
     *
     * @param initialState the initial agent state
     * @return the final agent state after graph execution
     */
    public AgentState run(AgentState initialState) {
        try {
            CompiledGraph<org.bsc.langgraph4j.state.AgentState> compiled = buildGraph();
            compiled.setMaxIterations(behavior.getMaxIterations());

            Map<String, Object> input = initialState.toMap();
            log.info("LangGraph4j execution starting at '{}'", behavior.getStartNode());

            Optional<org.bsc.langgraph4j.state.AgentState> result = compiled.invoke(input);

            if (result.isPresent()) {
                AgentState finalState = AgentState.fromMap(result.get().data());
                log.info("LangGraph4j execution complete after {} iteration(s)",
                        finalState.getIterationCount());
                return finalState;
            }

            log.warn("LangGraph4j returned no result – returning initial state");
            return initialState;

        } catch (Exception e) {
            log.error("LangGraph4j execution failed", e);
            initialState.setError("Graph execution failed: " + e.getMessage());
            if (initialState.getPendingResponse() == null) {
                initialState.setPendingResponse("An error occurred during processing.");
            }
            return initialState;
        }
    }

    // ---- internal graph construction ----

    private CompiledGraph<org.bsc.langgraph4j.state.AgentState> buildGraph() throws GraphStateException {
        StateGraph<org.bsc.langgraph4j.state.AgentState> graph =
                new StateGraph<>(org.bsc.langgraph4j.state.AgentState::new);

        // 1. Add all nodes from the behavior definition
        for (NodeDefinition nodeDef : behavior.getNodes()) {
            AgentNode impl = nodeImplementations.get(nodeDef.getType());
            if (impl == null) {
                throw new GraphStateException("No implementation registered for node type: " + nodeDef.getType());
            }
            graph.addNode(nodeDef.getId(), wrapAsLg4jAction(nodeDef.getId(), nodeDef.getType(), impl));
        }

        // 2. Wire START → first node
        graph.addEdge(StateGraph.START, behavior.getStartNode());

        // 3. Group edges by source node
        Map<String, List<EdgeDefinition>> edgesBySource = new LinkedHashMap<>();
        for (EdgeDefinition edge : behavior.getEdges()) {
            edgesBySource.computeIfAbsent(edge.getFrom(), k -> new ArrayList<>()).add(edge);
        }

        // 4. For each source node, add edges
        for (Map.Entry<String, List<EdgeDefinition>> entry : edgesBySource.entrySet()) {
            String from = entry.getKey();
            List<EdgeDefinition> edges = entry.getValue();

            if (edges.size() == 1 && isUnconditional(edges.getFirst())) {
                // Simple static edge
                String to = resolveTarget(edges.getFirst().getTo());
                graph.addEdge(from, to);
            } else {
                // Conditional edges – build route map and router
                Map<String, String> routeMap = new LinkedHashMap<>();
                for (EdgeDefinition edge : edges) {
                    String routeKey = edge.getTo();
                    String target = resolveTarget(edge.getTo());
                    routeMap.put(routeKey, target);
                }

                EdgeAction<org.bsc.langgraph4j.state.AgentState> router = lg4jState -> {
                    AgentState state = AgentState.fromMap(lg4jState.data());
                    for (EdgeDefinition edge : edges) {
                        if (evaluateCondition(edge.getCondition(), state)) {
                            return edge.getTo();
                        }
                    }
                    // Fallback: last edge or END
                    return edges.getLast().getTo();
                };

                graph.addConditionalEdges(from, AsyncEdgeAction.edge_async(router), routeMap);
            }
        }

        // 5. Terminal nodes (no outgoing edges) → connect to END
        Set<String> sourcesWithEdges = edgesBySource.keySet();
        for (NodeDefinition nodeDef : behavior.getNodes()) {
            if (!sourcesWithEdges.contains(nodeDef.getId())) {
                graph.addEdge(nodeDef.getId(), StateGraph.END);
            }
        }

        return graph.compile();
    }

    private AsyncNodeAction<org.bsc.langgraph4j.state.AgentState> wrapAsLg4jAction(
            String nodeId, NodeType type, AgentNode impl) {

        NodeAction<org.bsc.langgraph4j.state.AgentState> syncAction = lg4jState -> {
            AgentState state = AgentState.fromMap(lg4jState.data());
            state.incrementIteration();

            log.debug("Executing node '{}' (type: {})", nodeId, type);
            AgentState result = impl.execute(state);

            // Return the full updated state as a partial update map
            return result.toMap();
        };

        return AsyncNodeAction.node_async(syncAction);
    }

    private boolean isUnconditional(EdgeDefinition edge) {
        return edge.getCondition() == null || edge.getCondition() == RoutingCondition.ALWAYS;
    }

    private String resolveTarget(String target) {
        // Check if target is a known node; if not, it's a terminal → map to END
        boolean isKnownNode = behavior.getNodes().stream()
                .anyMatch(n -> n.getId().equals(target));
        return isKnownNode ? target : StateGraph.END;
    }

    private boolean evaluateCondition(RoutingCondition cond, AgentState state) {
        if (cond == null) return true;
        return switch (cond) {
            case TOOL_REQUIRED             -> state.isNeedsTool();
            case TOOL_NOT_REQUIRED         -> !state.isNeedsTool();
            case TOOL_ALLOWED              -> state.isToolAllowed();
            case TOOL_NOT_ALLOWED          -> !state.isToolAllowed();
            case APPROVAL_REQUIRED         -> state.isNeedsApproval();
            case APPROVAL_NOT_REQUIRED     -> !state.isNeedsApproval();
            case RESULT_SUFFICIENT         -> state.isResultSufficient();
            case RESULT_INSUFFICIENT       -> !state.isResultSufficient();
            case CLARIFICATION_REQUIRED    -> state.isNeedsClarification();
            case CLARIFICATION_NOT_REQUIRED -> !state.isNeedsClarification();
            case MAX_RETRIES_REACHED       -> state.getIterationCount() >= behavior.getMaxIterations();
            case COMPLETED                 -> state.getPendingResponse() != null;
            case ALWAYS                    -> true;
        };
    }
}

