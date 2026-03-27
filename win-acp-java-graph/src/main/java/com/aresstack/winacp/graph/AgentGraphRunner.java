package com.aresstack.winacp.graph;

import com.aresstack.winacp.config.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Builds and executes the agent behavior graph based on an
 * {@link AgentBehaviorDefinition} loaded from external configuration.
 * <p>
 * Nodes are mapped from {@link NodeType} to concrete {@link AgentNode}
 * implementations. Edges and routing conditions drive the traversal.
 */
public class AgentGraphRunner {

    private static final Logger log = LoggerFactory.getLogger(AgentGraphRunner.class);

    private final Map<String, AgentNode> nodeById = new LinkedHashMap<>();
    private final Map<String, List<EdgeDefinition>> edgesFrom = new HashMap<>();
    private final AgentBehaviorDefinition behavior;
    private final Map<NodeType, AgentNode> nodeImplementations = new EnumMap<>(NodeType.class);
    private com.aresstack.winacp.inference.InferenceEngine inferenceEngine;

    public AgentGraphRunner(AgentBehaviorDefinition behavior) {
        this.behavior = behavior;
        indexEdges();
    }

    /** Inject the inference engine used by INFER nodes. */
    public void setInferenceEngine(com.aresstack.winacp.inference.InferenceEngine inferenceEngine) {
        this.inferenceEngine = inferenceEngine;
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

        if (inferenceEngine != null) {
            registerNode(NodeType.INFER, DefaultNodes.inferNode(inferenceEngine));
        }
    }

    /**
     * Execute the graph with the given initial state.
     * Walks through nodes following edges and routing conditions.
     */
    public AgentState run(AgentState state) {
        String currentNodeId = behavior.getStartNode();
        log.info("Graph execution starting at '{}'", currentNodeId);

        for (int i = 0; i < behavior.getMaxIterations(); i++) {
            state.incrementIteration();

            // Find and execute the current node
            NodeDefinition nodeDef = findNode(currentNodeId);
            if (nodeDef == null) {
                log.error("Node '{}' not found in behavior definition", currentNodeId);
                state.setError("Node not found: " + currentNodeId);
                break;
            }

            AgentNode impl = nodeImplementations.get(nodeDef.getType());
            if (impl == null) {
                log.error("No implementation for node type {}", nodeDef.getType());
                state.setError("No implementation for: " + nodeDef.getType());
                break;
            }

            log.debug("Executing node '{}' (type: {})", currentNodeId, nodeDef.getType());
            state = impl.execute(state);

            // Find the next node via edges
            String nextNodeId = resolveNextNode(currentNodeId, state);
            if (nextNodeId == null) {
                log.info("No outgoing edge from '{}' – graph complete", currentNodeId);
                break;
            }

            log.debug("Transition: '{}' → '{}'", currentNodeId, nextNodeId);
            currentNodeId = nextNodeId;
        }

        if (state.getIterationCount() >= behavior.getMaxIterations()) {
            log.warn("Max iterations ({}) reached", behavior.getMaxIterations());
            if (state.getPendingResponse() == null) {
                state.setPendingResponse(behavior.getAbortMessage());
            }
        }

        log.info("Graph execution complete after {} iteration(s)", state.getIterationCount());
        return state;
    }

    private void indexEdges() {
        for (EdgeDefinition edge : behavior.getEdges()) {
            edgesFrom.computeIfAbsent(edge.getFrom(), k -> new ArrayList<>()).add(edge);
        }
    }

    private NodeDefinition findNode(String id) {
        return behavior.getNodes().stream()
                .filter(n -> n.getId().equals(id))
                .findFirst().orElse(null);
    }

    private String resolveNextNode(String currentNodeId, AgentState state) {
        List<EdgeDefinition> candidates = edgesFrom.getOrDefault(currentNodeId, List.of());

        for (EdgeDefinition edge : candidates) {
            if (edge.getCondition() == null || edge.getCondition() == RoutingCondition.ALWAYS) {
                return edge.getTo();
            }
            if (evaluateCondition(edge.getCondition(), state)) {
                return edge.getTo();
            }
        }
        return null; // terminal
    }

    private boolean evaluateCondition(RoutingCondition cond, AgentState state) {
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
