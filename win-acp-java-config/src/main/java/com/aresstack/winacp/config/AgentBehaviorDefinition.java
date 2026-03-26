package com.aresstack.winacp.config;

import java.util.ArrayList;
import java.util.List;

/**
 * Declarative description of agent behavior.
 * <p>
 * Defines which nodes are active, how they are connected,
 * where the graph starts, and under which conditions it terminates.
 * This is loaded from external configuration – no arbitrary Java code.
 */
public class AgentBehaviorDefinition {

    private String startNode;
    private List<NodeDefinition> nodes = new ArrayList<>();
    private List<EdgeDefinition> edges = new ArrayList<>();
    private int maxIterations = 20;
    private String abortMessage;

    public AgentBehaviorDefinition() {}

    public String getStartNode() { return startNode; }
    public void setStartNode(String startNode) { this.startNode = startNode; }

    public List<NodeDefinition> getNodes() { return nodes; }
    public void setNodes(List<NodeDefinition> nodes) { this.nodes = nodes; }

    public List<EdgeDefinition> getEdges() { return edges; }
    public void setEdges(List<EdgeDefinition> edges) { this.edges = edges; }

    public int getMaxIterations() { return maxIterations; }
    public void setMaxIterations(int maxIterations) { this.maxIterations = maxIterations; }

    public String getAbortMessage() { return abortMessage; }
    public void setAbortMessage(String abortMessage) { this.abortMessage = abortMessage; }
}

