package com.aresstack.winacp.graph;

/**
 * Contract for a single node in the agent behavior graph.
 * <p>
 * Each implementation corresponds to one {@link com.aresstack.winacp.config.NodeType}.
 * Nodes read from and write to the shared {@link AgentState}.
 */
@FunctionalInterface
public interface AgentNode {

    /**
     * Execute this node's logic and return the (possibly modified) state.
     */
    AgentState execute(AgentState state);
}

