package com.aresstack.winacp.graph;

/**
 * Contract for a conditional edge (router) in the agent behavior graph.
 * <p>
 * Evaluates the current {@link AgentState} and returns the ID of the
 * next node to visit.
 */
@FunctionalInterface
public interface AgentRouter {

    /**
     * Determine the next node ID based on the current state.
     */
    String route(AgentState state);
}

