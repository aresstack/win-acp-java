package com.aresstack.winacp.config;

/**
 * Definition of an edge (transition) in the agent behavior graph.
 * <p>
 * Edges connect nodes and optionally carry a routing condition
 * drawn from a predefined set ({@link RoutingCondition}).
 */
public class EdgeDefinition {

    private String from;
    private String to;
    private RoutingCondition condition;

    public EdgeDefinition() {}

    public String getFrom() { return from; }
    public void setFrom(String from) { this.from = from; }

    public String getTo() { return to; }
    public void setTo(String to) { this.to = to; }

    public RoutingCondition getCondition() { return condition; }
    public void setCondition(RoutingCondition condition) { this.condition = condition; }
}

