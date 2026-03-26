package com.aresstack.winacp.config;

import java.util.HashMap;
import java.util.Map;

/**
 * Definition of a single node in the agent behavior graph.
 * <p>
 * Each node has a fixed type drawn from a predefined set of categories
 * (see {@link NodeType}). Arbitrary code is not permitted.
 */
public class NodeDefinition {

    private String id;
    private NodeType type;
    private Map<String, String> parameters = new HashMap<>();

    public NodeDefinition() {}

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public NodeType getType() { return type; }
    public void setType(NodeType type) { this.type = type; }

    public Map<String, String> getParameters() { return parameters; }
    public void setParameters(Map<String, String> parameters) { this.parameters = parameters; }
}

