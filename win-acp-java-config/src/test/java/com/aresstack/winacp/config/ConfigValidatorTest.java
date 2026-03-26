package com.aresstack.winacp.config;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ConfigValidatorTest {

    private final ConfigValidator validator = new ConfigValidator();

    @Test
    void validMinimalConfigPasses() {
        RuntimeConfiguration config = minimalValidConfig();
        List<String> errors = validator.validate(config);
        assertTrue(errors.isEmpty(), "Expected no errors but got: " + errors);
    }

    @Test
    void missingProfileIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();
        config.setProfile(null);

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("profile")),
                "Expected profile error but got: " + errors);
    }

    @Test
    void missingBehaviorIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();
        config.setBehavior(null);

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("behavior")),
                "Expected behavior error but got: " + errors);
    }

    @Test
    void missingStartNodeIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();
        config.getBehavior().setStartNode(null);

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("start node")),
                "Expected start node error but got: " + errors);
    }

    @Test
    void emptyNodesIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();
        config.getBehavior().setNodes(List.of());

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("at least one node")),
                "Expected nodes error but got: " + errors);
    }

    @Test
    void startNodeNotInNodesIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();
        config.getBehavior().setStartNode("nonexistent");

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("nonexistent")),
                "Expected dangling start node error but got: " + errors);
    }

    @Test
    void edgeReferencingUnknownNodeIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();

        EdgeDefinition bad = new EdgeDefinition();
        bad.setFrom("analyze");
        bad.setTo("ghost");
        bad.setCondition(RoutingCondition.ALWAYS);
        config.getBehavior().setEdges(List.of(bad));

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("ghost")),
                "Expected unknown target node error but got: " + errors);
    }

    @Test
    void nodeWithoutTypeIsRejected() {
        RuntimeConfiguration config = minimalValidConfig();
        config.getBehavior().getNodes().getFirst().setType(null);

        List<String> errors = validator.validate(config);

        assertTrue(errors.stream().anyMatch(e -> e.contains("no type")),
                "Expected missing type error but got: " + errors);
    }

    // --- helper ---

    private RuntimeConfiguration minimalValidConfig() {
        AgentProfile profile = new AgentProfile();
        profile.setId("test");
        profile.setName("Test");

        NodeDefinition node = new NodeDefinition();
        node.setId("analyze");
        node.setType(NodeType.ANALYZE_INPUT);

        AgentBehaviorDefinition behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setNodes(new java.util.ArrayList<>(List.of(node)));
        behavior.setEdges(List.of());

        RuntimeConfiguration config = new RuntimeConfiguration();
        config.setProfile(profile);
        config.setBehavior(behavior);
        return config;
    }
}

