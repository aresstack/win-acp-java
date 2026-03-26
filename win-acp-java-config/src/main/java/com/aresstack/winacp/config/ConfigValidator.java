package com.aresstack.winacp.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Validates a {@link RuntimeConfiguration} before the agent enters operational mode.
 * <p>
 * Invalid or unsafe configurations are rejected with clear error messages (§8.6).
 */
public class ConfigValidator {

    private static final Logger log = LoggerFactory.getLogger(ConfigValidator.class);

    /**
     * Validate the given configuration. Returns a list of error messages.
     * An empty list means the configuration is valid.
     */
    public List<String> validate(RuntimeConfiguration config) {
        List<String> errors = new ArrayList<>();

        if (config.getProfile() == null) {
            errors.add("Agent profile is required");
        }

        if (config.getBehavior() == null) {
            errors.add("Agent behavior definition is required");
        } else {
            validateBehavior(config.getBehavior(), errors);
        }

        if (config.getToolPolicy() == null) {
            log.warn("No tool policy defined – using restrictive defaults");
        }

        for (String error : errors) {
            log.error("Configuration error: {}", error);
        }

        if (errors.isEmpty()) {
            log.info("Configuration validation passed");
        }

        return errors;
    }

    private void validateBehavior(AgentBehaviorDefinition behavior, List<String> errors) {
        if (behavior.getStartNode() == null || behavior.getStartNode().isBlank()) {
            errors.add("Behavior must define a start node");
        }

        if (behavior.getNodes().isEmpty()) {
            errors.add("Behavior must define at least one node");
        }

        Set<String> nodeIds = behavior.getNodes().stream()
                .map(NodeDefinition::getId)
                .collect(Collectors.toSet());

        if (behavior.getStartNode() != null && !nodeIds.contains(behavior.getStartNode())) {
            errors.add("Start node '" + behavior.getStartNode() + "' not found in node definitions");
        }

        for (EdgeDefinition edge : behavior.getEdges()) {
            if (!nodeIds.contains(edge.getFrom())) {
                errors.add("Edge references unknown source node: " + edge.getFrom());
            }
            if (!nodeIds.contains(edge.getTo())) {
                errors.add("Edge references unknown target node: " + edge.getTo());
            }
        }

        for (NodeDefinition node : behavior.getNodes()) {
            if (node.getType() == null) {
                errors.add("Node '" + node.getId() + "' has no type");
            }
        }
    }
}

