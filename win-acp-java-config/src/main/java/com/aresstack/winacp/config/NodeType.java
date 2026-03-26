package com.aresstack.winacp.config;

/**
 * Predefined node categories for the agent behavior graph.
 * <p>
 * Only these types are allowed in external configuration.
 * Adding a new type requires a code change – this is intentional
 * to prevent arbitrary behavior injection.
 */
public enum NodeType {

    /** Parse and classify the incoming user request. */
    ANALYZE_INPUT,

    /** Determine the goal or intent of the current turn. */
    DETERMINE_GOAL,

    /** Load context relevant to the current request. */
    LOAD_CONTEXT,

    /** Select the best matching tool from available candidates. */
    SELECT_TOOL,

    /** Execute a previously selected MCP tool. */
    EXECUTE_TOOL,

    /** Evaluate the result returned by a tool. */
    EVALUATE_TOOL_RESULT,

    /** Generate a clarification question for the user. */
    GENERATE_CLARIFICATION,

    /** Formulate the final response for the user. */
    FORMULATE_RESPONSE,

    /** Produce the final completion signal. */
    FINALIZE,

    /** Handle an error that occurred during processing. */
    HANDLE_ERROR,

    /** Request human approval before continuing. */
    REQUEST_APPROVAL
}

