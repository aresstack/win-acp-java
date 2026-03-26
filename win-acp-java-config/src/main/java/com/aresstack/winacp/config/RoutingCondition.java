package com.aresstack.winacp.config;

/**
 * Predefined routing conditions for conditional edges.
 * <p>
 * Only these conditions are allowed in external configuration.
 */
public enum RoutingCondition {

    /** A tool is required to continue. */
    TOOL_REQUIRED,

    /** A tool is not required. */
    TOOL_NOT_REQUIRED,

    /** The selected tool is allowed by policy. */
    TOOL_ALLOWED,

    /** The selected tool is blocked by policy. */
    TOOL_NOT_ALLOWED,

    /** Human approval is required before continuing. */
    APPROVAL_REQUIRED,

    /** No approval is required. */
    APPROVAL_NOT_REQUIRED,

    /** The result is sufficient to produce a final answer. */
    RESULT_SUFFICIENT,

    /** The result is not sufficient; further action needed. */
    RESULT_INSUFFICIENT,

    /** A clarification from the user is needed. */
    CLARIFICATION_REQUIRED,

    /** No clarification is needed. */
    CLARIFICATION_NOT_REQUIRED,

    /** The maximum number of iterations/retries has been reached. */
    MAX_RETRIES_REACHED,

    /** The agent has reached a terminal state. */
    COMPLETED,

    /** Unconditional transition (always taken). */
    ALWAYS
}

