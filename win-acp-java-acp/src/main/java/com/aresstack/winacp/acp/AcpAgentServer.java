package com.aresstack.winacp.acp;

import com.aresstack.winacp.graph.AgentGraphRunner;
import com.aresstack.winacp.graph.AgentState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ACP protocol server that an ACP client (e.g. JetBrains IDE) connects to.
 * <p>
 * Receives ACP requests over stdio, delegates to the {@link AgentGraphRunner},
 * and sends ACP-conformant responses back.
 */
public class AcpAgentServer {

    private static final Logger log = LoggerFactory.getLogger(AcpAgentServer.class);

    private final AgentGraphRunner graphRunner;

    public AcpAgentServer(AgentGraphRunner graphRunner) {
        this.graphRunner = graphRunner;
    }

    /**
     * Start listening for ACP messages on stdin/stdout.
     */
    public void start() {
        log.info("ACP Agent Server starting (stdio transport)");
        // TODO: integrate ACP Java SDK – read JSON-RPC from stdin, dispatch to graph
    }

    /**
     * Handle a single ACP request (for testing / unit-level use).
     */
    public String handleRequest(String userMessage) {
        log.info("Handling ACP request");
        AgentState state = new AgentState();
        state.setUserInput(userMessage);

        AgentState result = graphRunner.run(state);

        return result.getPendingResponse() != null
                ? result.getPendingResponse()
                : "(no response generated)";
    }

    public void shutdown() {
        log.info("ACP Agent Server shutting down");
    }
}

