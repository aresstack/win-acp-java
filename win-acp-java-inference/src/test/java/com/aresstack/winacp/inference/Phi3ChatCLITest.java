package com.aresstack.winacp.inference;

import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.Scanner;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Phi3ChatCLI's command handling and JSON parsing.
 * These tests do NOT load the model — they only exercise the CLI protocol layer.
 */
class Phi3ChatCLITest {

    /**
     * Helper: feeds input lines to the CLI, captures JSON output.
     * Model loading is skipped (test mode).
     */
    private String runCli(String... inputLines) {
        String input = String.join("\n", inputLines) + "\n";
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintWriter pw = new PrintWriter(baos, true);
        Phi3ChatCLI cli = new Phi3ChatCLI(pw, true);  // skip model loading
        cli.run(new Scanner(new StringReader(input)));
        return baos.toString();
    }

    // ── Slash commands ───────────────────────────────────────────────────

    @Test
    void exitCommandTerminatesCli() {
        String output = runCli("/exit");
        assertTrue(output.contains("\"type\":\"system\""), "Should emit system messages");
        assertTrue(output.contains("\"event\":\"exit\""), "Should emit exit event");
        assertTrue(output.contains("Bye"), "Should say bye");
    }

    @Test
    void quitCommandTerminatesCli() {
        String output = runCli("/quit");
        assertTrue(output.contains("\"event\":\"exit\""));
    }

    @Test
    void helpCommandShowsHelp() {
        String output = runCli("/help", "/exit");
        assertTrue(output.contains("\"type\":\"command\""), "Should emit command response");
        assertTrue(output.contains("/help"), "Help should mention /help");
        assertTrue(output.contains("/exit"), "Help should mention /exit");
        assertTrue(output.contains("/maxTokens"), "Help should mention /maxTokens");
    }

    @Test
    void maxTokensCommandValid() {
        String output = runCli("/maxTokens 128", "/exit");
        assertTrue(output.contains("maxTokens set to 128"));
    }

    @Test
    void maxTokensCommandUnlimited() {
        String output = runCli("/maxTokens -1", "/exit");
        assertTrue(output.contains("unlimited"), "Should accept -1 as unlimited");
    }

    @Test
    void maxTokensCommandInvalid() {
        String output = runCli("/maxTokens abc", "/exit");
        assertTrue(output.contains("\"type\":\"error\""), "Should emit error for invalid value");
    }

    @Test
    void maxTokensCommandOutOfRange() {
        String output = runCli("/maxTokens 9999", "/exit");
        assertTrue(output.contains("\"type\":\"error\""), "Should reject out-of-range value");
    }

    @Test
    void maxTokensCommandZeroRejected() {
        String output = runCli("/maxTokens 0", "/exit");
        assertTrue(output.contains("\"type\":\"error\""), "Should reject 0");
    }

    @Test
    void systemPromptSetAndClear() {
        String output = runCli("/systemPrompt You are a pirate", "/systemPrompt", "/exit");
        assertTrue(output.contains("System prompt set."));
        assertTrue(output.contains("System prompt cleared."));
    }

    @Test
    void clearCommandClearsHistory() {
        String output = runCli("/clear", "/exit");
        assertTrue(output.contains("Conversation history cleared."));
    }

    @Test
    void historyCommandReturnsEmptyList() {
        String output = runCli("/history", "/exit");
        assertTrue(output.contains("\"messages\":[]"), "History should be empty initially");
    }

    @Test
    void unknownCommandReturnsError() {
        String output = runCli("/foobar", "/exit");
        assertTrue(output.contains("\"type\":\"error\""));
        assertTrue(output.contains("Unknown command"));
    }

    @Test
    void emptyLinesAreSkipped() {
        String output = runCli("", "", "/exit");
        assertTrue(output.contains("\"event\":\"exit\""));
    }

    // ── JSON input without model ─────────────────────────────────────────

    @Test
    void jsonInputWithoutModelReturnsError() {
        // Model won't be loaded (no model files), so JSON messages should fail gracefully
        String output = runCli("{\"prompt\":\"hello\"}", "/exit");
        assertTrue(output.contains("\"type\":\"error\""), "Should emit error when model not loaded");
    }

    @Test
    void invalidJsonReturnsError() {
        String output = runCli("not json at all", "/exit");
        assertTrue(output.contains("\"type\":\"error\""), "Should emit error for invalid JSON");
    }

    @Test
    void jsonMissingPromptReturnsError() {
        // Even if model were loaded, missing prompt should fail
        String output = runCli("{\"maxTokens\":32}", "/exit");
        assertTrue(output.contains("\"type\":\"error\""));
    }

    // ── JSON output format ───────────────────────────────────────────────

    @Test
    void allOutputLinesAreValidJson() {
        String output = runCli("/help", "/status", "/history", "/exit");
        String[] lines = output.strip().split("\\r?\\n");
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty()) continue;
            assertTrue(line.startsWith("{") && line.endsWith("}"),
                    "Every output line must be a JSON object, got: " + line);
        }
    }

    @Test
    void statusCommandIncludesModelReadyFalse() {
        // Without model files, modelReady should be false
        String output = runCli("/status", "/exit");
        assertTrue(output.contains("\"modelReady\":false"));
    }
}
