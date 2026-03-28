package com.aresstack.winacp.inference.phi3;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Phi3Tokenizer}.
 * Requires the Phi-3 tokenizer.json to be present at the model path.
 */
class Phi3TokenizerTest {

    /** Resolve model path relative to project root (parent of this submodule). */
    private static final Path TOKENIZER_PATH = Path.of(
            System.getProperty("user.dir")).getParent()
            .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128/tokenizer.json");

    private static Phi3Tokenizer tokenizer;

    static boolean tokenizerAvailable() {
        return Files.exists(TOKENIZER_PATH);
    }

    @BeforeAll
    static void loadTokenizer() throws IOException {
        if (tokenizerAvailable()) {
            tokenizer = Phi3Tokenizer.load(TOKENIZER_PATH);
        }
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void tokenizer_loads_with_correct_vocab_size() {
        assertTrue(tokenizer.vocabSize() >= 32011,
                "Vocab should include base vocab + added tokens");
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void encode_simple_text() {
        int[] ids = tokenizer.encode("Hello");
        assertNotNull(ids);
        assertTrue(ids.length > 0, "Encoding should produce at least one token");
        // Round-trip
        String decoded = tokenizer.decode(ids);
        assertEquals("Hello", decoded);
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void encode_decode_roundtrip() {
        String[] texts = {
                "Hello, world!",
                "The quick brown fox jumps over the lazy dog.",
                "1 + 1 = 2",
                "Java 21 FFM is great!",
        };
        for (String text : texts) {
            int[] ids = tokenizer.encode(text);
            String decoded = tokenizer.decode(ids);
            assertEquals(text, decoded, "Round-trip failed for: " + text);
        }
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void debug_special_tokens_map() {
        // Test that indexOf works for special tokens in text
        String text = "<system>\nHello<end>";
        System.out.println("text.indexOf('<system>'): " + text.indexOf("<system>"));
        System.out.println("text.indexOf('<end>'): " + text.indexOf("<end>"));
        // Check vocabSize and specific IDs
        System.out.println("vocabSize: " + tokenizer.vocabSize());
        System.out.println("decodeToken(32006): '" + tokenizer.decodeToken(32006) + "'");
        System.out.println("decodeToken(32007): '" + tokenizer.decodeToken(32007) + "'");
        System.out.println("decodeToken(32010): '" + tokenizer.decodeToken(32010) + "'");
        // Try encoding just the special token string
        int[] sysOnly = tokenizer.encode("<system>");
        System.out.println("encode('<system>'): " + java.util.Arrays.toString(sysOnly));
        for (int id : sysOnly) System.out.println("  " + id + " -> '" + tokenizer.decodeToken(id) + "'");
        // Check if decode(32006) matches "<system>"
        String decoded32006 = tokenizer.decodeToken(32006);
        System.out.println("decoded32006 equals '<system>': " + "<system>".equals(decoded32006));
        System.out.println("decoded32006 length: " + decoded32006.length());
        System.out.println("decoded32006 chars: ");
        for (int i = 0; i < decoded32006.length(); i++) {
            System.out.println("  [" + i + "] = " + (int) decoded32006.charAt(i) + " '" + decoded32006.charAt(i) + "'");
        }
        // Print the specialTokens map
        System.out.println("specialTokens map (" + tokenizer.specialTokensMap().size() + "):");
        for (var entry : tokenizer.specialTokensMap().entrySet()) {
            System.out.println("  '" + entry.getKey() + "' -> " + entry.getValue()
                    + " (len=" + entry.getKey().length() + ", first=" + (int) entry.getKey().charAt(0) + ")");
        }
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void encode_special_tokens() {
        // Phi-3 uses pipe-delimited special tokens: <|system|>, <|end|>, <|user|>, <|assistant|>
        String input = "<|system|>\nYou are helpful.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n";
        int[] ids = tokenizer.encode(input);
        // Debug output
        System.out.println("Input: " + input.replace("\n", "\\n"));
        System.out.println("Token IDs (" + ids.length + "): " + java.util.Arrays.toString(ids));
        for (int id : ids) {
            System.out.println("  " + id + " -> " + tokenizer.decodeToken(id));
        }
        // Should contain the special token IDs
        boolean hasSystem = false, hasEnd = false, hasUser = false, hasAssistant = false;
        for (int id : ids) {
            if (id == Phi3Tokenizer.SYSTEM_ID) hasSystem = true;
            if (id == Phi3Tokenizer.END_ID) hasEnd = true;
            if (id == Phi3Tokenizer.USER_ID) hasUser = true;
            if (id == Phi3Tokenizer.ASSISTANT_ID) hasAssistant = true;
        }
        assertTrue(hasSystem, "Should contain <system> token (32006), got IDs: " + java.util.Arrays.toString(ids));
        assertTrue(hasEnd, "Should contain <end> token");
        assertTrue(hasUser, "Should contain <user> token");
        assertTrue(hasAssistant, "Should contain <assistant> token");
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void format_chat_template() {
        String formatted = tokenizer.formatChat("You are a helpful assistant.", "What is 2+2?");
        assertTrue(formatted.contains("<|system|>"), "Should contain system tag");
        assertTrue(formatted.contains("<|end|>"), "Should contain end tag");
        assertTrue(formatted.contains("<|user|>"), "Should contain user tag");
        assertTrue(formatted.contains("<|assistant|>"), "Should contain assistant tag");
        assertTrue(formatted.contains("What is 2+2?"), "Should contain user message");
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void eos_detection() {
        assertTrue(tokenizer.isEos(Phi3Tokenizer.ENDOFTEXT_ID));
        assertTrue(tokenizer.isEos(Phi3Tokenizer.ASSISTANT_ID));
        assertTrue(tokenizer.isEos(Phi3Tokenizer.END_ID));
        assertFalse(tokenizer.isEos(0));
        assertFalse(tokenizer.isEos(100));
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void encode_empty_string() {
        int[] ids = tokenizer.encode("");
        assertEquals(0, ids.length);
    }

    @Test
    @EnabledIf("tokenizerAvailable")
    void decode_single_token() {
        // Token 0 = <unk>
        String s = tokenizer.decodeToken(0);
        assertEquals("<unk>", s);
    }
}
