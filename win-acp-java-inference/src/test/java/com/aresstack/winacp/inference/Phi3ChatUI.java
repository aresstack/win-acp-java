package com.aresstack.winacp.inference;

import com.aresstack.winacp.inference.phi3.Phi3Config;
import com.aresstack.winacp.inference.phi3.Phi3Runtime;
import com.aresstack.winacp.inference.phi3.Phi3Tokenizer;
import com.aresstack.winacp.inference.phi3.Phi3Weights;

import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

/**
 * Simple Swing chat UI for testing the Phi-3 model interactively.
 * <p>
 * Start this class directly from IntelliJ (right-click → Run).
 * <p>
 * JVM args required:
 * <pre>
 *   --enable-native-access=ALL-UNNAMED -Xmx4g
 * </pre>
 * <p>
 * The UI loads the model on startup (takes ~5–10 s), then lets you
 * type messages and see the model's responses in real time.
 * Generation runs on a background thread so the UI stays responsive.
 */
public class Phi3ChatUI {

    // ── Model path ───────────────────────────────────────────────────────
    // Adjust if your model is in a different location.
    private static final Path MODEL_DIR = resolveModelDir();

    private static Path resolveModelDir() {
        // Try relative to working directory (project root)
        Path rel = Path.of("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(rel.resolve("model.onnx"))) return rel;

        // Try relative to submodule (when CWD = win-acp-java-inference)
        Path parent = Path.of(System.getProperty("user.dir")).getParent()
                .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(parent.resolve("model.onnx"))) return parent;

        // Fallback
        return rel;
    }

    // ── State ────────────────────────────────────────────────────────────
    private Phi3Config config;
    private Phi3Tokenizer tokenizer;
    private Phi3Weights weights;
    private Phi3Runtime runtime;
    private volatile boolean modelReady = false;
    private volatile boolean generating = false;

    // ── UI components ────────────────────────────────────────────────────
    private JFrame frame;
    private JTextPane chatPane;
    private StyledDocument chatDoc;
    private JTextField inputField;
    private JButton sendButton;
    private JLabel statusLabel;
    private JSpinner maxTokensSpinner;

    // ── Styles ───────────────────────────────────────────────────────────
    private Style styleUser;
    private Style styleBot;
    private Style styleSystem;
    private Style styleTime;

    // ══════════════════════════════════════════════════════════════════════

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new Phi3ChatUI().createAndShow());
    }

    private void createAndShow() {
        // ── Frame ────────────────────────────────────────────────────
        frame = new JFrame("Phi-3 Chat – win-acp-java");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(720, 600);
        frame.setLocationRelativeTo(null);

        // ── Chat area ────────────────────────────────────────────────
        chatPane = new JTextPane();
        chatPane.setEditable(false);
        chatPane.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        chatDoc = chatPane.getStyledDocument();

        initStyles();

        JScrollPane scroll = new JScrollPane(chatPane);
        scroll.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        // ── Input bar ────────────────────────────────────────────────
        inputField = new JTextField();
        inputField.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        inputField.setEnabled(false);

        sendButton = new JButton("Senden");
        sendButton.setEnabled(false);
        sendButton.setPreferredSize(new Dimension(100, 32));

        maxTokensSpinner = new JSpinner(new SpinnerNumberModel(64, 1, 2048, 16));
        maxTokensSpinner.setToolTipText("Max Tokens");
        maxTokensSpinner.setPreferredSize(new Dimension(70, 32));

        JPanel inputBar = new JPanel(new BorderLayout(4, 0));
        inputBar.setBorder(BorderFactory.createEmptyBorder(4, 8, 8, 8));

        JPanel rightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 4, 0));
        rightPanel.add(new JLabel("Max:"));
        rightPanel.add(maxTokensSpinner);
        rightPanel.add(sendButton);

        inputBar.add(inputField, BorderLayout.CENTER);
        inputBar.add(rightPanel, BorderLayout.EAST);

        // ── Status bar ───────────────────────────────────────────────
        statusLabel = new JLabel("  Modell wird geladen...");
        statusLabel.setFont(new Font("Segoe UI", Font.ITALIC, 12));
        statusLabel.setForeground(Color.GRAY);
        statusLabel.setBorder(BorderFactory.createEmptyBorder(2, 8, 4, 8));

        // ── Layout ───────────────────────────────────────────────────
        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(scroll, BorderLayout.CENTER);
        mainPanel.add(inputBar, BorderLayout.SOUTH);
        mainPanel.add(statusLabel, BorderLayout.NORTH);

        frame.setContentPane(mainPanel);

        // ── Actions ──────────────────────────────────────────────────
        Runnable sendAction = this::onSend;
        sendButton.addActionListener(e -> sendAction.run());
        inputField.addActionListener(e -> sendAction.run());

        // Ctrl+Enter also sends
        inputField.getInputMap().put(
                KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, KeyEvent.CTRL_DOWN_MASK), "send");
        inputField.getActionMap().put("send", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { sendAction.run(); }
        });

        frame.setVisible(true);

        // ── Load model in background ─────────────────────────────────
        appendSystem("Lade Phi-3 Modell von:\n" + MODEL_DIR.toAbsolutePath());
        new Thread(this::loadModel, "model-loader").start();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Model loading
    // ══════════════════════════════════════════════════════════════════════

    private void loadModel() {
        long t0 = System.currentTimeMillis();
        try {
            if (!Phi3InferenceEngine.isValidModelDir(MODEL_DIR)) {
                SwingUtilities.invokeLater(() -> {
                    appendSystem("❌ Modell nicht gefunden in: " + MODEL_DIR.toAbsolutePath());
                    appendSystem("Bitte stelle sicher, dass config.json, tokenizer.json, "
                            + "model.onnx und model.onnx.data vorhanden sind.");
                    statusLabel.setText("  Modell nicht gefunden");
                });
                return;
            }

            config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
            tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));
            weights = Phi3Weights.load(MODEL_DIR, config);
            runtime = new Phi3Runtime(config, weights, tokenizer);

            long elapsed = System.currentTimeMillis() - t0;
            modelReady = true;

            SwingUtilities.invokeLater(() -> {
                appendSystem(String.format("✅ Modell geladen in %.1f s  (hidden=%d, layers=%d, vocab=%d)",
                        elapsed / 1000.0, config.hiddenSize(),
                        config.numHiddenLayers(), config.vocabSize()));
                appendSystem("CPU-Modus aktiv. Tippe eine Nachricht und drücke Enter.");
                appendSystem("─".repeat(60));
                inputField.setEnabled(true);
                sendButton.setEnabled(true);
                inputField.requestFocusInWindow();
                statusLabel.setText("  Bereit – CPU-Modus");
            });

        } catch (Exception e) {
            SwingUtilities.invokeLater(() -> {
                appendSystem("❌ Fehler beim Laden: " + e.getMessage());
                statusLabel.setText("  Fehler beim Laden");
            });
            e.printStackTrace();
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Send / generate
    // ══════════════════════════════════════════════════════════════════════

    private void onSend() {
        if (!modelReady || generating) return;

        String userText = inputField.getText().trim();
        if (userText.isEmpty()) return;

        inputField.setText("");
        inputField.setEnabled(false);
        sendButton.setEnabled(false);
        generating = true;

        appendUser(userText);

        int maxTokens = (int) maxTokensSpinner.getValue();
        statusLabel.setText("  Generiere...");

        new Thread(() -> {
            try {
                String prompt = tokenizer.formatChat(null, userText);
                runtime.resetCache();

                long t0 = System.currentTimeMillis();
                String response = runtime.generate(prompt, maxTokens);
                long elapsed = System.currentTimeMillis() - t0;

                int tokens = tokenizer.encode(response).length;
                String stats = String.format("[%d tokens, %.1f s, %.1f ms/token]",
                        tokens, elapsed / 1000.0,
                        tokens > 0 ? (double) elapsed / tokens : 0);

                SwingUtilities.invokeLater(() -> {
                    appendBot(response.strip());
                    appendSystem(stats);
                    statusLabel.setText("  Bereit – " + stats);
                    inputField.setEnabled(true);
                    sendButton.setEnabled(true);
                    generating = false;
                    inputField.requestFocusInWindow();
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    appendSystem("❌ Fehler: " + e.getMessage());
                    statusLabel.setText("  Fehler bei Generierung");
                    inputField.setEnabled(true);
                    sendButton.setEnabled(true);
                    generating = false;
                });
                e.printStackTrace();
            }
        }, "phi3-generate").start();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Chat append helpers
    // ══════════════════════════════════════════════════════════════════════

    private void appendUser(String text) {
        append(timestamp() + " Du:\n", styleTime);
        append(text + "\n\n", styleUser);
    }

    private void appendBot(String text) {
        append(timestamp() + " Phi-3:\n", styleTime);
        append(text + "\n\n", styleBot);
    }

    private void appendSystem(String text) {
        append(text + "\n", styleSystem);
    }

    private void append(String text, Style style) {
        try {
            chatDoc.insertString(chatDoc.getLength(), text, style);
            chatPane.setCaretPosition(chatDoc.getLength());
        } catch (BadLocationException ignored) {}
    }

    private String timestamp() {
        return LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Styles
    // ══════════════════════════════════════════════════════════════════════

    private void initStyles() {
        Style def = StyleContext.getDefaultStyleContext().getStyle(StyleContext.DEFAULT_STYLE);

        styleUser = chatDoc.addStyle("user", def);
        StyleConstants.setForeground(styleUser, new Color(0, 90, 180));
        StyleConstants.setBold(styleUser, false);

        styleBot = chatDoc.addStyle("bot", def);
        StyleConstants.setForeground(styleBot, new Color(30, 30, 30));
        StyleConstants.setBold(styleBot, false);

        styleSystem = chatDoc.addStyle("system", def);
        StyleConstants.setForeground(styleSystem, new Color(120, 120, 120));
        StyleConstants.setItalic(styleSystem, true);
        StyleConstants.setFontSize(styleSystem, 12);

        styleTime = chatDoc.addStyle("time", def);
        StyleConstants.setForeground(styleTime, new Color(100, 100, 100));
        StyleConstants.setBold(styleTime, true);
        StyleConstants.setFontSize(styleTime, 12);
    }
}

