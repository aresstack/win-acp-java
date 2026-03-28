package com.aresstack.winacp.inference;

import com.aresstack.winacp.inference.phi3.Phi3Config;
import com.aresstack.winacp.inference.phi3.Phi3GpuKernels;
import com.aresstack.winacp.inference.phi3.Phi3Runtime;
import com.aresstack.winacp.inference.phi3.Phi3Tokenizer;
import com.aresstack.winacp.inference.phi3.Phi3Weights;
import com.aresstack.winacp.windows.WindowsBindings;

import javax.swing.*;
import javax.swing.text.View;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.KeyEvent;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

/**
 * Structured Swing chat UI for testing the Phi-3 model interactively.
 * <p>
 * Uses per-message colored panels with HTML rendering (inspired by
 * <a href="https://github.com/Miguel0888/MainframeMate">MainframeMate</a>'s
 * {@code ChatFormatter} / {@code RightDrawer} chat tab).
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
    private static final Path MODEL_DIR = resolveModelDir();

    private static Path resolveModelDir() {
        Path rel = Path.of("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(rel.resolve("model.onnx"))) return rel;

        Path parent = Path.of(System.getProperty("user.dir")).getParent()
                .resolve("model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128");
        if (Files.exists(parent.resolve("model.onnx"))) return parent;

        return rel;
    }

    // ── State ────────────────────────────────────────────────────────────
    private Phi3Config config;
    private Phi3Tokenizer tokenizer;
    private Phi3Weights weights;
    private Phi3Runtime runtime;
    private WindowsBindings wb;
    private Phi3GpuKernels gpuKernels;
    private volatile boolean modelReady = false;
    private volatile boolean generating = false;

    // ── UI components ────────────────────────────────────────────────────
    private JFrame frame;
    private JPanel messageContainer;
    private JScrollPane chatScroll;
    private JTextField inputField;
    private JButton sendButton;
    private JLabel statusLabel;
    private JSpinner maxTokensSpinner;

    // ── Streaming state ──────────────────────────────────────────────────
    private JTextPane currentBotTextPane;
    private JPanel currentBotPanel;
    private final StringBuilder botBuffer = new StringBuilder();

    // ── Dynamic sizing ───────────────────────────────────────────────────
    private final List<JTextPane> allTextPanes = new ArrayList<>();

    // ── Roles ────────────────────────────────────────────────────────────
    enum Role {
        USER("\uD83D\uDC64 Du", "#e6f0ff", new Color(0x3366AA)),
        BOT("\uD83E\uDD16 Phi-3", "#f0ffe6", new Color(0x66AA66)),
        SYSTEM("\u2699 System", "#f0f0f0", new Color(0xAAAAAA)),
        STATS("\uD83D\uDCCA Stats", "#fff8e6", new Color(0xCCA030));

        final String label;
        final String bgHex;
        final Color accentColor;

        Role(String label, String bgHex, Color accentColor) {
            this.label = label;
            this.bgHex = bgHex;
            this.accentColor = accentColor;
        }
    }

    // ══════════════════════════════════════════════════════════════════════

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new Phi3ChatUI().createAndShow());
    }

    private void createAndShow() {
        // ── Frame ────────────────────────────────────────────────────
        frame = new JFrame("Phi-3 Chat \u2013 win-acp-java");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(780, 650);
        frame.setLocationRelativeTo(null);

        // ── Chat area (scrollable panel of message cards) ────────────
        messageContainer = new JPanel();
        messageContainer.setLayout(new BoxLayout(messageContainer, BoxLayout.Y_AXIS));
        messageContainer.setBackground(UIManager.getColor("Panel.background"));
        messageContainer.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        chatScroll = new JScrollPane(messageContainer);
        chatScroll.setBorder(BorderFactory.createEmptyBorder());
        chatScroll.getVerticalScrollBar().setUnitIncrement(16);
        chatScroll.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

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
        statusLabel = new JLabel("  Modell wird geladen\u2026");
        statusLabel.setFont(new Font("Segoe UI", Font.ITALIC, 12));
        statusLabel.setForeground(Color.GRAY);
        statusLabel.setBorder(BorderFactory.createEmptyBorder(2, 8, 4, 8));

        // ── Layout ───────────────────────────────────────────────────
        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(chatScroll, BorderLayout.CENTER);
        mainPanel.add(inputBar, BorderLayout.SOUTH);
        mainPanel.add(statusLabel, BorderLayout.NORTH);

        frame.setContentPane(mainPanel);

        // ── Actions ──────────────────────────────────────────────────
        Runnable sendAction = this::onSend;
        sendButton.addActionListener(e -> sendAction.run());
        inputField.addActionListener(e -> sendAction.run());

        inputField.getInputMap().put(
                KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, KeyEvent.CTRL_DOWN_MASK), "send");
        inputField.getActionMap().put("send", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent e) { sendAction.run(); }
        });

        // ── Resize listener for dynamic text pane sizing ─────────────
        frame.addComponentListener(new ComponentAdapter() {
            @Override public void componentResized(ComponentEvent e) { resizeAllTextPanes(); }
        });

        frame.setVisible(true);

        // ── Cleanup GPU on close ─────────────────────────────────────
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent e) {
                if (gpuKernels != null) try { gpuKernels.close(); } catch (Exception ignored) {}
                if (wb != null) try { wb.close(); } catch (Exception ignored) {}
            }
        });

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
                    appendSystem("\u274C Modell nicht gefunden in: " + MODEL_DIR.toAbsolutePath());
                    appendSystem("Bitte stelle sicher, dass config.json, tokenizer.json, "
                            + "model.onnx und model.onnx.data vorhanden sind.");
                    statusLabel.setText("  Modell nicht gefunden");
                });
                return;
            }

            config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
            tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));
            weights = Phi3Weights.load(MODEL_DIR, config);

            // ── Try GPU acceleration ─────────────────────────────────
            String mode = "CPU";
            try {
                if (WindowsBindings.isSupported()) {
                    wb = new WindowsBindings();
                    wb.init("directml");
                    if (wb.hasDirectMl()) {
                        int gpuLayers = Integer.getInteger("phi3.gpu.layers",
                                config.numHiddenLayers());
                        boolean gpuLmHead = Boolean.parseBoolean(
                                System.getProperty("phi3.gpu.lmhead", "true"));
                        gpuKernels = Phi3GpuKernels.create(
                                wb, weights, config, gpuLayers, gpuLmHead);
                        mode = "GPU (" + gpuKernels.getGpuLayers() + "/"
                                + config.numHiddenLayers() + " layers)";
                    }
                }
            } catch (Exception gpuEx) {
                System.err.println("GPU init failed, falling back to CPU: " + gpuEx.getMessage());
                gpuEx.printStackTrace();
                if (gpuKernels != null) { try { gpuKernels.close(); } catch (Exception ignored) {} gpuKernels = null; }
                if (wb != null) { try { wb.close(); } catch (Exception ignored) {} wb = null; }
            }

            runtime = new Phi3Runtime(config, weights, tokenizer, gpuKernels);

            long elapsed = System.currentTimeMillis() - t0;
            modelReady = true;

            final String modeLabel = mode;
            SwingUtilities.invokeLater(() -> {
                appendSystem(String.format("\u2705 Modell geladen in %.1f s  (hidden=%d, layers=%d, vocab=%d)",
                        elapsed / 1000.0, config.hiddenSize(),
                        config.numHiddenLayers(), config.vocabSize()));
                appendSystem(modeLabel + "-Modus aktiv. Tippe eine Nachricht und dr\u00FCcke Enter.");
                appendDivider();
                inputField.setEnabled(true);
                sendButton.setEnabled(true);
                inputField.requestFocusInWindow();
                statusLabel.setText("  Bereit \u2013 " + modeLabel);
            });

        } catch (Exception e) {
            SwingUtilities.invokeLater(() -> {
                appendSystem("\u274C Fehler beim Laden: " + e.getMessage());
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
        statusLabel.setText("  Generiere\u2026");

        new Thread(() -> {
            try {
                String prompt = tokenizer.formatChat(null, userText);
                runtime.resetCache();

                // Start bot message (streaming)
                SwingUtilities.invokeLater(this::startBotMessage);

                long t0 = System.currentTimeMillis();
                final int[] tokenCount = {0};

                // Stream tokens into the bot message panel
                String response = runtime.generateStreaming(prompt, maxTokens,
                        (tokenId, textSoFar, delta) -> {
                            tokenCount[0]++;
                            SwingUtilities.invokeLater(() -> appendBotChunk(delta));
                        });

                long elapsed = System.currentTimeMillis() - t0;
                String stats = String.format("%d tokens, %.1f s, %.1f ms/token",
                        tokenCount[0], elapsed / 1000.0,
                        tokenCount[0] > 0 ? (double) elapsed / tokenCount[0] : 0);

                SwingUtilities.invokeLater(() -> {
                    endBotMessage();
                    appendStats(stats);
                    statusLabel.setText("  Bereit \u2013 " + stats);
                    inputField.setEnabled(true);
                    sendButton.setEnabled(true);
                    generating = false;
                    inputField.requestFocusInWindow();
                });

            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    endBotMessage();
                    appendSystem("\u274C Fehler: " + e.getMessage());
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
    // Message panel construction (ChatFormatter-style)
    // ══════════════════════════════════════════════════════════════════════

    private void appendUser(String text) {
        addMessagePanel(Role.USER, escapeHtml(text).replace("\n", "<br/>"));
    }

    private void appendSystem(String text) {
        addMessagePanel(Role.SYSTEM, escapeHtml(text).replace("\n", "<br/>"));
    }

    private void appendStats(String text) {
        addMessagePanel(Role.STATS, escapeHtml(text));
    }

    /** Insert an HTML horizontal rule as a visual divider between message groups. */
    private void appendDivider() {
        JPanel divider = new JPanel();
        divider.setMaximumSize(new Dimension(Integer.MAX_VALUE, 2));
        divider.setPreferredSize(new Dimension(100, 2));
        divider.setBackground(new Color(0xCCCCCC));
        divider.setAlignmentX(Component.LEFT_ALIGNMENT);
        messageContainer.add(Box.createVerticalStrut(4));
        messageContainer.add(divider);
        messageContainer.add(Box.createVerticalStrut(4));
        scrollToBottom();
    }

    // ── Streaming bot message ────────────────────────────────────────────

    private void startBotMessage() {
        botBuffer.setLength(0);
        currentBotTextPane = createTextPane();
        currentBotPanel = createPanelWrapper(Role.BOT);

        JPanel header = createHeader(Role.BOT);
        currentBotPanel.add(header);
        currentBotPanel.add(Box.createVerticalStrut(4));
        currentBotTextPane.setAlignmentX(Component.LEFT_ALIGNMENT);
        currentBotPanel.add(currentBotTextPane);

        messageContainer.add(currentBotPanel);
        messageContainer.add(Box.createVerticalStrut(6));
        scrollToBottom();
    }

    private void appendBotChunk(String chunk) {
        if (chunk == null || chunk.isEmpty() || currentBotTextPane == null) return;
        botBuffer.append(chunk);
        String html = wrapHtml(escapeHtml(botBuffer.toString()).replace("\n", "<br/>"));
        currentBotTextPane.setText(html);
        applyDynamicSizing(currentBotTextPane);
        scrollToBottom();
    }

    private void endBotMessage() {
        if (currentBotTextPane != null) {
            applyDynamicSizing(currentBotTextPane);
        }
        currentBotTextPane = null;
        currentBotPanel = null;
    }

    // ── Generic message panel builder ────────────────────────────────────

    private void addMessagePanel(Role role, String htmlBody) {
        JTextPane pane = createTextPane();
        pane.setText(wrapHtml(htmlBody));
        applyDynamicSizing(pane);

        JPanel wrapper = createPanelWrapper(role);
        JPanel header = createHeader(role);
        wrapper.add(header);
        wrapper.add(Box.createVerticalStrut(4));
        pane.setAlignmentX(Component.LEFT_ALIGNMENT);
        wrapper.add(pane);

        messageContainer.add(wrapper);
        messageContainer.add(Box.createVerticalStrut(6));
        scrollToBottom();
    }

    private JPanel createPanelWrapper(Role role) {
        JPanel wrapper = new JPanel();
        wrapper.setLayout(new BoxLayout(wrapper, BoxLayout.Y_AXIS));
        wrapper.setBackground(Color.decode(role.bgHex));
        wrapper.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 4, 0, 0, role.accentColor),
                BorderFactory.createEmptyBorder(6, 10, 6, 10)
        ));
        wrapper.setAlignmentX(Component.LEFT_ALIGNMENT);
        // Constrain width so panels don't get arbitrarily tall
        wrapper.setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE));
        return wrapper;
    }

    private JPanel createHeader(Role role) {
        JPanel header = new JPanel();
        header.setLayout(new BoxLayout(header, BoxLayout.X_AXIS));
        header.setAlignmentX(Component.LEFT_ALIGNMENT);
        header.setOpaque(false);

        JLabel titleLabel = new JLabel(role.label);
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 12));
        titleLabel.setForeground(role.accentColor.darker());
        header.add(titleLabel);
        header.add(Box.createHorizontalGlue());

        JLabel timeLabel = new JLabel(timestamp());
        timeLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        timeLabel.setForeground(new Color(0x999999));
        header.add(timeLabel);

        return header;
    }

    // ══════════════════════════════════════════════════════════════════════
    // HTML text pane helpers
    // ══════════════════════════════════════════════════════════════════════

    private JTextPane createTextPane() {
        JTextPane pane = new JTextPane();
        pane.setContentType("text/html");
        pane.setEditable(false);
        pane.setOpaque(false);
        pane.setBorder(null);
        pane.setAlignmentX(Component.LEFT_ALIGNMENT);
        allTextPanes.add(pane);
        return pane;
    }

    private String wrapHtml(String bodyHtml) {
        String css =
                "body { font-family: 'Segoe UI', sans-serif; font-size: 14px; " +
                "       margin: 0; padding: 0; line-height: 1.5; } " +
                "p { margin: 2px 0; } " +
                "code { background-color: #f5f5f5; padding: 1px 4px; border-radius: 3px; " +
                "       font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px; } " +
                "pre { background-color: #f5f5f5; padding: 6px; border-radius: 4px; " +
                "      border: 1px solid #ddd; font-family: monospace; " +
                "      white-space: pre-wrap; word-wrap: break-word; } " +
                "hr { border: none; border-top: 1px solid #ccc; margin: 6px 0; } " +
                "strong { font-weight: bold; } " +
                "em { font-style: italic; }";

        return "<html><head><style>" + css + "</style></head><body>" + bodyHtml + "</body></html>";
    }

    private void applyDynamicSizing(JTextPane pane) {
        int width = (messageContainer.getParent() != null)
                ? messageContainer.getParent().getWidth() - 60
                : 600;
        if (width < 200) width = 600;

        pane.setSize(new Dimension(width, Integer.MAX_VALUE));
        View rootView = pane.getUI().getRootView(pane);
        rootView.setSize(width, Integer.MAX_VALUE);
        int height = (int) rootView.getPreferredSpan(View.Y_AXIS) + 4;

        pane.setPreferredSize(new Dimension(width, height));
        pane.setMaximumSize(new Dimension(Integer.MAX_VALUE, height));
        pane.revalidate();
        pane.repaint();
    }

    private void resizeAllTextPanes() {
        SwingUtilities.invokeLater(() -> {
            int width = (messageContainer.getParent() != null)
                    ? messageContainer.getParent().getWidth() - 60
                    : 600;
            if (width < 200) width = 600;

            for (JTextPane pane : allTextPanes) {
                pane.setSize(new Dimension(width, Integer.MAX_VALUE));
                View rootView = pane.getUI().getRootView(pane);
                rootView.setSize(width, Integer.MAX_VALUE);
                int height = (int) rootView.getPreferredSpan(View.Y_AXIS) + 4;
                pane.setPreferredSize(new Dimension(width, height));
                pane.setMaximumSize(new Dimension(Integer.MAX_VALUE, height));
                pane.revalidate();
            }
            messageContainer.revalidate();
            messageContainer.repaint();
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // Utility
    // ══════════════════════════════════════════════════════════════════════

    private String escapeHtml(String input) {
        if (input == null) return "";
        return input
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");
    }

    private void scrollToBottom() {
        SwingUtilities.invokeLater(() -> {
            JScrollBar vBar = chatScroll.getVerticalScrollBar();
            vBar.setValue(vBar.getMaximum());
        });
    }

    private String timestamp() {
        return LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));
    }
}
