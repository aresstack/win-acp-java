package com.aresstack.winacp.inference.app;

import com.aresstack.winacp.inference.Phi3InferenceEngine;
import com.aresstack.winacp.inference.phi3.Phi3Config;
import com.aresstack.winacp.inference.phi3.Phi3GpuKernels;
import com.aresstack.winacp.inference.phi3.Phi3Runtime;
import com.aresstack.winacp.inference.phi3.Phi3Tokenizer;
import com.aresstack.winacp.inference.phi3.Phi3Tokenizer.ChatMessage;
import com.aresstack.winacp.inference.phi3.Phi3Weights;
import com.aresstack.winacp.windows.WindowsBindings;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.KeyStroke;
import javax.swing.ScrollPaneConstants;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import javax.swing.text.View;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Toolkit;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.KeyEvent;
import java.nio.file.Path;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class Phi3ChatUI {

    private static final Path MODEL_DIR = Phi3ModelDirectoryResolver.resolve();

    private static final String SYSTEM_PROMPT =
            "You are a helpful AI assistant. Answer concisely and accurately. " +
                    "Respond in the same language the user writes in.";

    private Phi3Config config;
    private Phi3Tokenizer tokenizer;
    private Phi3Weights weights;
    private Phi3Runtime runtime;
    private WindowsBindings windowsBindings;
    private Phi3GpuKernels gpuKernels;
    private volatile boolean modelReady = false;
    private volatile boolean generating = false;

    private final List<String> chatHistory = new ArrayList<String>();
    private final List<ChatMessage> conversationMessages = new ArrayList<ChatMessage>();
    private final List<JTextPane> allTextPanes = new ArrayList<JTextPane>();
    private final StringBuilder botBuffer = new StringBuilder();

    private JFrame frame;
    private JPanel messageContainer;
    private JScrollPane chatScroll;
    private JTextField inputField;
    private JButton sendButton;
    private JButton copyButton;
    private JLabel statusLabel;
    private JComboBox<String> maxTokensCombo;
    private JTextPane currentBotTextPane;

    private enum Role {
        USER("\uD83D\uDC64 Du", "#e6f0ff", new Color(0x3366AA)),
        BOT("\uD83E\uDD16 Phi-3", "#f0ffe6", new Color(0x66AA66)),
        SYSTEM("\u2699 System", "#f0f0f0", new Color(0xAAAAAA)),
        STATS("\uD83D\uDCCA Stats", "#fff8e6", new Color(0xCCA030));

        private final String label;
        private final String backgroundHex;
        private final Color accentColor;

        Role(String label, String backgroundHex, Color accentColor) {
            this.label = label;
            this.backgroundHex = backgroundHex;
            this.accentColor = accentColor;
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new Phi3ChatUI().createAndShow();
            }
        });
    }

    private void createAndShow() {
        frame = new JFrame("Phi-3 Chat – win-acp-java");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(780, 650);
        frame.setLocationRelativeTo(null);

        messageContainer = new JPanel();
        messageContainer.setLayout(new BoxLayout(messageContainer, BoxLayout.Y_AXIS));
        messageContainer.setBackground(UIManager.getColor("Panel.background"));
        messageContainer.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        chatScroll = new JScrollPane(messageContainer);
        chatScroll.setBorder(BorderFactory.createEmptyBorder());
        chatScroll.getVerticalScrollBar().setUnitIncrement(16);
        chatScroll.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);

        inputField = new JTextField();
        inputField.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        inputField.setEnabled(false);

        sendButton = new JButton("Senden");
        sendButton.setEnabled(false);
        sendButton.setPreferredSize(new Dimension(100, 32));

        maxTokensCombo = new JComboBox<String>(new String[]{
                "\u221E bis EOS", "128", "256", "512", "1024", "2048"
        });
        maxTokensCombo.setSelectedIndex(0);
        maxTokensCombo.setToolTipText("Max Tokens (\u221E = Modell entscheidet, wann Schluss ist)");
        maxTokensCombo.setPreferredSize(new Dimension(110, 32));

        JPanel inputBar = new JPanel(new BorderLayout(4, 0));
        inputBar.setBorder(BorderFactory.createEmptyBorder(4, 8, 8, 8));

        JPanel rightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 4, 0));
        rightPanel.add(new JLabel("Max:"));
        rightPanel.add(maxTokensCombo);
        rightPanel.add(sendButton);

        copyButton = new JButton("\uD83D\uDCCB Copy");
        copyButton.setPreferredSize(new Dimension(80, 32));
        copyButton.setToolTipText("Gesamten Chat in die Zwischenablage kopieren");
        copyButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                copyChat();
            }
        });
        rightPanel.add(copyButton);

        inputBar.add(inputField, BorderLayout.CENTER);
        inputBar.add(rightPanel, BorderLayout.EAST);

        statusLabel = new JLabel("  Modell wird geladen…");
        statusLabel.setFont(new Font("Segoe UI", Font.ITALIC, 12));
        statusLabel.setForeground(Color.GRAY);
        statusLabel.setBorder(BorderFactory.createEmptyBorder(2, 8, 4, 8));

        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(chatScroll, BorderLayout.CENTER);
        mainPanel.add(inputBar, BorderLayout.SOUTH);
        mainPanel.add(statusLabel, BorderLayout.NORTH);
        frame.setContentPane(mainPanel);

        final Runnable sendAction = new Runnable() {
            @Override
            public void run() {
                onSend();
            }
        };

        sendButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                sendAction.run();
            }
        });

        inputField.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                sendAction.run();
            }
        });

        inputField.getInputMap().put(
                KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, KeyEvent.CTRL_DOWN_MASK),
                "send"
        );
        inputField.getActionMap().put("send", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                sendAction.run();
            }
        });

        frame.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent event) {
                resizeAllTextPanes();
            }
        });

        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent event) {
                closeGpuResources();
            }
        });

        frame.setVisible(true);

        appendSystem("Lade Phi-3 Modell von:\n" + MODEL_DIR.toAbsolutePath());
        new Thread(new Runnable() {
            @Override
            public void run() {
                loadModel();
            }
        }, "model-loader").start();
    }

    private void loadModel() {
        long startTime = System.currentTimeMillis();
        try {
            if (!Phi3InferenceEngine.isValidModelDir(MODEL_DIR)) {
                SwingUtilities.invokeLater(new Runnable() {
                    @Override
                    public void run() {
                        appendSystem("❌ Modell nicht gefunden in: " + MODEL_DIR.toAbsolutePath());
                        appendSystem("Bitte stelle sicher, dass config.json, tokenizer.json, "
                                + "model.onnx und model.onnx.data vorhanden sind.");
                        statusLabel.setText("  Modell nicht gefunden");
                    }
                });
                return;
            }

            config = Phi3Config.load(MODEL_DIR.resolve("config.json"));
            tokenizer = Phi3Tokenizer.load(MODEL_DIR.resolve("tokenizer.json"));
            weights = Phi3Weights.load(MODEL_DIR, config);

            final String activeMode = initialiseInferenceMode();
            runtime = new Phi3Runtime(config, weights, tokenizer, gpuKernels);
            modelReady = true;

            final long elapsedMillis = System.currentTimeMillis() - startTime;
            SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    appendSystem(String.format(
                            "✅ Modell geladen in %.1f s  (hidden=%d, layers=%d, vocab=%d)",
                            elapsedMillis / 1000.0,
                            config.hiddenSize(),
                            config.numHiddenLayers(),
                            config.vocabSize()));
                    appendSystem(activeMode + "-Modus aktiv. Tippe eine Nachricht und drücke Enter.");
                    appendDivider();

                    inputField.setEnabled(true);
                    sendButton.setEnabled(true);
                    inputField.requestFocusInWindow();
                    statusLabel.setText("  Bereit – " + activeMode);
                }
            });
        } catch (final Exception exception) {
            SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    appendSystem("❌ Fehler beim Laden: " + exception.getMessage());
                    statusLabel.setText("  Fehler beim Laden");
                }
            });
            exception.printStackTrace();
        }
    }

    private String initialiseInferenceMode() {
        String backend = resolveBackend();
        if ("cpu".equals(backend)) {
            return "CPU";
        }

        try {
            if (WindowsBindings.isSupported()) {
                windowsBindings = new WindowsBindings();
                windowsBindings.init(backend);

                if (windowsBindings.hasDirectMl()) {
                    int gpuLayers = Integer.getInteger("phi3.gpu.layers", config.numHiddenLayers());
                    boolean gpuLmHead = Boolean.parseBoolean(
                            System.getProperty("phi3.gpu.lmhead", "true"));

                    gpuKernels = Phi3GpuKernels.create(
                            windowsBindings, weights, config, gpuLayers, gpuLmHead);

                    return "GPU (" + gpuKernels.getGpuLayers() + "/" + config.numHiddenLayers() + " layers)";
                }
            }
        } catch (Exception exception) {
            System.err.println("GPU init failed, falling back to CPU: " + exception.getMessage());
            exception.printStackTrace();
            closeGpuResources();
        }

        return "CPU";
    }

    private String resolveBackend() {
        String configuredValue = System.getProperty("phi3.backend", "auto");
        if (configuredValue == null) {
            return "auto";
        }

        String normalizedValue = configuredValue.trim().toLowerCase();
        if ("cpu".equals(normalizedValue)
                || "auto".equals(normalizedValue)
                || "directml".equals(normalizedValue)) {
            return normalizedValue;
        }

        return "auto";
    }

    private void onSend() {
        if (!modelReady || generating) {
            return;
        }

        final String userText = inputField.getText().trim();
        if (userText.isEmpty()) {
            return;
        }

        inputField.setText("");
        inputField.setEnabled(false);
        sendButton.setEnabled(false);
        generating = true;

        appendUser(userText);

        final int maxTokens = getSelectedMaxTokens();
        statusLabel.setText("  Generiere…");

        new Thread(new Runnable() {
            @Override
            public void run() {
                generateResponse(userText, maxTokens);
            }
        }, "phi3-generate").start();
    }

    private void generateResponse(final String userText, final int maxTokens) {
        try {
            List<ChatMessage> pendingHistory = new ArrayList<ChatMessage>(conversationMessages);
            pendingHistory.add(ChatMessage.user(userText));
            String prompt = tokenizer.formatMultiTurnChat(SYSTEM_PROMPT, pendingHistory);

            SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    startBotMessage();
                }
            });

            long startTime = System.currentTimeMillis();
            final int[] tokenCount = {0};

            String response = runtime.generateStreaming(prompt, maxTokens,
                    (tokenId, textSoFar, delta) -> {
                        tokenCount[0]++;
                        SwingUtilities.invokeLater(new Runnable() {
                            @Override
                            public void run() {
                                appendBotChunk(delta);
                            }
                        });
                    });

            conversationMessages.add(ChatMessage.user(userText));
            conversationMessages.add(ChatMessage.assistant(response));

            long elapsedMillis = System.currentTimeMillis() - startTime;
            final String stats = String.format(
                    "%d tokens, %.1f s, %.1f ms/token",
                    tokenCount[0],
                    elapsedMillis / 1000.0,
                    tokenCount[0] > 0 ? (double) elapsedMillis / tokenCount[0] : 0.0
            );

            SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    endBotMessage();
                    appendStats(stats);
                    statusLabel.setText("  Bereit – " + stats);
                    inputField.setEnabled(true);
                    sendButton.setEnabled(true);
                    generating = false;
                    inputField.requestFocusInWindow();
                }
            });

        } catch (final Exception exception) {
            SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    endBotMessage();
                    appendSystem("❌ Fehler: " + exception.getMessage());
                    statusLabel.setText("  Fehler bei Generierung");
                    inputField.setEnabled(true);
                    sendButton.setEnabled(true);
                    generating = false;
                }
            });
            exception.printStackTrace();
        }
    }

    private void appendUser(String text) {
        chatHistory.add("[" + timestamp() + "] Du: " + text);
        addMessagePanel(Role.USER, escapeHtml(text).replace("\n", "<br/>"));
    }

    private void appendSystem(String text) {
        addMessagePanel(Role.SYSTEM, escapeHtml(text).replace("\n", "<br/>"));
    }

    private void appendStats(String text) {
        chatHistory.add(text);
        String profile = runtime != null ? runtime.getLastProfile() : null;
        String combined = profile != null ? text + "\n" + profile : text;
        addMessagePanel(Role.STATS, "<pre>" + escapeHtml(combined) + "</pre>");
    }

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

    private void startBotMessage() {
        botBuffer.setLength(0);
        currentBotTextPane = createTextPane();

        JPanel wrapper = createPanelWrapper(Role.BOT);
        wrapper.add(createHeader(Role.BOT));
        wrapper.add(Box.createVerticalStrut(4));
        currentBotTextPane.setAlignmentX(Component.LEFT_ALIGNMENT);
        wrapper.add(currentBotTextPane);

        messageContainer.add(wrapper);
        messageContainer.add(Box.createVerticalStrut(6));
        scrollToBottom();
    }

    private void appendBotChunk(String chunk) {
        if (chunk == null || chunk.isEmpty() || currentBotTextPane == null) {
            return;
        }

        botBuffer.append(chunk);
        currentBotTextPane.setText(
                wrapHtml(escapeHtml(botBuffer.toString()).replace("\n", "<br/>"))
        );
        applyDynamicSizing(currentBotTextPane);
        scrollToBottom();
    }

    private void endBotMessage() {
        if (currentBotTextPane != null) {
            applyDynamicSizing(currentBotTextPane);
            chatHistory.add("[" + timestamp() + "] Phi-3: " + botBuffer.toString());
        }
        currentBotTextPane = null;
    }

    private void addMessagePanel(Role role, String htmlBody) {
        JTextPane pane = createTextPane();
        pane.setText(wrapHtml(htmlBody));
        applyDynamicSizing(pane);

        JPanel wrapper = createPanelWrapper(role);
        wrapper.add(createHeader(role));
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
        wrapper.setBackground(Color.decode(role.backgroundHex));
        wrapper.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 4, 0, 0, role.accentColor),
                BorderFactory.createEmptyBorder(6, 10, 6, 10)
        ));
        wrapper.setAlignmentX(Component.LEFT_ALIGNMENT);
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
                "body { font-family: 'Segoe UI', sans-serif; font-size: 14px; margin: 0; padding: 0; line-height: 1.5; } " +
                        "p { margin: 2px 0; } " +
                        "code { background-color: #f5f5f5; padding: 1px 4px; border-radius: 3px; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 13px; } " +
                        "pre { background-color: #f5f5f5; padding: 6px; border-radius: 4px; border: 1px solid #ddd; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; } " +
                        "hr { border: none; border-top: 1px solid #ccc; margin: 6px 0; } " +
                        "strong { font-weight: bold; } " +
                        "em { font-style: italic; }";

        return "<html><head><style>" + css + "</style></head><body>" + bodyHtml + "</body></html>";
    }

    private void applyDynamicSizing(JTextPane pane) {
        int width = messageContainer.getParent() != null
                ? messageContainer.getParent().getWidth() - 60
                : 600;

        if (width < 200) {
            width = 600;
        }

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
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                int width = messageContainer.getParent() != null
                        ? messageContainer.getParent().getWidth() - 60
                        : 600;

                if (width < 200) {
                    width = 600;
                }

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
            }
        });
    }

    private int getSelectedMaxTokens() {
        String selected = (String) maxTokensCombo.getSelectedItem();
        if (selected == null || selected.startsWith("\u221E")) {
            return config != null ? config.maxPositionEmbeddings() : 4096;
        }

        try {
            return Integer.parseInt(selected.trim());
        } catch (NumberFormatException exception) {
            return 4096;
        }
    }

    private String escapeHtml(String input) {
        if (input == null) {
            return "";
        }

        return input
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");
    }

    private void scrollToBottom() {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                JScrollBar verticalBar = chatScroll.getVerticalScrollBar();
                verticalBar.setValue(verticalBar.getMaximum());
            }
        });
    }

    private String timestamp() {
        return LocalTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss"));
    }

    private void copyChat() {
        if (chatHistory.isEmpty()) {
            return;
        }

        StringBuilder builder = new StringBuilder();
        for (String entry : chatHistory) {
            builder.append(entry).append("\n");
        }

        StringSelection selection = new StringSelection(builder.toString());
        Toolkit.getDefaultToolkit().getSystemClipboard().setContents(selection, null);

        JOptionPane.showMessageDialog(
                frame,
                "Gesamter Chat in die Zwischenablage kopiert!",
                "Chat kopieren",
                JOptionPane.INFORMATION_MESSAGE
        );
    }

    private void closeGpuResources() {
        if (gpuKernels != null) {
            try {
                gpuKernels.close();
            } catch (Exception ignored) {
            }
            gpuKernels = null;
        }

        if (windowsBindings != null) {
            try {
                windowsBindings.close();
            } catch (Exception ignored) {
            }
            windowsBindings = null;
        }
    }
}