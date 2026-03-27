package com.aresstack.winacp.windows;

/**
 * Exception thrown when a Windows native call fails.
 * <p>
 * Carries the raw {@code HRESULT} so callers can decide how to react.
 * This replaces the old OnnxRuntimeBridgeException – no third-party
 * runtime types leak out of this module.
 */
public class WindowsNativeException extends Exception {

    private final int hresult;

    public WindowsNativeException(String message) {
        super(message);
        this.hresult = 0;
    }

    public WindowsNativeException(String message, int hresult) {
        super(message);
        this.hresult = hresult;
    }

    public WindowsNativeException(String message, Throwable cause) {
        super(message, cause);
        this.hresult = 0;
    }

    /** The raw HRESULT returned by the failed Windows API call, or 0 if not applicable. */
    public int getHresult() {
        return hresult;
    }
}

