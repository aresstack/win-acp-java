package com.aresstack.winacp.windows;

/**
 * Utility for dealing with Windows HRESULT codes.
 * <p>
 * Every Windows SDK function returns an {@code HRESULT} (a 32-bit signed int).
 * Bit 31 is the severity flag: 0 = success, 1 = error.
 */
public final class HResult {

    /** S_OK – the universal "it worked" code. */
    public static final int S_OK = 0;

    private HResult() {}

    /** {@code true} when the HRESULT indicates success (bit 31 clear). */
    public static boolean succeeded(int hr) {
        return hr >= 0;
    }

    /** {@code true} when the HRESULT indicates failure (bit 31 set). */
    public static boolean failed(int hr) {
        return hr < 0;
    }

    /**
     * Throw {@link WindowsNativeException} if the HRESULT indicates failure.
     *
     * @param hr   HRESULT value
     * @param call human-readable description of the call that produced it
     */
    public static void check(int hr, String call) throws WindowsNativeException {
        if (failed(hr)) {
            throw new WindowsNativeException(
                    String.format("%s failed: HRESULT 0x%08X", call, hr), hr);
        }
    }

    /** Format an HRESULT as a hex string, e.g. {@code 0x80070057}. */
    public static String toHexString(int hr) {
        return String.format("0x%08X", hr);
    }
}

