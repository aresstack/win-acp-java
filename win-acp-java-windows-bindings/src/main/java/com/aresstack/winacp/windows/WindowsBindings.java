package com.aresstack.winacp.windows;

/**
 * Entry point for jextract-generated Windows native bindings.
 * <p>
 * This module encapsulates all FFM / Panama-based interop with
 * Windows 11 DLLs (DXGI, D3D12, DirectML). All low-level native
 * details are confined to this module and must NOT leak outward.
 * <p>
 * Bindings are generated with {@code jextract} from Windows SDK headers.
 * The JVM must be started with {@code --enable-native-access=ALL-UNNAMED}.
 *
 * @see <a href="https://learn.microsoft.com/en-us/windows/ai/directml/">DirectML</a>
 */
public final class WindowsBindings {

    private WindowsBindings() {}

    /**
     * Check whether the current platform supports the Windows native path.
     */
    public static boolean isSupported() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("windows");
    }
}

