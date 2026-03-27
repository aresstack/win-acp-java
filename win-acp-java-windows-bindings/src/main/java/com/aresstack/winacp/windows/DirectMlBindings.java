package com.aresstack.winacp.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Java 21 FFM bindings for {@code DirectML.dll} – DirectML device creation.
 * <p>
 * Calls directly into the Windows DirectML system DLL.
 * No third-party runtime (no ONNX Runtime, no wrapper libraries).
 *
 * <h3>Bound functions</h3>
 * <ul>
 *   <li>{@code HRESULT DMLCreateDevice(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**)}</li>
 * </ul>
 *
 * @see <a href="https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro">DirectML overview</a>
 */
public final class DirectMlBindings {

    private static final Logger log = LoggerFactory.getLogger(DirectMlBindings.class);

    private DirectMlBindings() {}

    // ── DML_CREATE_DEVICE_FLAGS ──────────────────────────────────────────
    /** DML_CREATE_DEVICE_FLAG_NONE = 0 */
    public static final int DML_CREATE_DEVICE_FLAG_NONE = 0;
    /** DML_CREATE_DEVICE_FLAG_DEBUG = 1 */
    public static final int DML_CREATE_DEVICE_FLAG_DEBUG = 1;

    // ── Function descriptors ─────────────────────────────────────────────

    /**
     * {@code HRESULT DMLCreateDevice(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**)}
     */
    private static final FunctionDescriptor DML_CREATE_DEVICE_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,   // HRESULT return
                    ValueLayout.ADDRESS,     // ID3D12Device *d3d12Device
                    ValueLayout.JAVA_INT,   // DML_CREATE_DEVICE_FLAGS flags
                    ValueLayout.ADDRESS,     // REFIID riid
                    ValueLayout.ADDRESS      // void **ppv
            );

    // ── Lazy-initialised handle ──────────────────────────────────────────

    private static volatile MethodHandle dmlCreateDeviceHandle;

    private static MethodHandle getDmlCreateDeviceHandle() {
        if (dmlCreateDeviceHandle == null) {
            synchronized (DirectMlBindings.class) {
                if (dmlCreateDeviceHandle == null) {
                    SymbolLookup dml = SymbolLookup.libraryLookup("DirectML.dll", Arena.global());
                    MemorySegment addr = dml.find("DMLCreateDevice")
                            .orElseThrow(() -> new UnsatisfiedLinkError("DMLCreateDevice not found in DirectML.dll"));
                    dmlCreateDeviceHandle = Linker.nativeLinker()
                            .downcallHandle(addr, DML_CREATE_DEVICE_DESC);
                    log.debug("Resolved DMLCreateDevice at {}", addr);
                }
            }
        }
        return dmlCreateDeviceHandle;
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Create a DirectML device backed by the given D3D12 device.
     *
     * @param d3d12Device COM pointer to {@code ID3D12Device}
     * @param flags       combination of {@code DML_CREATE_DEVICE_FLAGS}
     * @param arena       arena for allocations
     * @return COM pointer to {@code IDMLDevice}
     */
    public static MemorySegment createDevice(MemorySegment d3d12Device, int flags, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_IDMLDevice_BYTES);
            MemorySegment ppDevice = arena.allocate(ValueLayout.ADDRESS);

            int hr = (int) getDmlCreateDeviceHandle().invokeExact(d3d12Device, flags, riid, ppDevice);
            HResult.check(hr, "DMLCreateDevice");

            MemorySegment dmlDevice = ppDevice.get(ValueLayout.ADDRESS, 0)
                    .reinterpret(Long.MAX_VALUE);
            log.info("IDMLDevice created: {}", dmlDevice);
            return dmlDevice;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("DMLCreateDevice invocation failed", t);
        }
    }

    /**
     * Check whether DirectML.dll is loadable on this system.
     *
     * @return true if DirectML.dll can be loaded
     */
    public static boolean isAvailable() {
        try {
            getDmlCreateDeviceHandle();
            return true;
        } catch (UnsatisfiedLinkError | Exception e) {
            log.debug("DirectML.dll not available: {}", e.getMessage());
            return false;
        }
    }
}

