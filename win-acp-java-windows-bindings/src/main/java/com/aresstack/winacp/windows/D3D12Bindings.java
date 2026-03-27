package com.aresstack.winacp.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * Java 21 FFM bindings for {@code d3d12.dll} – Direct3D 12 device creation.
 * <p>
 * Calls directly into the Windows system DLL. No third-party wrapper.
 *
 * <h3>Bound functions</h3>
 * <ul>
 *   <li>{@code HRESULT D3D12CreateDevice(IUnknown *pAdapter, D3D_FEATURE_LEVEL, REFIID riid, void **ppDevice)}</li>
 * </ul>
 *
 * @see <a href="https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-d3d12createdevice">D3D12CreateDevice</a>
 */
public final class D3D12Bindings {

    private static final Logger log = LoggerFactory.getLogger(D3D12Bindings.class);

    private D3D12Bindings() {}

    // ── D3D_FEATURE_LEVEL constants ──────────────────────────────────────
    /** D3D_FEATURE_LEVEL_11_0 = 0xb000 */
    public static final int D3D_FEATURE_LEVEL_11_0 = 0xb000;
    /** D3D_FEATURE_LEVEL_11_1 = 0xb100 */
    public static final int D3D_FEATURE_LEVEL_11_1 = 0xb100;
    /** D3D_FEATURE_LEVEL_12_0 = 0xc000 */
    public static final int D3D_FEATURE_LEVEL_12_0 = 0xc000;
    /** D3D_FEATURE_LEVEL_12_1 = 0xc100 */
    public static final int D3D_FEATURE_LEVEL_12_1 = 0xc100;

    // ── Function descriptors ─────────────────────────────────────────────

    /**
     * {@code HRESULT D3D12CreateDevice(IUnknown *pAdapter, D3D_FEATURE_LEVEL, REFIID riid, void **ppDevice)}
     */
    private static final FunctionDescriptor D3D12_CREATE_DEVICE_DESC =
            FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,   // HRESULT return
                    ValueLayout.ADDRESS,     // IUnknown *pAdapter (nullable)
                    ValueLayout.JAVA_INT,   // D3D_FEATURE_LEVEL (enum → int)
                    ValueLayout.ADDRESS,     // REFIID riid
                    ValueLayout.ADDRESS      // void **ppDevice
            );

    // ── Lazy-initialised handle ──────────────────────────────────────────

    private static volatile MethodHandle d3d12CreateDeviceHandle;

    private static MethodHandle getD3D12CreateDeviceHandle() {
        if (d3d12CreateDeviceHandle == null) {
            synchronized (D3D12Bindings.class) {
                if (d3d12CreateDeviceHandle == null) {
                    SymbolLookup d3d12 = SymbolLookup.libraryLookup("d3d12.dll", Arena.global());
                    MemorySegment addr = d3d12.find("D3D12CreateDevice")
                            .orElseThrow(() -> new UnsatisfiedLinkError("D3D12CreateDevice not found in d3d12.dll"));
                    d3d12CreateDeviceHandle = Linker.nativeLinker()
                            .downcallHandle(addr, D3D12_CREATE_DEVICE_DESC);
                    log.debug("Resolved D3D12CreateDevice at {}", addr);
                }
            }
        }
        return d3d12CreateDeviceHandle;
    }

    // ── Public API ───────────────────────────────────────────────────────

    /**
     * Create a D3D12 device for the given adapter.
     *
     * @param adapter      COM pointer to an {@code IDXGIAdapter} (or {@code NULL} for default)
     * @param featureLevel minimum feature level (e.g. {@link #D3D_FEATURE_LEVEL_11_0})
     * @param arena        arena for allocations
     * @return COM pointer to {@code ID3D12Device}
     */
    public static MemorySegment createDevice(MemorySegment adapter, int featureLevel, Arena arena)
            throws WindowsNativeException {
        try {
            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12Device_BYTES);
            MemorySegment ppDevice = arena.allocate(ValueLayout.ADDRESS);

            int hr = (int) getD3D12CreateDeviceHandle().invokeExact(adapter, featureLevel, riid, ppDevice);
            HResult.check(hr, "D3D12CreateDevice");

            MemorySegment device = ppDevice.get(ValueLayout.ADDRESS, 0)
                    .reinterpret(Long.MAX_VALUE);
            log.info("ID3D12Device created: {} (featureLevel=0x{})  ",
                    device, Integer.toHexString(featureLevel));
            return device;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("D3D12CreateDevice invocation failed", t);
        }
    }

    /**
     * Create a Direct3D 12 command queue for the given device.
     * <p>
     * DirectML needs a DIRECT command queue to dispatch operations.
     *
     * @param device D3D12 device COM pointer
     * @param arena  arena for allocations
     * @return COM pointer to {@code ID3D12CommandQueue}
     */
    public static MemorySegment createCommandQueue(MemorySegment device, Arena arena)
            throws WindowsNativeException {
        try {
            // D3D12_COMMAND_QUEUE_DESC struct: {Type=DIRECT(0), Priority=NORMAL(0), Flags=NONE(0), NodeMask=0}
            // Layout: 4 ints = 16 bytes
            MemorySegment desc = arena.allocate(16);
            desc.set(ValueLayout.JAVA_INT, 0, 0);  // Type = D3D12_COMMAND_LIST_TYPE_DIRECT
            desc.set(ValueLayout.JAVA_INT, 4, 0);  // Priority = NORMAL
            desc.set(ValueLayout.JAVA_INT, 8, 0);  // Flags = NONE
            desc.set(ValueLayout.JAVA_INT, 12, 0); // NodeMask

            MemorySegment riid = ComIID.allocateGuid(arena, ComIID.IID_ID3D12CommandQueue_BYTES);
            MemorySegment ppQueue = arena.allocate(ValueLayout.ADDRESS);

            // ID3D12Device::CreateCommandQueue(this, desc, riid, ppQueue)
            // Vtable: IUnknown(3) + ID3D12Object(3) + ID3D12Device before CreateCommandQueue
            // CreateCommandQueue is slot 8 in ID3D12Device vtable
            MethodHandle createCQ = DxgiBindings.vtableMethod(device, 8,
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,   // HRESULT
                            ValueLayout.ADDRESS,     // this
                            ValueLayout.ADDRESS,     // const D3D12_COMMAND_QUEUE_DESC *
                            ValueLayout.ADDRESS,     // REFIID
                            ValueLayout.ADDRESS      // void **
                    ));

            int hr = (int) createCQ.invokeExact(device, desc, riid, ppQueue);
            HResult.check(hr, "ID3D12Device::CreateCommandQueue");

            MemorySegment queue = ppQueue.get(ValueLayout.ADDRESS, 0)
                    .reinterpret(Long.MAX_VALUE);
            log.info("ID3D12CommandQueue created: {}", queue);
            return queue;
        } catch (WindowsNativeException e) {
            throw e;
        } catch (Throwable t) {
            throw new WindowsNativeException("CreateCommandQueue invocation failed", t);
        }
    }
}

