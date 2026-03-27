package com.aresstack.winacp.windows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the FFM-based Windows native bindings.
 * <p>
 * These tests call directly into {@code dxgi.dll}, {@code d3d12.dll},
 * and {@code DirectML.dll} using Java 21 Foreign Function & Memory API.
 * They only run on Windows (annotated with {@code @EnabledOnOs}).
 */
class WindowsBindingsTest {

    @Test
    void isSupported_onWindows_returnsTrue() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("windows")) {
            assertTrue(WindowsBindings.isSupported());
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void createDxgiFactory1_succeeds() throws Exception {
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var factory = DxgiBindings.createFactory1(arena);
            assertNotNull(factory);
            assertFalse(factory.equals(java.lang.foreign.MemorySegment.NULL));
            DxgiBindings.release(factory);
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void enumAdapters1_findsAtLeastOneAdapter() throws Exception {
        try (var arena = java.lang.foreign.Arena.ofConfined()) {
            var factory = DxgiBindings.createFactory1(arena);
            var adapter = DxgiBindings.enumAdapters1(factory, 0, arena);
            assertNotNull(adapter, "Should find at least one GPU adapter");
            DxgiBindings.release(adapter);
            DxgiBindings.release(factory);
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void fullStack_dxgi_d3d12_directml() throws Exception {
        try (WindowsBindings bindings = new WindowsBindings()) {
            bindings.init("auto");
            assertTrue(bindings.isInitialised());
            assertNotNull(bindings.getD3d12Device());
            assertNotNull(bindings.getCommandQueue());
            // DirectML might not be available on all machines, but the D3D12 stack must work
        }
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void directMlBindings_isAvailable_checksWithoutCrash() {
        // Should not throw – just returns true/false
        boolean available = DirectMlBindings.isAvailable();
        System.out.println("DirectML available: " + available);
    }

    @Test
    void hresult_succeeded_and_failed() {
        assertTrue(HResult.succeeded(0));           // S_OK
        assertTrue(HResult.succeeded(1));           // S_FALSE
        assertTrue(HResult.failed(0x80070057));     // E_INVALIDARG (bit 31 set)
        assertTrue(HResult.failed(-2147024809));    // same as above, signed
        assertEquals("0x80070057", HResult.toHexString(0x80070057));
    }
}

