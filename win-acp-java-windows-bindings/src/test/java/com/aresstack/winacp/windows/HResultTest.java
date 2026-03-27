package com.aresstack.winacp.windows;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link HResult} diagnostics.
 */
class HResultTest {

    @Test
    void succeeded_trueForSOK() {
        assertTrue(HResult.succeeded(HResult.S_OK));
    }

    @Test
    void succeeded_falseForError() {
        assertFalse(HResult.succeeded(HResult.E_INVALIDARG));
        assertFalse(HResult.succeeded(HResult.DXGI_ERROR_DEVICE_REMOVED));
    }

    @Test
    void failed_trueForError() {
        assertTrue(HResult.failed(HResult.E_INVALIDARG));
        assertTrue(HResult.failed(HResult.E_OUTOFMEMORY));
    }

    @Test
    void failed_falseForSOK() {
        assertFalse(HResult.failed(HResult.S_OK));
    }

    @Test
    void describe_knownCode_includesName() {
        String desc = HResult.describe(HResult.E_INVALIDARG);
        assertTrue(desc.contains("E_INVALIDARG"), "Should contain symbolic name");
        assertTrue(desc.contains("0x80070057"), "Should contain hex code");
    }

    @Test
    void describe_unknownCode_hexOnly() {
        String desc = HResult.describe(0x88888888);
        assertEquals("0x88888888", desc);
    }

    @Test
    void describe_sok_includesName() {
        String desc = HResult.describe(HResult.S_OK);
        assertTrue(desc.contains("S_OK"));
    }

    @Test
    void toHexString_format() {
        assertEquals("0x00000000", HResult.toHexString(0));
        assertEquals("0x80070057", HResult.toHexString(HResult.E_INVALIDARG));
    }

    @Test
    void check_successDoesNotThrow() {
        assertDoesNotThrow(() -> HResult.check(HResult.S_OK, "test"));
    }

    @Test
    void check_failureThrowsWithDiagnosticMessage() {
        WindowsNativeException ex = assertThrows(WindowsNativeException.class, () ->
                HResult.check(HResult.E_INVALIDARG, "TestCall"));
        assertTrue(ex.getMessage().contains("TestCall"));
        assertTrue(ex.getMessage().contains("E_INVALIDARG"));
        assertEquals(HResult.E_INVALIDARG, ex.getHresult());
    }

    @Test
    void check_dxgiError_includesSymbolicName() {
        WindowsNativeException ex = assertThrows(WindowsNativeException.class, () ->
                HResult.check(HResult.DXGI_ERROR_DEVICE_REMOVED, "GPU op"));
        assertTrue(ex.getMessage().contains("DXGI_ERROR_DEVICE_REMOVED"));
    }
}

