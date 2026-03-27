package com.aresstack.winacp.windows;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

/**
 * COM interface IID constants and GUID memory layout helpers.
 * <p>
 * A COM GUID is 16 bytes: {@code {Data1: DWORD, Data2: WORD, Data3: WORD, Data4: BYTE[8]}}.
 * We store them as {@link MemorySegment}s in native memory so they can be
 * passed directly to Windows SDK functions that expect {@code REFIID}.
 * <p>
 * All IIDs here come straight from the Windows SDK headers.
 */
public final class ComIID {

    private ComIID() {}

    /** Standard GUID struct layout: 4 + 2 + 2 + 8 = 16 bytes. */
    public static final MemoryLayout GUID_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("Data1"),
            ValueLayout.JAVA_SHORT.withName("Data2"),
            ValueLayout.JAVA_SHORT.withName("Data3"),
            MemoryLayout.sequenceLayout(8, ValueLayout.JAVA_BYTE).withName("Data4")
    ).withName("GUID");

    /** Size of a GUID in bytes. */
    public static final long GUID_SIZE = GUID_LAYOUT.byteSize(); // 16

    // ── DXGI ──────────────────────────────────────────────────────────────
    /** IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387} */
    public static final byte[] IID_IDXGIFactory1_BYTES = guidBytes(
            0x770aae78, (short) 0xf26f, (short) 0x4dba,
            (byte) 0xa8, (byte) 0x29, (byte) 0x25, (byte) 0x3c,
            (byte) 0x83, (byte) 0xd1, (byte) 0xb3, (byte) 0x87);

    /** IID_IDXGIAdapter1 = {29038f61-3839-4626-91fd-086879011a05} */
    public static final byte[] IID_IDXGIAdapter1_BYTES = guidBytes(
            0x29038f61, (short) 0x3839, (short) 0x4626,
            (byte) 0x91, (byte) 0xfd, (byte) 0x08, (byte) 0x68,
            (byte) 0x79, (byte) 0x01, (byte) 0x1a, (byte) 0x05);

    // ── D3D12 ─────────────────────────────────────────────────────────────
    /** IID_ID3D12Device = {189819f1-1db6-4b57-be54-1821339b85f7} */
    public static final byte[] IID_ID3D12Device_BYTES = guidBytes(
            0x189819f1, (short) 0x1db6, (short) 0x4b57,
            (byte) 0xbe, (byte) 0x54, (byte) 0x18, (byte) 0x21,
            (byte) 0x33, (byte) 0x9b, (byte) 0x85, (byte) 0xf7);

    /** IID_ID3D12CommandQueue = {0ec870a6-5d7e-4c22-8cfc-5baae07616ed} */
    public static final byte[] IID_ID3D12CommandQueue_BYTES = guidBytes(
            0x0ec870a6, (short) 0x5d7e, (short) 0x4c22,
            (byte) 0x8c, (byte) 0xfc, (byte) 0x5b, (byte) 0xaa,
            (byte) 0xe0, (byte) 0x76, (byte) 0x16, (byte) 0xed);

    // ── DirectML ──────────────────────────────────────────────────────────
    /** IID_IDMLDevice = {6dbd6437-96fd-423f-a98c-ae5e7c2a573f} */
    public static final byte[] IID_IDMLDevice_BYTES = guidBytes(
            0x6dbd6437, (short) 0x96fd, (short) 0x423f,
            (byte) 0xa9, (byte) 0x8c, (byte) 0xae, (byte) 0x5e,
            (byte) 0x7c, (byte) 0x2a, (byte) 0x57, (byte) 0x3f);

    /**
     * Allocate a native GUID segment from the given byte representation.
     *
     * @param arena arena for the allocation
     * @param guid  16-byte GUID in little-endian SDK layout
     * @return a {@link MemorySegment} of size 16 that can be passed as {@code REFIID}
     */
    public static MemorySegment allocateGuid(Arena arena, byte[] guid) {
        MemorySegment seg = arena.allocate(GUID_LAYOUT);
        MemorySegment.copy(guid, 0, seg, ValueLayout.JAVA_BYTE, 0, 16);
        return seg;
    }

    // ── internal ──────────────────────────────────────────────────────────

    /**
     * Pack a GUID into a 16-byte array in the native struct layout
     * (little-endian Data1/Data2/Data3, then raw Data4 bytes).
     */
    private static byte[] guidBytes(int d1, short d2, short d3,
                                     byte b0, byte b1, byte b2, byte b3,
                                     byte b4, byte b5, byte b6, byte b7) {
        byte[] b = new byte[16];
        // Data1 – little-endian DWORD
        b[0] = (byte) (d1);
        b[1] = (byte) (d1 >>> 8);
        b[2] = (byte) (d1 >>> 16);
        b[3] = (byte) (d1 >>> 24);
        // Data2 – little-endian WORD
        b[4] = (byte) (d2);
        b[5] = (byte) (d2 >>> 8);
        // Data3 – little-endian WORD
        b[6] = (byte) (d3);
        b[7] = (byte) (d3 >>> 8);
        // Data4 – raw bytes
        b[8] = b0; b[9] = b1; b[10] = b2; b[11] = b3;
        b[12] = b4; b[13] = b5; b[14] = b6; b[15] = b7;
        return b;
    }
}

