"""Lossless INT8 archive codec: small signed core plus exact sparse exceptions.

Inspired by Q38's core/residual accounting, not a copy of Q38 model code.
This reduces stored bytes when the distribution permits. decode() expands to
INT8: it does NOT reduce resident inference weights or establish a speedup.
No FSR parameters or vendor data are included.
"""
import hashlib
import struct

HEADER = struct.Struct("<4sBBHIII32s")
MAGIC = b"HWC1"
BLOCK = 32
MAX_WEIGHTS = 8 * 1024 * 1024


class CodecError(ValueError):
    pass


def encode(raw: bytes) -> bytes:
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_WEIGHTS:
        raise CodecError("Expected 1..8 MiB of signed INT8 bytes")
    payload = bytearray()
    for offset in range(0, len(raw), BLOCK):
        block = raw[offset:offset + BLOCK]
        nibbles = bytearray((len(block) + 1) // 2)
        mask = 0
        exceptions = bytearray()
        for i, byte in enumerate(block):
            value = byte if byte < 128 else byte - 256
            if -8 <= value <= 7:
                nibbles[i // 2] |= (value & 15) << (4 * (i % 2))
            else:
                mask |= 1 << i
                exceptions.append(byte)
        payload.extend(struct.pack("<I", mask))
        payload.extend(nibbles)
        payload.extend(exceptions)
    # Lossless raw fallback. Compare like-for-like headers, not payload vs file.
    mode = 1 if len(payload) < len(raw) else 0
    body = bytes(payload) if mode else raw
    return HEADER.pack(MAGIC, 1, mode, BLOCK, len(raw), len(body), 0,
                       hashlib.sha256(raw).digest()) + body


def decode(blob: bytes) -> bytes:
    if type(blob) is not bytes or not HEADER.size < len(blob) <= MAX_WEIGHTS + HEADER.size:
        raise CodecError("Invalid archive size")
    magic, version, mode, blocksize, count, length, reserved, digest = HEADER.unpack_from(blob)
    if (magic != MAGIC or version != 1 or mode not in (0, 1) or blocksize != BLOCK
            or reserved != 0 or not 1 <= count <= MAX_WEIGHTS
            or length != len(blob) - HEADER.size):
        raise CodecError("Invalid archive header")
    payload = blob[HEADER.size:]
    if mode == 0:
        if length != count:
            raise CodecError("Raw length mismatch")
        raw = payload
    else:
        if length >= count:
            raise CodecError("Non-saving packed representation is not canonical")
        out = bytearray()
        cursor = 0
        for offset in range(0, count, BLOCK):
            n = min(BLOCK, count - offset)
            nibble_bytes = (n + 1) // 2
            if cursor + 4 + nibble_bytes > length:
                raise CodecError("Truncated block")
            mask = struct.unpack_from("<I", payload, cursor)[0]
            cursor += 4
            if mask >> n:
                raise CodecError("Exception mask crosses the logical tail")
            values = payload[cursor:cursor + nibble_bytes]
            cursor += nibble_bytes
            if n % 2 and values[-1] >> 4:
                raise CodecError("Nonzero nibble padding")
            for i in range(n):
                nibble = (values[i // 2] >> (4 * (i % 2))) & 15
                if mask & (1 << i):
                    if nibble != 0 or cursor >= length:
                        raise CodecError("Invalid exception storage")
                    value = payload[cursor]
                    cursor += 1
                    signed = value if value < 128 else value - 256
                    if -8 <= signed <= 7:
                        raise CodecError("Unnecessary exception")
                else:
                    value = (nibble if nibble < 8 else nibble - 16) & 255
                out.append(value)
        if cursor != length:
            raise CodecError("Trailing packed data")
        raw = bytes(out)
    if hashlib.sha256(raw).digest() != digest:
        raise CodecError("Decoded data checksum mismatch")
    return raw


def byte_ledger(raw: bytes, archive: bytes) -> dict:
    if decode(archive) != raw:
        raise CodecError("Archive does not encode the declared weights")
    return {
        "logical_int8_bytes": len(raw),
        "raw_container_bytes": HEADER.size + len(raw),
        "archive_bytes": len(archive),
        "archive_header_bytes": HEADER.size,
        "archive_payload_bytes": len(archive) - HEADER.size,
        "resident_expanded_weight_bytes": len(raw),
        "decode_input_plus_output_logical_bytes": len(archive) + len(raw),
        "allocator_peak_measured": False,
        "mode": "nibble_exact_exceptions" if archive[5] else "raw_fallback",
        "lossless": True,
        "runtime_weight_memory_reduced": False,
        "inference_speedup_measured": False,
        "weight_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
    }
