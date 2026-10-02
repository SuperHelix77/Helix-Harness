import random
import struct
import unittest
from weight_codec import CodecError, HEADER, byte_ledger, decode, encode


class WeightCodecTests(unittest.TestCase):
    def test_every_int8_value_and_raw_fallback(self):
        raw = bytes(range(256)) * 3
        blob = encode(raw)
        self.assertEqual(decode(blob), raw)
        self.assertEqual(byte_ledger(raw, blob)["mode"], "raw_fallback")

    def test_every_signed_nibble_and_tails(self):
        for n in (1, 7, 31, 32, 33, 63, 64, 65, 1025):
            raw = bytes(((i % 16) - 8) & 255 for i in range(n))
            self.assertEqual(decode(encode(raw)), raw)

    def test_exact_extreme_exceptions(self):
        raw = bytearray([0] * 4097)
        for i in range(0, len(raw), 37):
            raw[i] = 128 if i % 2 else 127
        raw = bytes(raw); blob = encode(raw)
        self.assertEqual(decode(blob), raw)
        self.assertEqual(blob[5], 1)
        ledger = byte_ledger(raw, blob)
        self.assertLess(ledger["archive_bytes"], len(raw))
        self.assertEqual(ledger["resident_expanded_weight_bytes"], len(raw))
        self.assertFalse(ledger["runtime_weight_memory_reduced"])

    def test_random_roundtrip(self):
        for seed in range(4):
            r = random.Random(seed)
            for n in (2, 30, 99, 513):
                raw = bytes(r.randrange(256) for _ in range(n))
                self.assertEqual(decode(encode(raw)), raw)

    def test_deterministic(self):
        self.assertEqual(encode(bytes([1] * 99)), encode(bytes([1] * 99)))

    def test_checksum_corruption(self):
        for raw in (bytes(range(256)), bytes([1] * 256)):
            blob = bytearray(encode(raw)); blob[20] ^= 1
            with self.assertRaises(CodecError): decode(bytes(blob))

    def test_truncation_and_extra_data(self):
        blob = encode(bytes([0] * 512))
        for b in (b"", blob[:-1], blob + b"x", blob[:HEADER.size]):
            with self.assertRaises(CodecError): decode(b)

    def test_empty_and_wrong_type(self):
        for raw in (b"", [1, 2], bytearray([1])):
            with self.assertRaises(CodecError): encode(raw)

    def test_invalid_header(self):
        blob = encode(bytes([0] * 512))
        for offset in (0, 4, 5, 6, 16):
            b = bytearray(blob); b[offset] = 254
            with self.assertRaises(CodecError): decode(bytes(b))

    def test_illegal_tail_mask(self):
        b = bytearray(encode(bytes([0] * 65)))
        # Two full 20-byte blocks, then a one-value tail.
        struct.pack_into("<I", b, HEADER.size + 40, 2)
        with self.assertRaises(CodecError): decode(bytes(b))

    def test_nonzero_tail_nibble(self):
        b = bytearray(encode(bytes([0] * 65))); b[-1] = 0x10
        with self.assertRaises(CodecError): decode(bytes(b))

    def test_ledger_rejects_unrelated_raw(self):
        with self.assertRaises(CodecError): byte_ledger(bytes([1] * 99), encode(bytes([0] * 99)))


if __name__ == "__main__": unittest.main()
