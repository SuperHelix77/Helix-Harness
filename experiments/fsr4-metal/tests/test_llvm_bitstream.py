"""Tests for the LLVM bitstream reader.

The parser exists to read DXIL out of the FSR4 binary. It is validated here
against synthetic streams that are constructed byte by byte, so its correctness
does not depend on the binary being present and is not merely asserted.

The important regression this guards: LLVM encodes abbrev ids, block ids and
block lengths with the stream's *abbrev width* bits per byte (2 or 4), not with
7 bits per byte. Reading them as base-128 yields entirely wrong block ids that
still look plausible, which is how a parser can appear to work and silently
return nothing useful.
"""
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llvm_bitstream import Bitstream  # noqa: E402


def vbr(n, width=2):
    mask = (1 << width) - 1
    out = bytearray()
    shift = 0
    while True:
        b = (n >> shift) & mask
        shift += width
        if n >> shift:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def build_module(width=2, block_id=8):
    body = bytearray()
    body += vbr(2, width)      # DEFINE_ABBREV, abbrev id 2
    body += vbr(1, width)      # one field
    body += vbr(0, width)      # not local
    body += bytes([0x14])      # field op: VBR6, no literal
    rec = vbr(2, width) + vbr(1234, width)
    body += rec
    inner = len(body) + len(vbr(0, width))
    blk = vbr(1, width) + vbr(block_id, width) + vbr(inner, width) + bytes(body) + vbr(0, width)
    return b"BC\xc0\xde" + vbr(width, width) + blk


class BitstreamBasics(unittest.TestCase):
    def test_rejects_non_bitstream(self):
        with self.assertRaises(ValueError):
            Bitstream(b"NOPE" + b"\x02\x01\x08\x00\x00\x00")

    def test_reads_abbrev_width(self):
        for w in (2, 3, 4):
            self.assertEqual(Bitstream(build_module(w)).abbrev_width, w)

    def test_decodes_module_block_id(self):
        for block_id in (8, 12, 42, 191):
            with self.subTest(block_id=block_id):
                bs = Bitstream(build_module(2, block_id))
                blocks = [i[1] for i in bs.walk() if i[0] == "enter"]
                self.assertEqual(blocks, [block_id])

    def test_width_is_not_base128(self):
        """A 2-bit-width stream is a different encoding from base-128.

        If the reader used 7 bits per byte it would decode some block ids
        incorrectly but would not crash, which is the dangerous case.
        """
        stream = build_module(2, block_id=42)
        bs = Bitstream(stream)
        blocks = [i[1] for i in bs.walk() if i[0] == "enter"]
        self.assertEqual(blocks, [42])


class VBRWidthSemantics(unittest.TestCase):
    def test_multibyte_value_roundtrips(self):
        # 3000 does not fit in one 2-bit chunk.
        for w in (2, 3, 4):
            enc = vbr(3000, w)
            val, pos = __import__("llvm_bitstream")._vbr(enc, 0, w)
            self.assertEqual(val, 3000)
            self.assertEqual(pos, len(enc))

    def test_large_value_roundtrips(self):
        enc = vbr(1648, 2)
        val, pos = __import__("llvm_bitstream")._vbr(enc, 0, 2)
        self.assertEqual(val, 1648)


if __name__ == "__main__":
    unittest.main()
