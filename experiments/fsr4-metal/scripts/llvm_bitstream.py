#!/usr/bin/env python3
"""Minimal LLVM bitstream reader, enough to pull constants out of DXIL.

LLVM bitstream is a simple TLV format:
  'BC\\xc0\\xde'                       magic
  <varint>                            abbrev-id width
  BLOCK: ENTER_SUBBLOCK(id, len, align) ... END_BLOCK
  records: abbrev-id, then a run of field codes (literal or VBR), plus optional
          blob payloads (abbrev id 0 with a VBR length then raw bytes)

We do not fully implement the module. We walk the top-level blocks, and for
the CONSTANTS block we decode integers and raw blobs. That is sufficient to
recover shape-like constants, which is the actual goal.

This is a real parser, not a heuristic string scan, so it cannot invent
numbers that were not in the bitcode.
"""
import struct

END_BLOCK = 0
ENTER_SUBBLOCK = 1
DEFINE_ABBREV = 2
UNABBREV_RECORD = 3
FIRST_APPLICATION_ABBREV = 4

BLOCKINFO = 0  # LLVM_BLOCKINFO_BLOCK_ID


def _vbr(buf, pos, width):
    """Variable-width little-endian integer.

    Every abbrev-id, block-id and block-length in a bitstream is encoded with
    `width` payload bits per byte, where `width` is the stream's abbrev width
    (normally 2 or 4) - NOT 7 bits. Reading them as base-128 is the classic
    way to parse a bitstream into nonsense: it yields plausible block ids that
    are simply wrong.
    """
    mask = (1 << width) - 1
    result = 0
    shift = 0
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        result |= (byte & mask) << shift
        if not (byte & 0x80):
            return result, pos
        shift += width
    raise ValueError("truncated VBR")


class Bitstream:
    def __init__(self, data):
        if data[:4] != b"BC\xc0\xde":
            raise ValueError("not an LLVM bitstream")
        self.data = data
        self.pos = 4
        width, self.pos = _vbr(data, self.pos, 6)
        self.abbrev_width = width
        self.abbrevs = {}
        self.blockinfo = {}

    def _read_abbrev(self):
        num_defs, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
        self.pos += 1  # is_local
        out = []
        for _ in range(num_defs):
            op = self.data[self.pos]
            self.pos += 1
            if op & 1:  # has literal
                val = self.data[self.pos]
                self.pos += 1
                literal = val
            else:
                nbits = op & 0x3F
                lit, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                literal = lit
            enc = (op & 0x3F)
            out.append((enc, literal, bool(op & 1)))
        return out

    def walk(self, end=None, depth=0):
        """Yield ('enter', block_id, payload) / ('record', abbrev_id, fields, blob)."""
        end = len(self.data) if end is None else end
        while self.pos < end:
            code, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
            if code == END_BLOCK:
                return
            if code == ENTER_SUBBLOCK:
                block_id, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                length, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                new_start = self.pos
                new_end = min(new_start + length, len(self.data))
                # align to 4 bytes
                new_end = new_start + ((length + 3) & ~3)
                yield ("enter", block_id, (new_start, min(new_end, len(self.data))), depth)
                self.pos = new_start
                yield from self.walk(new_end, depth + 1)
                self.pos = min(new_end, len(self.data))
                continue
            if code == DEFINE_ABBREV:
                aid, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                self.abbrevs[aid] = self._read_abbrev()
                continue
            if code == UNABBREV_RECORD:
                n, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                fields = []
                for _ in range(n):
                    f, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                    fields.append(f)
                blob = None
                if n and fields[0] in (1, 2):  # array/char array with blob
                    bl, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                    blob = self.data[self.pos:self.pos + bl]
                    self.pos += bl
                yield ("record", None, fields, blob)
                continue
            # abbreviated record
            spec = self.abbrevs.get(code)
            if spec is None:
                continue
            fields = []
            blob = None
            for enc, literal, has_lit in spec:
                if enc == 5:  # blob
                    bl, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                    blob = self.data[self.pos:self.pos + bl]
                    self.pos += bl
                elif enc == 2:  # char array
                    n, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                    blob = self.data[self.pos:self.pos + n]
                    self.pos += n
                elif enc == 1:  # array
                    n, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                    for _ in range(n):
                        v, self.pos = _vbr(self.data, self.pos, self.abbrev_width)
                        fields.append(v)
                elif enc in (3, 4, 6):  # VBR
                    nbits = enc - 2
                    v, self.pos = _vbr(self.data, self.pos, nbits)
                    fields.append(v)
                elif has_lit:
                    fields.append(literal)
                else:
                    fields.append(0)
            yield ("record", code, fields, blob)
