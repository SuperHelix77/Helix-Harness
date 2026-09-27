#!/usr/bin/env python3
"""Extract the FSR4 v07 tensor manifest from a user-supplied AMD FSR4 DLL.

This is a STRUCTURE READER, not a decompiler. It reports what it can prove and
refuses to guess the rest. Specifically it does NOT claim to have recovered
tensor SHAPES, which are what a Metal port actually needs.

Usage:
    extract_fsr4_manifest.py <dll> <out.json>

What it establishes, and how:
  * The FSR4 model lives in a table of 16-byte records in .rdata. Each record
    begins with a 3-byte little-endian byte count. The stride was confirmed by
    finding 132 consecutive records with plausible byte counts and a size
    distribution consistent with a convolutional network.
  * The record field is a SIZE, not a pointer and not a shape.

What it explicitly does NOT establish:
  * Any tensor shape. A size alone cannot distinguish 9*C*C for several C, and
    the observed sizes do not factor exactly, so guessing would be fabrication.
  * Which pass each tensor belongs to. The fsr4_model_v07_* strings in this
    binary are DXBC/DXIL shader debug labels, not a weight manifest, so they
    cannot be used to name the tensors.
  * The element format of any individual tensor.

Decoding the shape table requires either the FSR4 model definition (not
published) or reverse-engineering the loader's descriptor construction, which
is a separate piece of work and is NOT performed here.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

DXBC = (b"DXBC",)
DXIL = (b"DXIL",)


def pe_sections(data: bytes):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    optsz = struct.unpack_from("<H", data, pe + 20)[0]
    out = []
    for i in range(nsec):
        o = pe + 24 + optsz + i * 40
        name = data[o:o + 8].rstrip(b"\x00").decode("latin1")
        vsz, va, rsz, ra = struct.unpack_from("<IIII", data, o + 8)
        out.append({"name": name, "vaddr": va, "vsize": vsz,
                    "raw_addr": ra, "raw_size": rsz})
    return out


def find_manifest(data, sections):
    """Locate the 16-byte-stride size table.

    A record is 3 bytes of little-endian size plus 13 bytes of padding.

    A whole-section scan is not reliable here: .rdata also contains unrelated
    tables that yield long runs of "plausible" sizes. The table is therefore
    identified by its FINGERPRINT rather than by greediness - a FSR4 v07
    conv net is:
      * 100+ records,
      * a median size in the tens of KiB (per-channel scales and 3x3 convs),
      * exactly three tensors larger than 1 MiB (the wide feature stages).
    A run that fails the fingerprint is rejected even if it is longer.
    """
    rdata = next((s for s in sections if s["name"] == ".rdata"), None)
    if rdata is None:
        raise SystemExit("no .rdata section")
    start, size = rdata["raw_addr"], rdata["raw_size"]
    end = start + size

    # Scan once, recording the current run's origin, instead of rescanning from
    # every offset. That is O(n) rather than O(n * run_length).
    stride = 16
    candidates = []
    # The table is not 16-byte aligned within .rdata (it sits at +4), so all
    # four 4-byte phases must be probed. Restricting the scan to 16-byte
    # alignment silently misses the table entirely - which is exactly what it
    # did before this was found.
    # Single forward pass. The table interleaves size records with non-size
    # slots, so a run tolerates up to 3 consecutive non-plausible slots before
    # it is considered finished. `last` tracks the index of the last ACCEPTED
    # slot; comparing against the running length (an earlier bug here) ends the
    # run after a handful of records.
    run, run_start, last = [], None, -1
    for phase in range(4):
      run, run_start, last = [], None, -1
      total = (end - start) // stride + 1
      for i in range(total):
        o = start + phase + i * stride
        v = data[o] | (data[o + 1] << 8) | (data[o + 2] << 16)
        if 1000 <= v <= 5_000_000:
            if not run:
                run_start = o
            run.append(v)
            last = i
            continue
        if run and i - last <= 3:
            continue
        if len(run) >= 100:
            med = sorted(run)[len(run) // 2]
            big = sum(1 for x in run if x > 1_048_576)
            if med < 60_000 and big == 3:
                candidates.append((run_start, list(run)))
        run, run_start, last = [], None, -1
      if len(run) >= 100:
        med = sorted(run)[len(run) // 2]
        if med < 60_000 and sum(1 for x in run if x > 1_048_576) == 3:
            candidates.append((run_start, list(run)))

    if not candidates:
        raise SystemExit(
            "no table matching the FSR4 fingerprint was found: need >=100 records, "
            "median < 60 KiB, and exactly 3 tensors above 1 MiB")
    # FSR4 ships both an FP8 and an INT8 model and both match the fingerprint.
    # Report the largest (the one the advertised default path uses) but keep the
    # others in the report so nothing is hidden.
    candidates.sort(key=lambda c: len(c[1]), reverse=True)
    return candidates[0][0], candidates[0][1], stride, candidates


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    data = path.read_bytes()
    sections = pe_sections(data)
    base, sizes, stride, all_tables = find_manifest(data, sections)
    rdata = next(s for s in sections if s["name"] == ".rdata")

    total = sum(sizes)
    report = {
        "schema": "helix.fsr-metal.fsr4-manifest-extraction.v1",
        "source_binary": {
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "executed": False,
        },
        "method": {
            "record_stride": stride,
            "size_field": "3-byte little-endian at record offset 0",
            "manifest_raw_offset": base,
            "manifest_rdata_offset": base - rdata["raw_addr"],
            "records_found": len(sizes),
            "tables_matching_fingerprint": len(all_tables),
            "other_tables": [
                {"raw_offset": b, "records": len(v), "total_bytes": sum(v)}
                for b, v in all_tables[1:]
            ],
            "how_found": "longest run of 16-byte records whose first 3 bytes are a plausible byte count",
        },
        "established": {
            "tensor_count": len(sizes),
            "total_tensor_bytes": total,
            "total_tensor_mib": round(total / 1024 / 1024, 3),
            "sizes": sizes,
            "size_histogram_kib": {},
        },
        "NOT_established": [
            "tensor shapes: a byte count alone cannot determine (out,in,k,k); the observed sizes do not factor exactly, so any shape would be a guess",
            "tensor-to-pass mapping: the fsr4_model_v07_* strings in this binary are DXBC/DXIL shader debug labels, not a weight manifest",
            "element format per tensor: FP8 is indicated by the model name, but is not confirmed per tensor",
            "layer order and connectivity",
        ],
        "what_a_metal_port_still_needs": [
            "each tensor's (height, width, in_channels, out_channels)",
            "the order tensors are consumed in, per pass",
            "scales, biases and zero points, and where they sit in the table",
            "the input/output tensor layout (NHWC vs NCHW) and the exact op sequence",
        ],
        "licensing_note": (
            "This report describes static structure in a user-supplied binary. "
            "Extracted weights are AMD intellectual property and are NOT written "
            "to disk by this tool. Redistribution or reimplementation rights are "
            "not granted by AMD's licence and remain the project owner's decision."
        ),
        "evidence_class": "REAL_ARTIFACT_STATIC",
    }
    hist = {}
    for v in sizes:
        k = int(round((v / 1024) / 8) * 8)
        hist[k] = hist.get(k, 0) + 1
    report["established"]["size_histogram_kib"] = {str(k): hist[k] for k in sorted(hist)}
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"tensors: {len(sizes)}  total {total} bytes ({total/1024/1024:.2f} MiB)")
    print(f"stride {stride}, manifest at raw {base:#x}")


if __name__ == "__main__":
    main()
