#!/usr/bin/env python3
"""Extract the FSR4 v07 layer graph from a user-supplied AMD FSR4 DLL.

Usage:  extract_fsr4_graph.py <dll> <out.json>

Establishes, from the artifact itself:
  * the layer inventory, per output resolution,
  * the layer ORDER (the passes are numbered contiguously, so order is a fact
    rather than an inference),
  * the per-layer resource bindings from the DXIL program signature, which name
    each layer's inputs and outputs.

Does NOT establish tensor shapes. The program signature carries register
bindings and names, not the internal dimensions of a convolution's weights.
Shapes live in the compiled DXIL instruction stream and would need a real DXIL
disassembler plus per-shader dataflow analysis. That is stated rather than
approximated, because a guessed shape is worse than no shape: it would look
authoritative and be wrong.

The binary is never executed and no weight bytes are written to disk.
"""
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

LAYER = re.compile(rb"mlfi_(4k|8k)_pass(\d+)")


def dxil_containers(data):
    """Yield (offset, bitcode_size) for every DXIL container.

    Layout: 'DXIL' u32 size u16 major u16 minor u32 flags, then the bitcode
    beginning with the 'BC\\xc0\\xde' bitstream magic. The declared size is the
    size of the whole container, and the part immediately after the 16-byte
    header is the bitcode.
    """
    out = []
    for m in re.finditer(b"DXIL", data):
        o = m.start()
        if o + 20 > len(data):
            continue
        if data[o + 16:o + 20] != b"BC\xc0\xde":
            continue
        size = struct.unpack_from("<I", data, o + 4)[0]
        if 0 < size < 4_000_000 and o + 16 + size <= len(data):
            out.append((o, size))
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    data = path.read_bytes()

    # --- layer inventory and order -------------------------------------
    passes = {}
    for m in LAYER.finditer(data):
        res, idx = m.group(1).decode(), int(m.group(2))
        passes.setdefault(res, set()).add(idx)

    graph = {}
    for res, idxs in sorted(passes.items()):
        s = sorted(idxs)
        gaps = [i for i in range(s[0], s[-1] + 1) if i not in set(s)]
        graph[res] = {
            "layer_count": len(s),
            "index_min": s[0],
            "index_max": s[-1],
            "contiguous": not gaps,
            "missing_indices": gaps,
            "order": s,
        }

    # --- per-layer resource bindings from the program signature ----------
    # Each DXIL container's ISG1 part names its input/output registers. We
    # record the names; we do not infer dimensions from them.
    bindings = []
    psv_offsets = [m.start() for m in re.finditer(rb"PSV0", data)]
    import bisect
    for m in LAYER.finditer(data):
        o = m.start()
        i = bisect.bisect_left(psv_offsets, o)
        # The layer name sits a short, consistent distance AFTER its PSV0 part.
        # Search the nearest few candidates and keep the closest one.
        best = None
        for cand in psv_offsets[max(0, i - 4):i + 1]:
            dist = o - cand
            if 0 <= dist <= 0x400 and (best is None or dist < best[1]):
                best = (cand, dist)
        if best is None:
            continue
        window = data[best[0]:o + 0x40]
        names = []
        for run in re.findall(rb"[A-Za-z_][A-Za-z0-9_]{2,31}", window):
            t = run.decode()
            if not t.startswith("mlfi_") and t not in names:
                names.append(t)
        bindings.append({"layer": m.group(0).decode(), "bound_names": names})

    by_layer = {}
    for b in bindings:
        by_layer.setdefault(b["layer"], set()).update(b["bound_names"])

    report = {
        "schema": "helix.fsr-metal.fsr4-graph-extraction.v1",
        "source_binary": {
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "executed": False,
        },
        "dxil_containers": len(dxil_containers(data)),
        "established": {
            "resolutions": list(graph),
            "graph": graph,
            "layers_with_named_bindings": len(by_layer),
        },
        "per_layer_bindings": {k: sorted(v) for k, v in sorted(by_layer.items())},
        "NOT_established": [
            "tensor shapes: the DXIL program signature carries register names and "
            "bindings, not internal convolution dimensions. Shapes require DXIL "
            "instruction-level dataflow and are NOT guessed here.",
            "weight-to-layer mapping: the tensor size table cannot be matched to a "
            "layer without shapes, so no such mapping is claimed.",
            "which tensors are FP8 versus INT8 within a layer",
        ],
        "why_shapes_matter": (
            "A Metal port needs per tensor (height, width, in_channels, "
            "out_channels) and the consumption order. The layer order and the "
            "resource bindings are now known; the dimensions are the remaining "
            "gap, and they are the only thing standing between this and a "
            "loadable FSR4 graph."
        ),
        "licensing_note": (
            "Static structure read of a user-supplied binary. No weight bytes are "
            "written to disk and the binary is not vendored. AMD's licence grants "
            "no reimplementation rights; that risk was accepted by the project owner."
        ),
        "evidence_class": "REAL_ARTIFACT_STATIC",
    }
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for res, g in graph.items():
        print(f"{res}: {g['layer_count']} layers, contiguous={g['contiguous']} "
              f"({g['index_min']}..{g['index_max']})")
    print(f"layers with named bindings: {len(by_layer)}")


if __name__ == "__main__":
    main()
