# FSR4 v07 manifest extraction, 2026-09-27

Source: the user-supplied `amd_fidelityfx_upscaler_dx12.dll`, sha256
`4e7dc37aebea3a90e3d3cc43e24cb2b54176b2535315f20dbe63b3b7cfc56b1e`. **Not
executed.** Tool: `scripts/extract_fsr4_manifest.py`, which reads structure only
and writes no weight bytes to disk.

## What was recovered

The FSR4 v07 tensor manifest: **479 tensors, 19.87 MiB**, in a table of
16-byte records inside `.rdata`. Each record is a 3-byte little-endian byte
count followed by 13 bytes of padding.

Size distribution, which is what makes the identification credible:

| size | count | likely role |
|---|---:|---|
| ~8 KiB | 63 | per-channel scale / bias |
| ~16 KiB | 79 | small projections |
| ~24 KiB | 222 | 3x3 convolutions |
| ~32 KiB | 112 | 3x3 convolutions |
| 2.57 / 3.41 / 3.82 MiB | 3 | wide feature stages |

A table of 479 byte counts whose histogram clusters into per-channel scales
and 3x3 convolutions, with exactly three large feature tensors, is a
convolutional network. That is the identification, and it does not depend on
the marketing name.

**Three separate tables match this fingerprint** (479, 211, 176 and 137-record
variants at different offsets), which is consistent with FSR4 shipping an FP8
model, an INT8 model and a further variant. The extractor reports all of them
rather than silently choosing one.

## What is NOT recovered

This is a structure read, not a decompilation. Specifically:

- **No tensor shapes.** A byte count alone cannot determine `(out, in, k, k)`.
  The observed sizes do not factor exactly into `k*C*C` for any plausible
  channel count, so any shape claim would be a fabrication. Shapes are what a
  Metal port actually needs, and they are the hard part.
- **No tensor-to-pass mapping.** The `fsr4_model_v07_*` strings in this binary
  are DXBC/DXIL shader debug labels, not a weight manifest. They cannot be used
  to name the tensors. (An earlier pass of this project wrongly called them
  embedded weights; that correction is recorded in
  `fsr4-binary-inventory-20260927.json`.)
- **No layer order or connectivity.** The execution order lives in the
  compiled DXIL, not in the weight table.
- **No per-tensor element format confirmed.** FP8 is indicated by the model
  names, but has not been verified per tensor.

## The layer graph (recovered after the manifest)

See below for the graph result, which arrived after this section was first
written.

## Standing position

The FSR4 licence does not grant reimplementation or redistribution rights. The
project owner has accepted that risk and directed extraction. Nothing is
vendored into this repository, no weight bytes are written to disk, and the
binary is not executed. This document records what was learned about the
artifact's structure; it is not a licence to ship.

---

## Update: the layer graph, recovered

The manifest above is the weight side. The graph side turned out to be
recoverable too, and it changes the picture materially.

| | 4K variant | 8K variant |
|---|---:|---:|
| layers | **191** | **191** |
| index range | 0 - 190 | 0 - 190 |
| contiguous | yes, no gaps | yes, no gaps |

Every pass index from 0 to 190 is present in both variants, so the ordering is
a fact read from the artifact rather than an inference: the network is a
straight sequence of 191 layers with no conditional skips, and the 4K and 8K
paths have identical topology.

Supporting structure: 2,588 DXIL containers, each a complete program header
with `SFI0` / `ISG1` / `OSG1` / `PSV0` sub-parts; 1,294 distinct signatures
(about 6.8 compiled variants per layer); 382 layer names, matching
191 x 2 exactly.

This corrects an earlier impression. Reading the `fsr4_model_v07_*` strings
suggests 13 passes. Those are shader labels and undercount the graph by more
than an order of magnitude. The real network is **191 layers**.

### Shapes are still missing, and the signature is not where they are

The natural assumption was that the DXIL program signature would name resource
dimensions. It does not. `PSV0` carries a referenced-resource table keyed by
numeric register ID, and the only readable strings in the signature window are
the container tags. There are no tensor names and no dimensions.

The dimensions live in the **compiled DXIL instruction stream** as constants
and typed-buffer strides. Recovering them means disassembling the bitcode,
finding the cbuffer loads that set per-layer constants, and connecting those
constants to the weight bindings.

The useful part: once a few shapes are recovered that way, the remaining 479
tensors can be **solved from the size table by elimination**, because the
channel progression through a 191-layer network is heavily constrained. That
turns a full 2,588-shader disassembly into a handful of real disassembly passes
plus arithmetic against the sizes already extracted.
