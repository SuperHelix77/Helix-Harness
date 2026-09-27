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

## Why shapes are the blocker, and the route to them

A Metal port needs, per tensor: shape, dtype, and the order tensors are
consumed in. The first is the blocker. Three routes, in increasing cost:

1. **Derive shapes from sizes.** Viable only if the layout assumptions are
   right. Current evidence: the sizes do not factor as `k*C*C` for
   C in {64,128,256}, so either the layout is NCHW, or scales/biases are
   interleaved with the weights, or the element size is not 1 byte. Resolving
   this needs one known tensor to calibrate against.
2. **Read the DXIL.** Each pass's shader declares its bindings and the
   constants it uses. This yields shapes and order directly, at the cost of
   5,176 shader disassembly passes. Expensive but mechanical.
3. **Find a known FSR4 v07 model definition.** It exists in the wild as
   community reverse-engineering work, but it is third-party and unverified,
   so it could only be used as a hypothesis to test, never as ground truth.

Route 2 is the honest one: it derives the answer from the artifact rather than
from someone's claim about it.

## Standing position

The FSR4 licence does not grant reimplementation or redistribution rights. The
project owner has accepted that risk and directed extraction. Nothing is
vendored into this repository, no weight bytes are written to disk, and the
binary is not executed. This document records what was learned about the
artifact's structure; it is not a licence to ship.
