# Shape recovery: what I established, and where it stalls

Date: 2026-09-27. Companion to `FSR4_EXTRACTION.md`.

## The goal

The FSR4 graph is recovered (191 layers, contiguous, two resolutions). The 479
tensor sizes are recovered. What is missing is the mapping from sizes to
shapes, and without shapes the weights cannot be loaded into Metal.

## What I ruled out, with evidence

**The program signature does not carry dimensions.** `PSV0` holds a
referenced-resource table keyed by numeric register ID. The only readable
strings anywhere in the signature window are the container tags `DXIL`, `PSV0`,
`STAT`, `SFI0`, `ISG1`, `OSG1`. There are no tensor names and no dimensions.
This is now measured rather than assumed: 436 layer names were checked against
1,294 signature blocks, and the signature windows contain no dimension
information at all.

**The sizes do not factor into obvious shapes.** For FSR4's channel counts
(64/128/256) a 3x3 convolution of C->C in FP8 would be exactly `9*C*C` bytes.
The observed sizes (12,704 / 12,563 / 12,712 / 43,310 / 45,466 ...) do not
factor that way. So either the layout is not what I assumed, or scales and
biases are packed alongside the weights, or the element size is not 1 byte.
Guessing a shape from this would be fabrication.

## Where it stalls

The dimensions live in the compiled DXIL. To read them I need the module
bitcode, and I have not got a correct handle on the container layout.

What I established:

- There are 2,588 DXIL containers, together accounting for **22.0 MB** of the
  63.5 MB `.rdata` section. The remaining 41.5 MB is tensor data, other
  tables, and unclassified content.
- Each container begins `DXIL` + declared size + `version 16.0`, and contains a
  `DxilProgramHeader` whose sub-parts are tagged `SFI0`, `ISG1`, `OSG1`,
  `PSV0`, `STAT`.
- The `BC\xc0\xde` byte sequence appears 2,588 times, but **always followed by
  the same 8 bytes**. That is the *bitcode header* of a DXIL program header
  (offset, size), not a standalone LLVM module stream.
- I wrote and unit-tested a real LLVM bitstream parser
  (`scripts/llvm_bitstream.py`), validating it against a synthetic stream I
  constructed by hand. It correctly decodes a module block. Fixed during that
  work: the variable-width integers in a bitstream use the stream's *abbrev
  width* (2 or 4), not 7 bits per byte; reading them as base-128 produces
  plausible but entirely wrong block ids, which is exactly what happened on
  first contact.
- Against the real containers the parser finds **no** module stream, because
  the `BitcodeOffset` field appears to be relative to a base I have not
  identified, and every offset I have tried lands in unrelated data.

So the parser is correct and the container addressing is not. I have not solved
the addressing.

## Why I stopped rather than continuing to guess

I have tried, in order: a 24-byte record stride, a 16-byte stride, a 28-byte
stride; byte counts as u32, u24 and 3-byte LE; pointer tables, LEA-reference
scans, RTTI tracing, and a DXIL signature walk. The manifest and the graph both
fell out of that process, but each took several wrong attempts that produced
plausible-looking partial output rather than failing loudly. The bitstream
addressing is now showing the same pattern: many attempts, no convergence.

The remaining route that does not depend on guessing is to obtain a correct
DXIL container reader from a tool that already knows the format - a proper
`dxc`/`dxilconv` build, or DirectXShaderCompiler's `dxil.dll` - and point it
at the extracted containers. That is a tooling dependency, not a research
problem, and it is the right next step rather than a sixth guess.

## What is not affected

Everything established so far stands on its own evidence and does not depend on
shapes:

- the FSR3-derived Metal upscaler, with correctness measured (reconstruction
  RMSE 0.04878, beating bilinear) and performance measured (4K full pipeline
  13.2-13.8 ms across four runs, inside a 60 fps budget);
- the FSR4 binary identification and the fact that its weights are present;
- the 479-tensor size manifest;
- the 191-layer graph, contiguous, two resolutions.

Shapes are the gate between "we know FSR4 exists and what it is shaped like" and
"we can load FSR4 into Metal". That gate is still closed, and this document
records why rather than implying otherwise.
