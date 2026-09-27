# Provenance and scope

## What this project is

A native Apple-Silicon/Metal implementation of an FSR-class temporal
upscaler, built to be faster than the reference and measurably better in
picture quality, with CrossOver as the eventual host.

## Which FSR, and why

The objective is an **FSR 4.1** Mac port. That has one hard blocker and one
path around it, and it is important to be precise about which is which.

### FSR 4.1.1 cannot be ported from the official distribution

AMD publishes FSR 4.1.1 only as a signed Windows x86-64 binary,
`amd_fidelityfx_upscaler_dx12.dll`, shipped in the FidelityFX SDK under
`Kits/FidelityFX/signedbin/`. The public SDK tree contains **no FSR4 source and
no FSR4 model weights**. A DX12 DLL is not portable to Metal: it cannot execute
on Apple Silicon, and reimplementing it requires the weights and the graph,
neither of which AMD publishes.

### What is in the open SDK

The SDK ships the **FSR3** upscaler as source, under the MIT licence. This was
verified rather than assumed:

- inventory: `../evidence/fsr3-source-inventory-20260927.json`
- source commit: `60f4ea81909200d8542eca14dccb2628b763a9a3`
- 70 files, 891,575 bytes
- **51 of 51 C/C++ source files carry the MIT permission grant** verbatim
  ("Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software ... to use, copy, modify, merge, publish, distribute,
  sublicense, and/or sell copies"), copyright Advanced Micro Devices, Inc.

FSR3 is fully algorithmic. It has **no learned weights** - its only constant
data is a small Lanczos lookup table. That means the entire reconstruction
path is reproducible from published source.

### The user-supplied FSR4 binary

A user-provided FSR 4.1.1 DLL was retrieved from pixeldrain and identified
statically. It was **not executed**.

- inventory: `../evidence/fsr4-binary-inventory-20260927.json`
- sha256 `4e7dc37aebea3a90e3d3cc43e24cb2b54176b2535315f20dbe63b3b7cfc56b1e`
- 65,657,608 bytes, PE32+ x86-64, **not Authenticode signed**
- exports only `UpdateFfxApiProvider` and `UpdateFfxApiProviderEx`
- carries `Fsr4UpscalerModel` and `Fsr4Int8UpscalerModel` classes and
  `FFX_EFFECT_FSR4UPSCALER`

Composition analysis: its `.rdata` is 63.4 MB, of which **5,176 DXIL/DXBC
shader containers account for ~50 MB**. The remaining ~23 MB contains
tensor-shaped data, and one region begins with a regular 24-byte-stride record
table of plausible tensor byte counts (133,380 / 12,704 / 12,563 / 12,712 /
45,466 / 44,310 / 45,805 ...), distributed like a convolutional network.

**The 27 distinct `fsr4_model_v07_*` strings in that binary are DXBC/DXIL
shader debug labels, not a weight manifest.** An earlier pass of this project
recorded them as weights; that was wrong and the inventory now carries the
correction.

## Licensing position

| Item | Licence | Redistribution / reimplementation |
|---|---|---|
| FSR3 upscaler source | MIT | permitted, notice retained |
| FSR 4.1.1 source/weights | not published | no grant exists |
| FSR4 weights inside the binary | AMD proprietary | **not cleared** |

Extracting FSR4 weights from the supplied binary and reimplementing the network
would be using AMD's intellectual property without a grant. That is a
licensing decision for the project owner, not an engineering judgement, and it
has not been taken here. No weight bytes have been extracted, validated, or
dequantised. The binary is not vendored into this repository; only its hashes
and a static inventory are recorded.

## What is therefore being built

A Metal implementation of the **FSR3** temporal upscaler, ported from the
MIT-licensed reference, plus a quality layer designed against the measured
artefact classes the objective names. This is not a substitute for FSR4 and is
not claimed to be one. It is the path that is both legally clean and
achievable today, and it produces the working Metal pipeline and the FSR-class
quality baseline that the FSR4 work would later need anyway.

Concretely, the ports are of the FSR3 mechanisms that matter for quality:

- Lanczos2 reconstruction, including AMD's polynomial approximation
- RCAS contrast-adaptive sharpening
- temporal accumulation in YCoCg with a clipping box
- the lock mechanism, which is what suppresses shimmer

## Claims discipline

Per Q38 AGENTS.md section 10 and 17: quality is a hard admissibility
constraint, not a trade for speed, and surprising results require hostile
replication before promotion. So this project will report:

- same-run ratios, never bare absolute milliseconds (this host has two GPU
  clock states 2-4x apart)
- negative results as readily as positive ones
- every claim tagged with what class of evidence backs it
