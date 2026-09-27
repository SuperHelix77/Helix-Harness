# Game-runtime ledger activation — 2026-09-27

Current user instruction: research and begin a lower-demand FSR4 port through
Metal and CrossOver, with subagents where available.

The previous handoff names `docs/future/HELIX_GAME_RUNTIME_LEDGER.md` and records
FSR/game-runtime work as parked. Today's instruction reactivates the SR lane.
The exact ledger did not resolve on the inspected GitHub main or v3-release
branches. The fetched main snapshot is
`b6b86756bc73dda30018c06158b12ee0eac2b600`.
This document is a new activation record, not a reconstruction of unseen text.

Workspace: `/Users/mert/Desktop/Helix-FSR-Metal`.
Repository destination: `experiments/fsr4-metal/` on an isolated research branch
of `SuperHelix77/Helix-Harness`. No production integration or main-branch merge.

Subagent coordination was attempted twice and rejected with
`WORKER_IDENTITY_LOST`; no subagents were deployed. Implementation and
verification in this capsule were performed directly.

The unrelated offline video upscaler `MetalScale` was inspected only to identify
it and was left unchanged. No game/bottle/kernel changes were made. Existing
Helix/Q38 checkouts were not edited. A separate sparse clone is used to publish.

Resume from `RESEARCH.md` and the evidence receipts. Keep source/weight
provenance, native model correctness, bridge correctness and game performance
as separate gates. Do not convert a successful kernel test into a full FSR claim.
