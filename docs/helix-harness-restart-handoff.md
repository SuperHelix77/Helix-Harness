# Helix Harness Restart Handoff

Updated: 2026-09-16 — v2 integration checkpoint

## Authoritative workspace

Work only in `/Users/mert/llmspeed-work/Helix-Harness`. Do not use the old `unsloth-studio-v1.1` checkout. Baseline commit is `174e429`; the adaptive-cycle hardening commit before this v2 pass is `158d6bf`.

Do not reset/clean a working tree to make tests pass. Do not commit generated `studio/src-tauri/artifacts/`, `studio/node_modules`, runtime databases/logs, or temporary smoke state.

## Current runtime identity

The rebuilt macOS app and the installed managed runtime use the same dependency-closed backend overlay:

```text
contract: helix.adaptive.backend.v1
files:    640
sha256:   09b8dd171c857bea0184b22ef663cba21655101b6fc44e53ccdcd8009e787f28
```

The source app is at `studio/src-tauri/target/release/bundle/macos/Helix Harness.app`. It is ad-hoc signed and passes `codesign --verify --deep --strict`; it is not notarized.

Managed runtime is Unsloth `2026.9.5`. Applying the rebuilt app's bundled overlay reports `mem0=ready` and restores the Helix contract after upstream overwrite.

## Core features now closed

### Search / tool capture / evidence

- `require_tool_access` is a real policy gate.
- Same-version upstream overwrite invalidates the managed capability fingerprint.
- Real Qwen3.8-27B production runs called both `search_memory` and `search_conversation`; fresh logs contained no `require_tool_access` ImportError.
- Tool capture preserves return/exception/cancellation semantics.
- Verifier authority is backend-owned typed metadata, never command/tool strings.
- Passed verifier evidence must be exactly claim-bound.
- Unsupported `>=5x` acceptance claim was `UNVERIFIED` and could not trigger QLoRA.

### Hermes / adaptive cycle

- Same-model observable-artifact audit, Q/E/C/S quality, cache/context attribution, shadow decisions, Hermes routing, recurrence ledger, verified training-target receipts, and autonomous safeguards are wired.
- Self-audit cannot directly authorize training.
- One event cannot trigger QLoRA.
- Frontend self-audit JSON/index/ref validation and legacy QLoRA bypass defects are fixed.

### Mem0

- Managed runtime has `mem0ai 2.0.20` and local Qdrant.
- Mem0 2.x filters/result format are supported; old list format remains compatible.
- Real account-scoped local vector write/search succeeded using local MiniLM; smoke record was deleted afterward.
- Vector/graph copies of identical memory are deduplicated on recall.
- Installer/update overlay best-effort installs `mem0ai>=2.0.20,<3.0` with a 180-second bound. Failure leaves the bounded local graph active and does not break Studio.
- Existing Qdrant collection supports semantic vector search; do not claim optional BM25/hybrid support for that old collection schema.

### QLoRA / hot-swap / self-evolution state machine

- GGUF serving identity is separated from trainable source identity. `unsloth/Qwen3.8-27B-GGUF` maps narrowly to `unsloth/Qwen3.8-27B`; arbitrary local GGUF files are never guessed.
- Legacy 27B self-training state was migrated in place and existing example provenance preserved.
- Autonomous immutable dataset snapshots now live under the managed dataset root; the former `learning/self_qlora/runs` location was a real production-admission bug.
- Native MLX now supports resident-base `load_adapter`, named active-adapter validation, and reversible `revert_to_base_model` without checkpoint reload.
- Real Qwen2.5-0.5B production QLoRA produced an adapter; fresh-process hot-swap changed the calibration output, and revert restored the original base output exactly.
- Real self-training held-out benchmark rejected that synthetic adapter because it scored worse and was slower.
- Real autonomous controller smoke completed the entire sequence without sidebar intervention:
  `queued -> training -> 32-step QLoRA -> candidate-ready -> benchmark -> needs-more-data`.
- Manual/ask policy stops at `candidate-ready`; autonomous benchmark requires current `decisionMode=autonomous` and `allowQloraTraining=true`.
- Interrupted benchmark recovers to `candidate-ready` and re-enters only under current policy.
- A deterministic promotion-path test proves the candidate is reattached only after held-out score improvement and measured throughput preservation both pass.
- A promoted MLX adapter explicitly names the trainable source as its serving model. Helix does not falsely claim a PEFT/MLX adapter is active inside a GGUF llama.cpp process.

## Current checks

- Modified Python surfaces: Ruff PASS.
- Combined Helix/Hermes/Mem0/self-training backend suite: **96 passed**.
- MLX adapter-control selection: **11 passed** (`202 deselected`).
- Inference orchestrator lifecycle/cancellation: **100 passed**.
- Frontend Hermes tests: **9 passed**.
- Frontend typecheck: PASS.
- macOS installer/resource packaging: **5 passed**.
- `git diff --check`: PASS.
- `bash -n install.sh`: PASS.
- Final source app strict codesign: PASS.

Full v2 acceptance details are in `docs/helix-harness-v2-validation-20260916.md`.

## Remaining evidence boundaries

1. Native Tauri GUI updater orchestration has not yet been driven through another destructive live update after the v2 changes. Component overwrite/reapply behavior is proven; do not upgrade that to strongest-form GUI updater proof without running it.
2. Qwen3.8-27B itself has not been QLoRA-trained and evaluated before/after on held-out tasks in this v2 pass. Search/tool acceptance is real; 27B weight improvement is not demonstrated.
3. The real 0.5B adapter experiment proves training, saved-weight effect, hot-swap, revert, and autonomous benchmark mechanics. It is a synthetic memorization/mechanism test, not evidence of general intelligence improvement.
4. Existing Mem0 collection does not have the optional v3 BM25 sparse slot; semantic vector memory is proven.

## Prohibitions

- no reset/clean;
- no obsolete 25-file overlay;
- no full backend copy including requirements/tests;
- no package-version-only compatibility proof;
- no hidden chain-of-thought capture;
- no self-audit-as-training-truth;
- no one-event QLoRA;
- no verifier authority from strings/names;
- no global GGUF speculative counters represented as request-scoped proof;
- no unsupported performance claim;
- no claim of 27B/general model improvement without held-out before/after evidence.
