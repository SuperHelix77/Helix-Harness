# Helix Harness Restart Handoff

Updated: 2026-09-17 — adaptive-intelligence completion pass

## Authoritative workspace

The authoritative published source is private GitHub `SuperHelix77/Helix-Harness`, branch `main`. The 2026-09-17 adaptive-cycle completion was developed and validated in `/Users/mert/Desktop/Helix-Harness-spec-completion`. The older `/Users/mert/llmspeed-work/Helix-Harness` checkout was not destructively reset or cleaned; synchronize it later only with an ordinary safe fast-forward/pull if its local state permits.

The completion clone started from the post-audit `main` around `1ab8da6` (`docs: audit Helix adaptive cycle conformance`). The historical audit remains in `docs/helix-adaptive-cycle-conformance-20260917.md`; the completion evidence is `docs/helix-adaptive-cycle-acceptance-20260917.md`.

Do not reset/clean a working tree to make tests pass. Do not commit generated `studio/src-tauri/artifacts/`, `studio/node_modules`, dependency symlinks, `.helix-work/`, runtime databases/logs, bytecode/cache directories, authentication files, or temporary acceptance/smoke state.

## Packaged runtime identity boundary

The following overlay identity is the **historical packaged baseline from before this source-completion pass**:

```text
contract: helix.adaptive.backend.v1
files:    640
sha256:   09b8dd171c857bea0184b22ef663cba21655101b6fc44e53ccdcd8009e787f28
```

The existing desktop app is `/Users/mert/Desktop/Helix Harness v2.app` (`ai.helix.harness.v2`). It is ad-hoc signed and not notarized.

Managed runtime is Unsloth `2026.9.5`; Mem0 is `2.0.20`. This completion pass validated repository source using the managed Python environment but **did not rebuild/re-fingerprint the desktop app or managed overlay from the completion source**. Do not claim installed-app parity until a later packaging pass actually stages and validates these new backend/frontend files.

## Core features now closed

### Search / tool capture / evidence

- `require_tool_access` is a real policy gate.
- Same-version upstream overwrite invalidates the managed capability fingerprint.
- The final completion acceptance uses Qwythos-9B and real Mem0 retrieval through both `search_memory` and `search_conversation`; the controlled performance claim remains `UNVERIFIED`.
- Tool capture preserves return/exception/cancellation semantics.
- Verifier authority is backend-owned typed metadata, never command/tool strings.
- Passed verifier evidence must be exactly claim-bound.
- Unsupported `>=5x` acceptance claim was `UNVERIFIED` and could not trigger QLoRA.

### Hermes / adaptive cycle

- Same-model observable-artifact audit, Q/E/C/S quality, cache/context attribution, typed decisions, Hermes routing, recurrence ledger, verified training-target receipts, and autonomous safeguards are wired.
- The production chat path now records bounded hash/counter-only context provenance for system-prompt/tool-catalog changes, insertion/reinsertion and reorder observations; runtime-unobservable values remain explicitly unavailable rather than fabricated.
- `DEEP_SELF_AUDIT` is decided before generative audit work. Failed tools, explicit acceptance criteria, high-impact claims, and typed verification evidence can force the deep audit. Any controller/preparation failure preserves the old behavior and runs it.
- The same-model audit consumes the backend-resolved `helix.audit-input.v1` artifact bundle: objective/context, tool calls/errors, edits, typed tests/benchmarks, cache telemetry, evidence, result and acceptance criteria. Hidden reasoning is excluded.
- `helix.trajectory.v1` is the versioned observable trajectory wire envelope; internal execution retains the established dataclass.
- `helix.counterfactual.v1` is a first-class provenance-linked compression candidate. Shorter is not equivalent: candidates remain `UNVERIFIED`/training-ineligible until a future backend-owned replay/equivalence receipt exists.
- Objective retrospective decision labels are persisted and Brier/reliability bins are measured. The controller deliberately remains `calibrated=false` until a held-out calibration study provides sufficient evidence.
- The broader typed decision family remains supported, but only `DEEP_SELF_AUDIT` is currently a live pre-generative probabilistic allocation point; existing deterministic routing/safety behavior remains authoritative elsewhere.
- Backend claim discovery adds obvious high-impact final-answer assertions to evidence before self-audit even when the client supplied no `claims` array.
- `X-Helix-Background-Audit: 1` is a backend-owned no-tools boundary. It defeats a launcher-level `--enable-tools` override for the internal audit without changing ordinary chat tool policy.
- Qwythos reproducibly omitted only the closing `</helix-self-audit>` tag while returning one complete JSON object. The parser now accepts only that bounded shape; partial JSON or trailing prose remains invalid.
- Objective alias-equivalent duplicate retrieval now feeds cache attribution, computational-efficiency metrics, decision features and counterfactual credit even when both captured tool hints say `useful`.
- Hermes records `self_assessment_disagreement=true` when its evidence/cache-driven adaptation differs from the model's advisory recommendation.
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

- Final adaptive/ingest/pipeline/engine/routes/self-training/UI-policy backend suite: **134 passed**.
- Final focused frontend Hermes/context/Engine/timeout/route suite: **28 passed**.
- Frontend TypeScript typecheck: **PASS**.
- Frontend production build: **PASS**; only the established `::highlight`, dynamic-import and chunk-size warnings remain.
- Ruff across **20** modified/untracked Python files: **PASS**.
- `git diff --check`: **PASS**.

These are the final source-completion checks. Historical v2 packaging/MLX/inference checks remain documented in `docs/helix-harness-v2-validation-20260916.md`; they were not rerun merely to imply that the older installed `.app` contains this completion source.

Full historical v2 acceptance details are in `docs/helix-harness-v2-validation-20260916.md`.

The original Jev/Empryo/self-audit → Hermes plan was re-audited on 2026-09-17 and its concrete gaps were then implemented. See `docs/helix-adaptive-cycle-conformance-20260917.md` for the historical gap audit and `docs/helix-adaptive-cycle-acceptance-20260917.md` for the completion acceptance/overhead evidence. Decision calibration has begun and is measured, but is not statistically mature and is not described as calibrated. Counterfactual equivalence verification remains intentionally absent rather than being guessed from self-report.

## Remaining evidence boundaries

1. Native Tauri GUI updater orchestration has not yet been driven through another destructive live update after the v2 changes. Component overwrite/reapply behavior is proven; do not upgrade that to strongest-form GUI updater proof without running it.
2. Neither the final Qwythos-9B acceptance model nor Qwen3.8-27B has been shown to improve through a held-out before/after QLoRA evaluation in this completion pass. Search/tool/audit/Hermes acceptance is real; weight/intelligence improvement is not demonstrated.
3. The real 0.5B adapter experiment proves training, saved-weight effect, hot-swap, revert, and autonomous benchmark mechanics. It is a synthetic memorization/mechanism test, not evidence of general intelligence improvement.
4. Existing Mem0 collection does not have the optional v3 BM25 sparse slot; semantic vector memory is proven.
5. Counterfactual trajectory compression now has a versioned candidate/provenance record, but no backend replay/equivalence verifier exists yet. Therefore these candidates cannot become efficiency-training pairs.
6. Decision-controller probabilities now have objective retrospective labels/Brier bins, but the sample is not a held-out calibration study. `calibrated=false` is the correct current status.
7. GGUF request-scoped speculative accepted/rejected draft counters remain unavailable and are represented as unavailable, not as observed zero measurements.
8. The existing installed desktop app/managed overlay was not rebuilt from this source-completion pass.

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
- no claim of Qwythos/27B/general model improvement without held-out before/after evidence;
- no claim of packaged desktop parity until the app/overlay is rebuilt and revalidated.
