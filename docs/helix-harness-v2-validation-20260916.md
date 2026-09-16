# Helix Harness v2 — Integration and Self-Evolution Validation

Date: 2026-09-16

## Scope

This checkpoint validates the integrated Helix Harness feature set before producing the separate Desktop clone `Helix Harness v2.app`. It distinguishes runtime/mechanism proof from model-quality proof. A training job completing is not evidence that a model improved; promotion remains downstream of deterministic held-out scoring and measured throughput.

## Integrated feature status

### Helix Engine / Hermes

- Versioned adaptive-cycle controller, evidence adjudication, cache/context attribution, same-model observable-artifact self-audit, Hermes routing, recurrence ledger, typed verification receipts, and verified training-target receipts remain active.
- Backend verifier authority is backend-owned and exact-claim-bound; arbitrary tool/command names cannot mint verification authority.
- One anomaly cannot authorize QLoRA. Autonomous QLoRA uses only independently qualified examples and the existing user policy gates.
- Unsupported performance claims remain `UNVERIFIED`; the earlier real Qwen3.8 acceptance deliberately rejected an unsupported `>=5x` claim.

### Mem0 + bounded graph memory

- Managed runtime now contains `mem0ai 2.0.20` and local Qdrant support.
- Mem0 2.x API compatibility is implemented: account selectors use `filters`, and both structured v2 results and legacy list results are accepted.
- Real local write/search was exercised against the account-scoped Qdrant store using the local MiniLM embedder. The vector record itself was retrieved; the test record was then deleted from both Qdrant and the bounded graph.
- Vector and graph copies with identical memory text are deduplicated so one experience does not consume two retrieval slots.
- Mem0 remains optional/fail-open. Installer/update overlay validates the runtime and may install `mem0ai>=2.0.20,<3.0`; failure or a 180-second timeout leaves the bounded local graph active rather than breaking Studio.
- The existing Qdrant collection is semantic-search capable. Its old schema does not contain Mem0 v3's optional BM25 sparse slot; this is not represented as hybrid/BM25 support.

### QLoRA source identity

- A GGUF serving checkpoint is no longer treated as a trainable PEFT baseline.
- Canonical `*-GGUF` Hub identities and real Hugging Face cache GGUF paths map narrowly to their trainable source repository. Unknown standalone `.gguf` files are never guessed and remain subject to the training route's explicit GGUF-not-trainable rejection.
- State persists separate `baseModelId` (trainable source) and `baseServingModelId` (for example the GGUF distribution).
- Legacy Qwen3.8 state was migrated from `unsloth/Qwen3.8-27B-GGUF` to trainable source `unsloth/Qwen3.8-27B`, preserving the serving identity and example provenance.
- Immutable per-run training snapshots now live under the managed dataset root. This fixes a production admission bug where autonomous snapshots under `learning/self_qlora/runs` were correctly refused by the training path as an unauthorized dataset location.

### Native Apple-Silicon hot-swap

MLX now supports a real resident-base adapter lifecycle:

1. Validate a saved MLX LoRA artifact.
2. Under the generation lock, clear prompt cache and unwrap any previous LoRA wrappers.
3. Attach the new adapter with `mlx_lm.tuner.utils.load_adapters` without reloading/copying the base checkpoint.
4. Select only the actually attached adapter by name.
5. Revert by replacing LoRA wrappers with their retained base modules, again without checkpoint reload.
6. On attach failure, attempt the same unwrap rollback before returning failure.

The generic inference worker/orchestrator therefore has working `load_adapter` / `set_active_adapter` / `revert_to_base_model` semantics on the native MLX backend, not only Transformers/PEFT.

## Real QLoRA + hot-swap experiment

A complete production training smoke was run using the already cached full source checkpoint `Qwen/Qwen2.5-0.5B-Instruct` so the mechanism could be tested cheaply without pretending to validate the 27B model.

Training configuration:

- Studio production training route and subprocess, not a mocked trainer.
- MLX LoRA/QLoRA, runtime 4-bit base.
- LoRA rank 8, alpha 16.
- Completion-only synthetic calibration examples.
- 12-step first artifact smoke; a later autonomous-cycle run used the controller's actual 32-step configuration.

Observed 12-step run:

- Trainable parameters: 4,399,104.
- Loss: approximately 5.01 -> 0.00057 on the synthetic calibration set.
- Peak reported training memory: approximately 0.75 GB.
- Real `adapter_config.json` and `adapters.safetensors` were written by Studio.

Fresh-process inference acceptance:

- Base output for the calibration prompt: `The Helix calibration codename is "H1".`
- After production orchestrator hot-swap: `CERULEAN-PINE-483`.
- After resident-base revert: the exact original `H1` response returned.
- The model then unloaded and the inference subprocess exited cleanly.

This demonstrates a real learned-weight behavioral effect plus reversible hot-swap. It is deliberately a memorization/mechanism test, not evidence of general intelligence improvement.

## Objective promotion gate

The real `/self-training/benchmark` controller was run against the synthetic adapter:

- Base held-out score: 0.50.
- Candidate held-out score: 0.25.
- Base measured throughput: approximately 380 tok/s (runs varied around 380-386 tok/s).
- Candidate measured throughput: approximately 169-172 tok/s.
- `speedMeasured = true`.
- `intelligenceImproved = false`.
- `speedPreserved = false`.
- `promoted = false`.
- Candidate was reverted and the source inference model unloaded.

A deterministic promotion-path unit test separately proves that a candidate is activated a second time only after both score improvement and measured throughput preservation pass.

## Autonomous end-to-end cycle

A real autonomous controller smoke was run with four independently marked fixture examples in an isolated in-memory self-training state while using the production trainer/inference backends:

`queued -> training -> real 32-step MLX QLoRA -> candidate-ready -> benchmark-queued -> benchmark-loading-base -> benchmarking -> needs-more-data`

Observed:

- Production training job ID was captured.
- Adapter artifact was discovered automatically after completion.
- No sidebar/manual candidate-path action was required.
- Trainable source was loaded automatically for evaluation after training had freed inference memory.
- Candidate/base holdouts and real backend throughput were measured.
- Inferior candidate was rejected, reverted, and the evaluation-only base model was unloaded.
- Total end-to-end smoke wall time was about 21 seconds on this machine.

Manual/`ask` mode stops at `candidate-ready`. Automatic benchmark/promotion requires both `decisionMode=autonomous` and `allowQloraTraining=true`. Interrupted benchmark state recovers to the durable `candidate-ready` boundary and requeues only under the current policy.

For a promoted MLX adapter, state explicitly records the trainable source as the adapted serving model. Helix does **not** pretend that a PEFT/MLX adapter has been applied to a GGUF llama.cpp process. A rejected candidate leaves no hidden source-model replacement; normal chat may reload the user's selected GGUF on the next use.

## Current regression evidence

Current-tree checks after the v2 integration fixes:

- Ruff on modified Python/runtime/test surfaces: PASS.
- Helix/Hermes/Mem0/self-training combined backend suite: **96 passed**.
- MLX adapter-control selection: **11 passed** (`202 deselected`).
- Inference orchestrator lifecycle/unload/cancellation: **100 passed**.
- macOS installer/resource contract: **5 passed**.
- Frontend Hermes tests: **9 passed**.
- Frontend TypeScript typecheck: PASS.
- `git diff --check`: PASS.
- `bash -n install.sh`: PASS.

## Evidence grades

- IMPLEMENTED: **YES**.
- TESTED: **YES** for the scoped integrated features above.
- REAL MEM0 VECTOR WRITE/SEARCH: **YES**.
- REAL QLORA TRAINING ARTIFACT: **YES**.
- REAL RESIDENT MLX HOT-SWAP + REVERT: **YES**.
- REAL AUTONOMOUS TRAIN -> ARTIFACT -> BENCHMARK -> REJECT CYCLE: **YES**.
- OBJECTIVE PROMOTION GATE TESTED: **YES**.
- QWEN3.8-27B SEARCH-TOOL PRODUCTION ACCEPTANCE: **YES** (from the prior acceptance checkpoint).
- QWEN3.8-27B QLORA HELD-OUT IMPROVEMENT DEMONSTRATED: **NO**.
- GENERAL MODEL INTELLIGENCE IMPROVEMENT DEMONSTRATED: **NO**.

The last two remain NO until the real 27B source is trained and a before/after held-out evaluation demonstrates improvement without unacceptable throughput/regression cost.
