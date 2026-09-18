# Helix Harness

**Current release candidate: Helix Harness v2 RC0.1 (`v2.0.0-rc.0.1`)**

Helix Harness is a local-first agentic AI workstation built on **Unsloth Studio**. It keeps Unsloth's model-loading, training, inference, Apple-Silicon, GGUF/MLX, and desktop foundations, then adds a Helix-specific control plane for persistent memory, tool-using chat, agent-loop reliability, governed learning, observability, and desktop operation.

Unsloth Studio is developed by the Unsloth AI Inc. team and is licensed under AGPL-3.0. Helix Harness is a fork/derivative project; it is not presented as an upstream Unsloth release or as an endorsement by Unsloth. See [CREDITS.md](CREDITS.md) and [LICENSE](LICENSE).

## What Helix Harness v2 is now

- A chat-first local model harness for macOS and the existing Unsloth-supported backends.
- A foreground agent loop with local tools, approvals, cancellation, bounded retries, exact exception preservation, and deterministic recovery behavior.
- Persistent memory through Mem0/Qdrant integration plus conversation/repository retrieval.
- Hermes post-task learning control: trajectories, typed evidence, self-audit artifacts, recurrence tracking, and guarded routing to learning workflows.
- Helix Engine adaptive telemetry: tool execution, prevented actions, retries, cache/context observations, quality signals, and versioned trajectory records.
- Advisory Jev-inspired decision control. Probabilistic controller outputs remain shadow/advisory while calibration is immature.
- QLoRA live-tuning/hot-swap infrastructure gated behind corroborated evidence and independently verified training targets.
- MLX, llama.cpp/GGUF, external-provider, Codex-subscription, DFlash/DSpark/MTP, GDN, and prompt-cache paths preserved behind fail-open boundaries.
- A standalone Helix-branded Tauri desktop app with the backend overlay embedded into the bundle.

## What Helix fixed or hardened

Helix Harness v2 includes fixes and hardening on top of the imported Unsloth Studio baseline, including:

- Restored the Helix Engine UI, Helix branding, main-window transparency, and desktop packaging parity.
- Reworked the passive Helix trajectory collector into a closed-loop **measurement and control** cycle without allowing self-audit text to mutate weights directly.
- Added typed quality/evidence handling so raw model prose is never verifier authority.
- Added runtime-observed tool-control events for prevented duplicate/repeated-failure actions without recording fake executions.
- Added semantic deduplication only for proven-equivalent read-only aliases, with an explicit-user-intent exemption.
- Bounded exact repeated tool failures while preserving retries and invalidating stale failure state after workspace-changing edits.
- Added post-edit verification obligations and verification-attempt telemetry without treating a command name as proof of correctness.
- Added objective retry/progress provenance and carried it into Helix/Hermes trajectories.
- Fixed optional-instrumentation compatibility so Helix metadata cannot break injected safetensors executors.
- Fixed external-provider/Codex Helix turn correlation bugs and preserved clean tool-card termination for skipped calls.
- Fixed Qwen3.8 text checkpoints that must run through `mlx-vlm`: Helix now follows the loader's real runtime/cache semantics without falsely advertising image support.
- Preserved exact tool return identity, original exceptions, cancellation semantics, and fail-open behavior for optional learning/telemetry components.

The detailed adaptive-cycle and agent-loop evidence is in `docs/helix-adaptive-cycle*.md`, `docs/helix-agentic-loop-analysis-20260917.md`, and `docs/helix-harness-restart-handoff.md`.

## What v2 does **not** claim

- Self-audit alone is not “self-improvement.”
- QLoRA is not triggered by one bad trajectory or raw self-criticism.
- The Jev-inspired controller is not claimed to be calibrated; authoritative promotion requires evidence.
- DFlash/MTP/speculative decoding is not claimed to produce a universal speedup. Helix keeps fallback paths and treats throughput as something to measure on the active model/hardware.
- No model-intelligence improvement is claimed without held-out before/after evaluation.
- The macOS RC is ad-hoc signed unless a later release explicitly documents Apple notarization.

## Baseline and licensing

The repository preserves an imported/frozen Unsloth Studio baseline under `baselines/unsloth-studio-v1.1`. The upstream groundwork remains credited to Unsloth. Helix-specific source, tests, documentation, and packaging changes are maintained in this fork under the repository's AGPL-3.0 obligations.

## Build the macOS app

```bash
cd studio
npx tauri build --bundles app
```

For the Helix Harness v2 RC bundle, the release workflow overrides the product name and bundle identifier to:

```text
Helix Harness v2
ai.helix.harness.v2
```

## Run the development API

```bash
cd studio/backend
PYTHONPATH=. python -m uvicorn main:app --host 127.0.0.1 --port 8888
```

Frontend: `studio/frontend` (`npm run build` / `npm run dev`).
