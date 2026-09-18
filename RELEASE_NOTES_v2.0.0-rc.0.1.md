# Helix Harness v2 RC0.1

Tag: `v2.0.0-rc.0.1`

This is the first public release-candidate label for the Helix Harness v2 architecture.

## Upstream credit

Helix Harness is a derivative of **Unsloth Studio** and retains the AGPL-3.0 license obligations and upstream notices. Unsloth provides the core model-training/inference and desktop/runtime foundation. Helix Harness adds the Helix-specific agentic, memory, learning-control, observability, reliability, branding, and packaging layers described below.

## What changed from the imported Unsloth Studio baseline

### Agentic loop

- Shared tool-loop controller across GGUF, safetensors, and external-provider/Codex paths.
- Proven-equivalent read-only alias suppression before execution, with explicit user-intent exemptions.
- Bounded repeated failures with retries preserved and workspace-change invalidation.
- Post-edit verification obligations and verification-attempt telemetry.
- Objective per-loop progress counters and provenance.
- Separate runtime `helix.tool-control.v1` events for prevented actions.
- Exact exception, cancellation, and real tool-result semantics preserved.

### Helix adaptive intelligence cycle

- Versioned trajectory capture with retry/control-event provenance.
- Cache/context efficiency observation.
- Typed quality/evidence pipeline.
- Post-task self-audit artifacts routed into Hermes.
- Advisory decision-control features and recurrence tracking.
- QLoRA/live-tuning eligibility remains gated by recurrence, corroborated evidence, and independently verified corrected targets.

### Runtime and compatibility fixes

- Safetensors optional Helix metadata is feature-detected so injected executors remain compatible.
- External-provider/Codex Helix turn-id wiring corrected.
- Prevented external tool calls terminate their UI cards truthfully.
- Qwen3.8 text checkpoints that the Unsloth MLX loader must run through `mlx-vlm` now use the matching VLM cache/generation semantics while remaining text-only at the product capability layer.

### Desktop product

- Helix Engine UI restored and integrated.
- Helix branding and main-window transparency controls.
- Embedded backend overlay with source/package/runtime parity checks.
- macOS Helix Harness v2 bundle identity: `ai.helix.harness.v2`.

## Release boundaries

- This RC does not claim held-out model intelligence improvement.
- Self-audit is evidence generation, not autonomous authority to mutate weights.
- The probabilistic decision controller is not claimed calibrated.
- Speculative decoding is available but no universal throughput multiplier is claimed; a local experiment that preserved output but regressed throughput was deliberately excluded from this RC.
- macOS signing is ad-hoc unless a release asset explicitly states otherwise; notarization is not claimed.

## Validation

The v2 adaptive-cycle and foreground-agent implementations have dedicated backend/frontend regression suites documented under `docs/`. RC0.1 additionally includes a regression for Qwen3.8 text-only checkpoints forced through the `mlx-vlm` runtime.
