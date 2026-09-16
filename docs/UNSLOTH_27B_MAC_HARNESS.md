# Unsloth Studio: 27B Mac harness

This fork turns Studio into a local, Codex-shaped harness for long-context inference, governed learning, and model optimization. Existing Unsloth training and inference paths remain available; the new paths are additive and fall back when a model or backend cannot support them.

## Runtime profile

- Qwen3.8-27B on macOS gets the V1.1 profile when settings are automatic: 64K context, unified KV, KV `q4_0`, single-slot decode, larger prefill batches, and the target-matched `z-lab/Qwen3.8-27B-DFlash2` drafter.
- MLX Qwythos-9B-v2 gets a durable 64K default when no explicit context is supplied. Its compatible automatic draft path is the published Qwen3.5-9B DFlash checkpoint. DFlare is not guessed for Qwythos because there is no target-specific published checkpoint in this repository.
- Explicit user settings win. If a sidecar cannot load, the app keeps ordinary decoding usable and reports the fallback; it does not pretend that a 5x result was achieved.
- DFlash/DFlare are lossless speculative-decoding paths when accepted tokens are verified by the target. The observed speedup depends on hardware, quantization, prompt length, acceptance rate, and the drafter/target pair.

The 64K value is applied at load resolution and in the MLX speculative cache path, so enabling DFlash does not restore the old 8K cache default on the next turn.

## Learning ladder

The self-learning flow follows this order:

`experience → Mem0 retrieval → repeated pattern → Hermes skill → proven model deficiency → QLoRA`

Completed local text turns are recorded in a bounded task ledger. Mem0 retrieval is supplementary and is used to find recurrence. A procedural repetition becomes a plain-text Hermes skill before expensive weight updates are recommended. QLoRA is considered only for a repeated, measurable behavior gap that a skill or runtime repair cannot solve.

The self-QLoRA acceptance gate is external to the model:

- a fixed held-out benchmark scores base and candidate with deterministic checks;
- candidate intelligence must improve by at least 0.01;
- candidate throughput must be measured and must not regress;
- the original base remains available for comparison and rollback;
- promotion hotswaps only a validated local PEFT adapter;
- failed candidates stay inactive and can be reverted/retrained.

The model's own score or critique is never sufficient evidence. The sidebar exposes Ask me versus Autonomous decision mode plus independent permissions for skill creation, QLoRA training, runtime repair, and Mem0. These controls do not interrupt normal chat.

On-the-fly skill drafts are emitted as a non-visible marker during completed tasks. The app stages them only after Mem0 finds a prior similar experience, and approval writes only a plain `SKILL.md`; no generated executable code is installed.

## Local Mem0

Mem0 is optional at import time and configured for account-scoped local storage:

- Qdrant files and history live under the account's `learning/mem0` directory;
- the vector-store user id is a one-way account hash, not an email or account subject;
- embeddings use a local Hugging Face model;
- the Mem0 LLM points to Studio's local OpenAI-compatible endpoint by default;
- Mem0 telemetry and Hugging Face telemetry are disabled by the adapter;
- if Mem0 is unavailable, the bounded JSON ledger continues to work.

Install the optional dependency in a compatible environment with:

```bash
pip install 'mem0ai>=0.1.118,<1.0'
```

The shipped runtime may already contain the package. The application does not silently downgrade its existing protobuf stack to satisfy an optional memory package; the `/api/memory` status endpoint reports availability and the local ledger remains the fallback.

## Harness surfaces

- Slash commands discover `/github`, `/plan`, `/goal`, `/speed`, `/learn`, installed Codex/Claude `SKILL.md` files, and dynamic `/skill-name` commands. `/github` uses the configured GitHub connector context instead of asking for a URL when repository context is available.
- The chat workspace has Outputs, Background, Sources, and Subagents tabs.
- Computer use is available through the Code tool toggle on macOS. Screenshot is read-only; click, type, key, scroll, and open-app actions require approval and use bounded native macOS APIs. Enable Accessibility for Unsloth Studio in System Settings → Privacy & Security → Accessibility.
- The local `unsloth-local-codex` plugin provides the read-only GitHub MCP connection for the Codex app.
