---
name: unsloth-local-codex
description: Use the local Unsloth Studio repository as a Codex-like coding, planning, and inference workspace with durable Goal/Plan mode, GitHub ledger context, long-context Mac safeguards, and evidence-based speculative-decoding changes.
---

# Unsloth Local Codex

Use this skill when working on the local Unsloth Studio app or when the user asks to turn it into a local Codex-style environment.

## Operating rules

- Keep Goal mode for the concrete outcome and Plan mode for the durable execution plan. Preserve both when reopening or switching chat threads.
- Inspect the relevant GitHub ledger before applying a ledger-derived optimization. Treat benchmark claims as evidence with model, hardware, context, and runtime scope; do not promote a paper's best-case speedup to a universal promise.
- Keep explicit user settings authoritative. Automatic policies may choose safe defaults, but must never silently overwrite an explicit context length, cache type, slot count, speculative mode, or placement choice.
- For Mac Qwen3.8 GGUF, the V1.1 target is 64K context with conservative unified-memory settings. A fit-reduced window is runtime outcome, not user intent, and must not become a sticky request on the next reload.
- MTP and DFlash are experimental unless the exact target/drafter pair and the installed llama.cpp capability probe agree. If a drafter fails, preserve target availability through ordinary-decoding fallback and record the requested mode for a later retry.
- Qwythos-9B-v2 on MLX uses the published Qwen3.5-9B DFlash drafter when the target is compatible. Keep the app's durable 64K context policy, cap quantized verification at five tokens, and do not label this pairing DFlare: no target-specific Qwythos DFlare checkpoint is published.
- On the supported Qwen3.8-27B Mac profile, the optimal automatic speculative choice is the target-matched DFlash/DFlash2 path available to the selected backend. Do not force DFlare on GGUF or call an MLX DFlash2 run DFlare; unsupported pairs must fall back clearly.

## GitHub context

In the desktop app, the visible Skills menu and these local commands are available:

- `/github [query]` resolves the authenticated GitHub account and local Git remotes, then reads relevant repository or ledger files without requiring a URL.
- `/plan [objective]` selects durable Plan mode.
- `/goal [objective]` selects durable Goal mode.
- `/speed` reports measured tok/s, context length, and whether speculative decoding is actually engaged.
- `/skills` shows the command menu.

The Skills menu also opens a local manager. It can create a portable `SKILL.md` or install a read-only GitHub skill repository into `~/.codex/skills`, `~/.claude/skills`, or both. Installed skill names appear in the `/` picker and `/skill-name request` injects the bounded `SKILL.md` instructions into the local request. Skill files are data for the local runtime; never execute repository scripts during installation.

Use the bundled GitHub MCP server for repository, issue, PR, and ledger inspection when running in Codex. OAuth is handled by the MCP host; do not copy tokens into the plugin, repository, prompts, or logs. Prefer primary GitHub source and preserve commit or file provenance in implementation notes.

For performance reports, distinguish ordinary decode from speculative decoding. A speed multiplier is only evidence-backed when the runtime reports an active speculative slot and non-zero accepted draft tokens.

## Delivery checklist

1. Read the applicable ledger and identify what is implemented, deferred, or merely proposed.
2. Add a small pure policy seam and regression test before wiring it through load, reload, and thread-restoration paths.
3. Verify backend and frontend tests, formatting/type checks, and `git diff --check`.
4. Build the Mac app before claiming the installed app changed. Report physical unified-memory limits separately from software policy limits.
