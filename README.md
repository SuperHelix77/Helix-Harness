# Helix Harness

Helix Harness is a chat-first GUI harness for local models. It is built on **Unsloth Studio** (AGPL-3.0). See [CREDITS.md](CREDITS.md).

Frozen baseline: Unsloth v1.1 at `baselines/unsloth-studio-v1.1`.

## What this app is

- Local chat + tools + memory + self-learning
- Qwen3.8-27B v1.1 Mac load profile (64K, q4_0 KV, unified KV, 1 slot, batch 1024/256)
- DFlash/GDN 5x **target** with fail-open to ordinary v1.1 decode (never claim 5x with zero accepted drafts)
- Computer use and web browse with a live feed
- Helix Engine baked in: trajectories → credit → compression → Hermes/QLoRA routing. Success never auto-authorizes weight updates.

## What this app is not

- No Train tab
- No Video/Image tab
- Helix Engine is not a separate webapp

## Mac app

```bash
cd studio
npx tauri build --bundles app
```

The signed-or-ad-hoc bundle lands at `studio/src-tauri/target/release/bundle/macos/Helix Harness.app`. Copy it to `/Applications`.

## Run (dev API)

```bash
cd studio/backend
PYTHONPATH=. python -m uvicorn main:app --host 127.0.0.1 --port 8888
```

Frontend: `studio/frontend` (`npm run build` / `npm run dev`).
