// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";

import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

type FeedEvent = {
  action?: string;
  url?: string;
  kind?: string;
  title?: string;
  snippet?: string;
};

type SessionStep = {
  name: string;
  arguments: string;
  result: string;
  useful_hint: string;
};

type AnalyzeResult = {
  gates?: string[];
  credit?: number[];
  compressed?: { tool_calls?: number; final_decision?: string };
  route?: string;
  decision?: string;
  success_authorizes_weight_update?: boolean;
  corrections?: Array<{ bad_action?: string; correct_action?: string; why?: string }>;
};

export function HelixEnginePage() {
  const threadId = useChatRuntimeStore((state) => state.activeThreadId);
  const projectId = useChatRuntimeStore((state) => state.activeProjectId);
  const sessionId =
    sandboxSessionIdFor(threadId ?? undefined, projectId) ?? threadId ?? "default";
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [steps, setSteps] = useState<SessionStep[]>([]);
  const [turns, setTurns] = useState("");
  const [turnSignal, setTurnSignal] = useState<{
    unnecessary_turns?: number;
    feed_self_improvement?: boolean;
  } | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Helix Engine";
  }, []);

  useEffect(() => {
    let live = true;
    const load = () => {
      void authFetch(`/api/helix-engine/live-feed?session_id=${encodeURIComponent(sessionId)}`)
        .then((response) => response.json())
        .then((body: { events?: FeedEvent[] }) => {
          if (live) setEvents(body.events ?? []);
        })
        .catch(() => undefined);
      void authFetch(`/api/helix-engine/session/${encodeURIComponent(sessionId)}`)
        .then((response) => response.json())
        .then((body: { steps?: SessionStep[] }) => {
          if (live) setSteps(body.steps ?? []);
        })
        .catch(() => undefined);
    };
    load();
    const timer = window.setInterval(load, 1_000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  async function analyzeSession() {
    setError(null);
    const payload = {
      prompt_state: `session ${sessionId}`,
      steps: steps.map((step) => ({
        name: step.name,
        arguments: step.arguments,
        result: step.result,
        useful_hint: step.useful_hint,
      })),
      verified: true,
      final_result: steps.at(-1)?.result ?? "",
    };
    const response = await authFetch("/api/helix-engine/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = (await response.json()) as AnalyzeResult & { detail?: string };
    if (!response.ok) {
      setError(body.detail ?? "Analyze failed");
      return;
    }
    setAnalysis(body);
  }

  async function checkTurns() {
    setError(null);
    const response = await authFetch("/api/helix-engine/semantic-turns", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        turns: turns
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean),
      }),
    });
    const body = (await response.json()) as {
      unnecessary_turns?: number;
      feed_self_improvement?: boolean;
      detail?: string;
    };
    if (!response.ok) {
      setError(body.detail ?? "Semantic-turn check failed");
      return;
    }
    setTurnSignal(body);
  }

  return (
    <div className="mx-auto flex h-full max-w-4xl flex-col gap-5 overflow-y-auto p-6" data-testid="helix-engine-page">
      <header>
        <h1 className="font-heading text-xl font-semibold tracking-wide">HELIX ENGINE</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          In-app learning control plane. Trajectories, credit, compression, and routing — not a separate webapp.
          Success never auto-authorizes weight updates.
        </p>
        <p className="mt-1 text-xs text-muted-foreground">Session {sessionId}</p>
      </header>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      <section className="rounded-xl border border-border p-4">
        <h2 className="text-sm font-semibold">Live computer / browse feed</h2>
        {events.length ? (
          events.slice(-12).map((event, index) => (
            <div key={`${event.action}-${index}`} className="mt-2 rounded-lg border border-border/70 p-2 text-sm">
              <p className="font-medium">
                {event.kind || "computer"} · {event.action}
              </p>
              {event.title ? <p className="text-xs">{event.title}</p> : null}
              {event.url ? <p className="break-all text-xs text-muted-foreground">{event.url}</p> : null}
            </div>
          ))
        ) : (
          <p className="mt-2 text-xs text-muted-foreground">No browse or computer-use events yet for this session.</p>
        )}
      </section>

      <section className="rounded-xl border border-border p-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Trajectory</h2>
          <button
            type="button"
            className="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground"
            onClick={() => void analyzeSession()}
          >
            Analyze
          </button>
        </div>
        {steps.length ? (
          <ol className="mt-2 space-y-1 text-xs">
            {steps.map((step, index) => (
              <li key={`${step.name}-${index}`} className="rounded-md border border-border/60 p-2">
                <span className="font-medium">{step.name}</span>
                <span className="text-muted-foreground"> · {step.useful_hint || "useful"}</span>
                <p className="mt-1 truncate text-muted-foreground">{step.arguments}</p>
              </li>
            ))}
          </ol>
        ) : (
          <p className="mt-2 text-xs text-muted-foreground">Tool steps appear here as the model works.</p>
        )}
        {analysis ? (
          <div className="mt-3 space-y-1 text-xs">
            <p>Route: {analysis.route} · Decision: {analysis.decision}</p>
            <p>Success authorizes weights: {String(analysis.success_authorizes_weight_update)}</p>
            <p>Kept steps: {(analysis.credit ?? []).join(", ") || "none"}</p>
            <p>Gates: {(analysis.gates ?? []).join(" → ")}</p>
          </div>
        ) : null}
      </section>

      <section className="rounded-xl border border-border p-4">
        <h2 className="text-sm font-semibold">Unnecessary semantic turns</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Paste ChatGPT/Codex-style filler turns, one per line. Helix Engine feeds this to self-improvement.
        </p>
        <textarea
          className="mt-2 min-h-24 w-full rounded-md border border-border bg-background p-2 text-sm"
          value={turns}
          onChange={(event) => setTurns(event.target.value)}
          placeholder={"ok\nsure\nlet me think"}
        />
        <button
          type="button"
          className="mt-2 rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground"
          onClick={() => void checkTurns()}
        >
          Monitor
        </button>
        {turnSignal ? (
          <p className="mt-2 text-xs">
            Unnecessary: {turnSignal.unnecessary_turns} · Feed self-improvement:{" "}
            {String(turnSignal.feed_self_improvement)}
          </p>
        ) : null}
      </section>
    </div>
  );
}
