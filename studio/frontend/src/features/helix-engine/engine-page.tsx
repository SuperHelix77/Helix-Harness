// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";

import { authFetch } from "@/features/auth";
import { getStoredChatThreadReadResult, useChatRuntimeStore } from "@/features/chat";

import {
  engineProjectIdForThread,
  engineSessionIdFor,
  fetchHelixJson,
  helixErrorMessage,
  loadHelixSnapshot,
  type FeedEvent,
  type SessionStep,
} from "./engine-data";

type AnalyzeResult = {
  gates?: string[];
  credit?: number[];
  compressed?: { tool_calls?: number; final_decision?: string };
  route?: string;
  decision?: string;
  success_authorizes_weight_update?: boolean;
  corrections?: Array<{ bad_action?: string; correct_action?: string; why?: string }>;
};

type SessionBinding = {
  threadId: string;
  sessionId: string;
  scope: string;
  warning: string | null;
};

type PollState = {
  key: string;
  events: FeedEvent[];
  steps: SessionStep[];
  status: "connected" | "degraded";
  error: string | null;
  lastUpdatedAt: number | null;
};

type AnalysisState = {
  key: string;
  value: AnalyzeResult;
};

export function HelixEnginePage() {
  const threadId = useChatRuntimeStore((state) => state.activeThreadId);
  const activeProjectId = useChatRuntimeStore((state) => state.activeProjectId);
  const fallbackSessionId = engineSessionIdFor(threadId, activeProjectId);
  const [sessionBinding, setSessionBinding] = useState<SessionBinding | null>(null);
  const resolvedBinding =
    threadId && sessionBinding?.threadId === threadId ? sessionBinding : null;
  const sessionId = resolvedBinding?.sessionId ?? fallbackSessionId;
  const sessionScope = !threadId
    ? "No active chat selected"
    : resolvedBinding?.scope ?? "Resolving saved chat scope…";
  const sessionWarning = resolvedBinding?.warning ?? null;
  const pollKey = JSON.stringify([threadId ?? "", sessionId]);
  const [pollState, setPollState] = useState<PollState | null>(null);
  const currentPoll = pollState?.key === pollKey ? pollState : null;
  const events = currentPoll?.events ?? [];
  const steps = currentPoll?.steps ?? [];
  const pollStatus = currentPoll?.status ?? "connecting";
  const pollError = currentPoll?.error ?? null;
  const lastUpdatedAt = currentPoll?.lastUpdatedAt ?? null;
  const [turns, setTurns] = useState("");
  const [turnSignal, setTurnSignal] = useState<{
    unnecessary_turns?: number;
    feed_self_improvement?: boolean;
  } | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState | null>(null);
  const analysis = analysisState?.key === pollKey ? analysisState.value : null;
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = "Helix Engine";
  }, []);

  useEffect(() => {
    if (!threadId) return;
    let live = true;
    const controller = new AbortController();
    void getStoredChatThreadReadResult(threadId, {
      bounded: true,
      timeoutMs: 2_500,
      signal: controller.signal,
    })
      .then(({ thread }) => {
        if (!live) return;
        const projectId = engineProjectIdForThread(thread, activeProjectId);
        setSessionBinding({
          threadId,
          sessionId: engineSessionIdFor(threadId, projectId),
          scope: projectId
            ? `Project workspace · thread ${threadId}`
            : `Thread workspace · ${threadId}`,
          warning: null,
        });
      })
      .catch((cause: unknown) => {
        if (!live || controller.signal.aborted) return;
        setSessionBinding({
          threadId,
          sessionId: fallbackSessionId,
          scope: activeProjectId
            ? `Project workspace · thread ${threadId}`
            : `Thread workspace · ${threadId}`,
          warning: `Saved chat scope could not be verified (${helixErrorMessage(cause)}). Using the current app scope.`,
        });
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [threadId, activeProjectId, fallbackSessionId]);

  useEffect(() => {
    let live = true;
    let timer: number | null = null;
    const load = async () => {
      const snapshot = await loadHelixSnapshot(authFetch, sessionId, threadId);
      if (!live) return;
      setPollState((previous) => {
        const prior = previous?.key === pollKey ? previous : null;
        return {
          key: pollKey,
          events: snapshot.events ?? prior?.events ?? [],
          steps: snapshot.steps ?? prior?.steps ?? [],
          status: snapshot.errors.length ? "degraded" : "connected",
          error: snapshot.errors.length ? snapshot.errors.join(" · ") : null,
          lastUpdatedAt:
            snapshot.successfulRequests > 0 ? Date.now() : (prior?.lastUpdatedAt ?? null),
        };
      });
      if (live) timer = window.setTimeout(() => void load(), 1_000);
    };
    void load();
    return () => {
      live = false;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [pollKey, sessionId, threadId]);

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
      // Manual inspection is not objective verification. The backend also
      // enforces this distinction even if an old client sends verified=true.
      verified: false,
      final_result: steps.at(-1)?.result ?? "",
    };
    try {
      const body = await fetchHelixJson<AnalyzeResult>(
        authFetch,
        "/api/helix-engine/analyze",
        "Analyze",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      setAnalysisState({ key: pollKey, value: body });
    } catch (cause) {
      setError(helixErrorMessage(cause));
    }
  }

  async function checkTurns() {
    setError(null);
    try {
      const body = await fetchHelixJson<{
        unnecessary_turns?: number;
        feed_self_improvement?: boolean;
      }>(authFetch, "/api/helix-engine/semantic-turns", "Semantic-turn check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          turns: turns
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean),
        }),
      });
      setTurnSignal(body);
    } catch (cause) {
      setError(helixErrorMessage(cause));
    }
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

      <section className="rounded-xl border border-border p-4" data-testid="helix-engine-status">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Engine status</h2>
          <span className="text-xs text-muted-foreground" aria-live="polite">
            {pollStatus === "connecting"
              ? "Connecting"
              : pollStatus === "degraded"
                ? "Retrying"
                : "Connected"}
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{sessionScope}</p>
        {!threadId ? (
          <p className="mt-2 text-sm">
            Helix Engine is ready. Select a chat and complete a turn; this page will follow that
            conversation automatically.
          </p>
        ) : pollStatus === "connected" && events.length === 0 && steps.length === 0 ? (
          <p className="mt-2 text-sm">
            Connected and idle. Tool activity and the latest completed trajectory will appear here
            as the selected chat runs.
          </p>
        ) : (
          <p className="mt-2 text-sm">
            Tracking {events.length} live event{events.length === 1 ? "" : "s"} and {steps.length}{" "}
            trajectory step{steps.length === 1 ? "" : "s"}.
          </p>
        )}
        {lastUpdatedAt ? (
          <p className="mt-1 text-xs text-muted-foreground">
            Last refresh {new Date(lastUpdatedAt).toLocaleTimeString()}
          </p>
        ) : null}
        {sessionWarning ? (
          <p className="mt-2 text-xs text-muted-foreground" role="status">
            {sessionWarning}
          </p>
        ) : null}
        {pollError ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            Telemetry refresh issue: {pollError}. Retrying automatically; already loaded data stays
            visible.
          </p>
        ) : null}
      </section>

      {error ? (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      ) : null}

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
            className="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void analyzeSession()}
            disabled={steps.length === 0}
            title={steps.length ? "Analyze the captured trajectory" : "No captured tool steps yet"}
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
