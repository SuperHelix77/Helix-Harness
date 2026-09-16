// SPDX-License-Identifier: AGPL-3.0-only
import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";

export type FeedEvent = {
  action?: string;
  url?: string;
  kind?: string;
  title?: string;
  snippet?: string;
};

export type SessionStep = {
  name: string;
  arguments: string;
  result: string;
  useful_hint: string;
};

type JsonFetcher = (input: string, init?: RequestInit) => Promise<Response>;

export function engineProjectIdForThread(
  storedThread: { projectId?: string | null } | undefined,
  activeProjectId: string | null,
): string | null {
  return storedThread ? (storedThread.projectId ?? null) : activeProjectId;
}

export function engineSessionIdFor(
  threadId: string | null | undefined,
  projectId: string | null | undefined,
): string {
  return sandboxSessionIdFor(threadId ?? undefined, projectId) ?? threadId ?? "default";
}

export function helixErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message.trim();
  if (typeof error === "string" && error.trim()) return error.trim();
  return "Unknown Helix Engine error";
}

function responseDetail(body: unknown): string {
  if (!body || typeof body !== "object") return "";
  const detail = (body as { detail?: unknown }).detail;
  return typeof detail === "string" ? detail.trim() : "";
}

export async function fetchHelixJson<T>(
  fetcher: JsonFetcher,
  url: string,
  label: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetcher(url, init);
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error(
      `${label}: ${response.ok ? "server returned an unreadable response" : `HTTP ${response.status}`}`,
    );
  }
  if (!response.ok) {
    throw new Error(`${label}: ${responseDetail(body) || `HTTP ${response.status}`}`);
  }
  return body as T;
}

export type HelixSnapshot = {
  events?: FeedEvent[];
  steps?: SessionStep[];
  errors: string[];
  successfulRequests: number;
};

export async function loadHelixSnapshot(
  fetcher: JsonFetcher,
  sessionId: string,
  threadId?: string | null,
): Promise<HelixSnapshot> {
  const traceQuery = threadId
    ? `?thread_id=${encodeURIComponent(threadId)}`
    : "";
  const requests = await Promise.allSettled([
    fetchHelixJson<{ events?: FeedEvent[] }>(
      fetcher,
      `/api/helix-engine/live-feed?session_id=${encodeURIComponent(sessionId)}`,
      "Live feed",
    ),
    fetchHelixJson<{ steps?: SessionStep[] }>(
      fetcher,
      `/api/helix-engine/session/${encodeURIComponent(sessionId)}${traceQuery}`,
      "Trajectory",
    ),
  ]);
  const snapshot: HelixSnapshot = { errors: [], successfulRequests: 0 };
  const [feed, trace] = requests;
  if (feed.status === "fulfilled") {
    snapshot.events = feed.value.events ?? [];
    snapshot.successfulRequests += 1;
  } else {
    snapshot.errors.push(helixErrorMessage(feed.reason));
  }
  if (trace.status === "fulfilled") {
    snapshot.steps = trace.value.steps ?? [];
    snapshot.successfulRequests += 1;
  } else {
    snapshot.errors.push(helixErrorMessage(trace.reason));
  }
  return snapshot;
}
