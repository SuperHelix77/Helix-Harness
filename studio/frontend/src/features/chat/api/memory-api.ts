// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type MemorySearchResult = {
  candidate_id?: string;
  record_id?: string;
  account_id?: string;
  thread_id?: string;
  text?: string;
  epistemic_class?: "OBSERVED" | "VERIFIED";
  evidence_refs?: Array<{
    id: string;
    sha256: string;
    byte_length: number;
    media_type: string;
  }>;
};

export type MemorySearchResponse = {
  results: MemorySearchResult[];
  available: boolean;
  status?: string;
  reason?: string;
  exclusions?: Array<{
    candidate_id: string;
    record_id: string;
    reason: string;
  }>;
};

async function jsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export type MemoryGraphSnapshot = {
  nodes: Array<{
    id?: string;
    title?: string;
    text?: string;
    kind?: string;
    thread_id?: string;
    entities?: string[];
  }>;
  edges: Array<{ from?: string; to?: string; relation?: string }>;
  available: boolean;
  reason?: string;
};

export async function fetchMemoryGraph(): Promise<MemoryGraphSnapshot> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4_000);
  try {
    return jsonOrThrow(
      await authFetch("/api/memory/graph", { signal: controller.signal }),
    );
  } catch {
    return { nodes: [], edges: [], available: false, reason: "memory-unavailable" };
  } finally {
    clearTimeout(timer);
  }
}

export async function addLearningExperience(input: {
  text: string;
  threadId?: string;
  kind?: string;
  title?: string;
  idempotencyKey?: string;
}): Promise<{ stored: boolean; reason?: string; transportError?: boolean }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4_000);
  try {
    return jsonOrThrow(
      await authFetch("/api/memory/experiences", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
        signal: controller.signal,
      }),
    );
  } catch {
    return {
      stored: false,
      reason: "memory-unavailable",
      transportError: true,
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function searchLearningMemory(
  query: string,
  limit = 5,
  threadId?: string,
): Promise<MemorySearchResponse> {
  // Project identity is not yet part of the trusted memory contract. Automatic
  // model preflight therefore stays thread-scoped instead of borrowing
  // same-account memories from another project/thread.
  if (!threadId?.trim()) {
    return {
      results: [],
      available: false,
      status: "scope_required",
      reason: "thread-scope-required",
    };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4_000);
  try {
    return jsonOrThrow(
      await authFetch("/api/memory/search/typed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit, threadId }),
        signal: controller.signal,
      }),
    );
  } catch {
    return {
      results: [],
      available: false,
      status: "unavailable",
      reason: "memory-unavailable",
    };
  } finally {
    clearTimeout(timer);
  }
}
