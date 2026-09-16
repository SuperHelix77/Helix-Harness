// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  engineProjectIdForThread,
  engineSessionIdFor,
  fetchHelixJson,
  loadHelixSnapshot,
} = await import("../src/features/helix-engine/engine-data.ts");

test("engine session scope matches chat sandbox identity", () => {
  assert.equal(engineSessionIdFor("thread-1", null), "thread-1");
  assert.equal(engineSessionIdFor("thread-1", "project-1"), "project-project-1");
  assert.equal(engineSessionIdFor(null, null), "default");

  assert.equal(
    engineProjectIdForThread({ projectId: "saved-project" }, "stale-project"),
    "saved-project",
  );
  assert.equal(
    engineProjectIdForThread({ projectId: null }, "stale-project"),
    null,
    "a persisted non-project thread must not inherit a stale active project",
  );
  assert.equal(engineProjectIdForThread(undefined, "fresh-project"), "fresh-project");
});

test("Helix JSON errors keep the backend detail", async () => {
  await assert.rejects(
    fetchHelixJson(
      async () =>
        new Response(JSON.stringify({ detail: "engine unavailable" }), {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }),
      "/api/helix-engine/example",
      "Trajectory",
    ),
    /Trajectory: engine unavailable/,
  );
});

test("snapshot polling preserves a successful half and reports the failed half", async () => {
  const urls: string[] = [];
  const snapshot = await loadHelixSnapshot(async (url) => {
    urls.push(url);
    if (url.startsWith("/api/helix-engine/live-feed")) {
      return new Response(
        JSON.stringify({ events: [{ kind: "computer", action: "click" }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(JSON.stringify({ detail: "trace temporarily unavailable" }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }, "project-p 1", "thread/a");

  assert.deepEqual(snapshot.events, [{ kind: "computer", action: "click" }]);
  assert.equal(snapshot.steps, undefined);
  assert.equal(snapshot.successfulRequests, 1);
  assert.deepEqual(snapshot.errors, ["Trajectory: trace temporarily unavailable"]);
  assert.deepEqual(urls, [
    "/api/helix-engine/live-feed?session_id=project-p%201",
    "/api/helix-engine/session/project-p%201?thread_id=thread%2Fa",
  ]);
});
