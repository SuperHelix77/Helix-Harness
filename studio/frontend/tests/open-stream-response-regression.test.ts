// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import { test } from "node:test";
import { openStreamResponse } from "../src/lib/open-stream-response.ts";
import type { StreamFetcher } from "../src/lib/open-stream-response.ts";

const url = "http://localhost:8888/api/events";

test("405 fallback cancels the discarded response and strips the POST body", async () => {
  let cancellations = 0;
  const rejected = new Response(new ReadableStream({
    cancel() { cancellations += 1; },
  }), { status: 405 });
  const success = new Response("events");
  const controller = new AbortController();
  const headers = { Authorization: "Bearer test-token" };
  const options = { retryNetworkErrors: false };
  const init = { body: "cursor=12", headers, signal: controller.signal };
  const calls: RequestInit[] = [];
  const fetcher: StreamFetcher = async (target, requestInit, requestOptions) => {
    assert.equal(target, url);
    assert.equal(requestOptions, options);
    calls.push(requestInit);
    // Use the platform's actual Request validation rather than a permissive mock.
    new Request(target, requestInit);
    return calls.length === 1 ? rejected : success;
  };
  assert.equal(await openStreamResponse(fetcher, url, init, options), success);
  assert.equal(cancellations, 1);
  assert.deepEqual(calls.map((call) => call.method), ["POST", "GET"]);
  assert.equal(calls[1].body, undefined);
  assert.equal(calls[1].headers, headers);
  assert.equal(calls[1].signal, controller.signal);
  assert.equal(init.body, "cursor=12");
});

test("405 response body is cancelled even when there is no POST body", async () => {
  let cancellations = 0;
  const rejected = new Response(new ReadableStream({
    cancel() { cancellations += 1; },
  }), { status: 405 });
  const success = new Response("events");
  let calls = 0;
  const result = await openStreamResponse(async () => ++calls === 1 ? rejected : success, url);
  assert.equal(result, success);
  assert.equal(cancellations, 1);
});

test("rejected cancellation cannot prevent the compatibility retry", async () => {
  const rejected = new Response(new ReadableStream({
    cancel() { return Promise.reject(new Error("cleanup failed")); },
  }), { status: 405 });
  const success = new Response("events");
  let calls = 0;
  assert.equal(await openStreamResponse(async () => ++calls === 1 ? rejected : success, url), success);
  assert.equal(calls, 2);
});

test("non-settling cancellation cannot hang the compatibility retry", { timeout: 2000 }, async () => {
  const rejected = new Response(new ReadableStream({
    cancel() { return new Promise<void>(() => undefined); },
  }), { status: 405 });
  const success = new Response("events");
  let calls = 0;
  assert.equal(await openStreamResponse(async () => ++calls === 1 ? rejected : success, url), success);
});

for (const status of [200, 401, 403, 404, 429, 500]) {
  test(`status ${status} is returned untouched without retries or cancellation`, async () => {
    let calls = 0;
    let cancellations = 0;
    const response = new Response(new ReadableStream({ cancel() { cancellations += 1; } }), { status });
    const result = await openStreamResponse(async () => { calls += 1; return response; }, url);
    assert.equal(result, response);
    assert.equal(calls, 1);
    assert.equal(cancellations, 0);
    await result.body?.cancel();
  });
}

test("network failures propagate without changing the request method", async () => {
  const failure = new Error("network offline");
  let calls = 0;
  await assert.rejects(openStreamResponse(async () => { calls += 1; throw failure; }, url), error => error === failure);
  assert.equal(calls, 1);
});

test("GET fallback happens only once, including another 405", async () => {
  const calls: string[] = [];
  const result = await openStreamResponse(async (_url, init) => {
    calls.push(init.method!);
    return new Response(null, { status: 405 });
  }, url);
  assert.equal(result.status, 405);
  assert.deepEqual(calls, ["POST", "GET"]);
});
