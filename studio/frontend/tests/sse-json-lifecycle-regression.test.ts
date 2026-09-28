// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSseJsonEvents } from "../src/lib/sse-json-events.ts";

const encoder = new TextEncoder();

function stream(chunks: string[], close = true) {
  let cancellations = 0;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      if (close) controller.close();
    },
    cancel() { cancellations += 1; },
  });
  return { body, cancellations: () => cancellations };
}

async function collect<T>(body: ReadableStream<Uint8Array>, stallMs?: number) {
  const result: T[] = [];
  for await (const item of readSseJsonEvents<T>(body, stallMs)) result.push(item);
  return result;
}

test("EOF releases the reader lock", async () => {
  const { body } = stream(['data: {"n":1}\n\n']);
  assert.deepEqual(await collect(body), [{ n: 1 }]);
  assert.equal(body.locked, false);
});

test("DONE releases the lock and cancels an open source", async () => {
  const s = stream(['data: 1\n\ndata: [DONE]\n\ndata: 2\n\n'], false);
  assert.deepEqual(await collect(s.body), [1]);
  assert.equal(s.cancellations(), 1);
  assert.equal(s.body.locked, false);
});

test("consumer break releases the lock and cancels exactly once", async () => {
  const s = stream(['data: 1\n\ndata: 2\n\n'], false);
  for await (const item of readSseJsonEvents(s.body)) {
    assert.equal(item, 1);
    break;
  }
  assert.equal(s.cancellations(), 1);
  assert.equal(s.body.locked, false);
});

test("a stalled stream terminates and releases the lock", { timeout: 2000 }, async () => {
  const s = stream([], false);
  assert.deepEqual(await collect(s.body, 5), []);
  assert.equal(s.cancellations(), 1);
  assert.equal(s.body.locked, false);
});

test("upstream errors propagate and release the lock", async () => {
  const failure = new Error("upstream disconnected");
  const body = new ReadableStream<Uint8Array>({
    start(controller) { controller.error(failure); },
  });
  await assert.rejects(collect(body), (error) => error === failure);
  assert.equal(body.locked, false);
});

test("consumer throw is not swallowed as a JSON parse error", async () => {
  const s = stream(['data: 1\n\ndata: 2\n\n'], false);
  const iterator = readSseJsonEvents(s.body);
  assert.equal((await iterator.next()).value, 1);
  const failure = new Error("consumer stopped");
  try {
    await assert.rejects(iterator.throw(failure), (error) => error === failure);
  } finally {
    await iterator.return(undefined);
  }
  assert.equal(s.body.locked, false);
  assert.equal(s.cancellations(), 1);
});

test("a rejected cancellation does not hide normal completion", async () => {
  const body = new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(encoder.encode("data: [DONE]\n\n")); },
    cancel() { return Promise.reject(new Error("cancel failed")); },
  });
  assert.deepEqual(await collect(body), []);
  assert.equal(body.locked, false);
});

test("a never-settling source cancellation cannot hold the reader lock", { timeout: 2000 }, async () => {
  const body = new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(encoder.encode("data: [DONE]\n\n")); },
    cancel() { return new Promise<void>(() => undefined); },
  });
  assert.deepEqual(await collect(body), []);
  assert.equal(body.locked, false);
});

test("malformed JSON is skipped without discarding subsequent frames", async () => {
  const { body } = stream(['data: bad json\n\ndata: {"ok":true}\n\n']);
  assert.deepEqual(await collect(body), [{ ok: true }]);
});

test("CRLF, comments, and multiline JSON remain supported", async () => {
  const { body } = stream([': heartbeat\r\n\r\nevent: update\r\ndata: {"n":\r\ndata: 7}\r\n\r\n']);
  assert.deepEqual(await collect(body), [{ n: 7 }]);
});

test("all byte boundaries preserve Unicode and CRLF framing", async () => {
  const bytes = encoder.encode('data: {"text":"İstanbul 🧬"}\r\n\r\ndata: [DONE]\r\n\r\n');
  for (let split = 0; split <= bytes.length; split += 1) {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      },
    });
    assert.deepEqual(await collect(body), [{ text: "İstanbul 🧬" }], `split ${split}`);
  }
});

test("EOF discards an unterminated event rather than accepting a partial frame", async () => {
  const { body } = stream(['data: {"n":1}']);
  assert.deepEqual(await collect(body), []);
});
