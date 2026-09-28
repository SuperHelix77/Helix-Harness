// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import { test } from "node:test";
import { BlobUrlCache } from "../src/lib/blob-url-cache.ts";

function objectUrl(text: string) {
  const blob = new Blob([text]);
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}

test("setting the same URL twice keeps the cached media usable", async () => {
  const cache = new BlobUrlCache(100);
  const { url, bytes } = objectUrl("media");
  try {
    cache.set("clip", url, bytes);
    cache.set("clip", url, bytes);
    assert.equal(cache.size, 1);
    assert.equal(cache.bytes, bytes);
    assert.equal(await (await fetch(url)).text(), "media");
  } finally { cache.clear(); }
});

test("same-URL replacement updates accounting and LRU order without revocation", async () => {
  const cache = new BlobUrlCache(6);
  const a = objectUrl("aaa");
  const b = objectUrl("bbb");
  try {
    cache.set("a", a.url, a.bytes);
    cache.set("b", b.url, b.bytes);
    cache.set("a", a.url, 4);
    assert.equal(cache.bytes, 7);
    assert.deepEqual(cache.ids(), ["b", "a"]);
    assert.deepEqual(cache.prune(), ["b"]);
    assert.equal(cache.bytes, 4);
    assert.equal(await (await fetch(a.url)).text(), "aaa");
    await assert.rejects(fetch(b.url));
  } finally { cache.clear(); }
});

test("different-URL replacement revokes only the superseded media", async () => {
  const cache = new BlobUrlCache(100);
  const a = objectUrl("before");
  const b = objectUrl("after");
  try {
    cache.set("clip", a.url, a.bytes);
    cache.set("clip", b.url, b.bytes);
    assert.equal(cache.bytes, b.bytes);
    await assert.rejects(fetch(a.url));
    assert.equal(await (await fetch(b.url)).text(), "after");
  } finally { cache.clear(); }
});

test("record conversion preserves special IDs as own data properties", () => {
  const cache = new BlobUrlCache(100);
  const ids = ["__proto__", "constructor", "toString"];
  try {
    for (const id of ids) {
      const item = objectUrl(id);
      cache.set(id, item.url, item.bytes);
    }
    const record = cache.toRecord();
    for (const id of ids) {
      assert.equal(Object.hasOwn(record, id), true, id);
      assert.equal(record[id], cache.get(id));
    }
    assert.equal(Object.getPrototypeOf(record), Object.prototype);
  } finally { cache.clear(); }
});

test("pruning preserves protected IDs and deletion remains idempotent", async () => {
  const cache = new BlobUrlCache(3);
  const a = objectUrl("aaaa");
  const b = objectUrl("bbbb");
  try {
    cache.set("a", a.url, a.bytes);
    cache.set("b", b.url, b.bytes);
    assert.deepEqual(cache.prune(["a"]), ["b"]);
    assert.equal(await (await fetch(a.url)).text(), "aaaa");
    assert.equal(cache.delete("a"), true);
    assert.equal(cache.delete("a"), false);
    assert.equal(cache.bytes, 0);
    assert.equal(cache.size, 0);
    await assert.rejects(fetch(a.url));
  } finally { cache.clear(); }
});

test("clear revokes every URL and resets accounting", async () => {
  const cache = new BlobUrlCache(100);
  const items = [objectUrl("one"), objectUrl("two")];
  items.forEach((item, index) => cache.set(String(index), item.url, item.bytes));
  cache.clear();
  cache.clear();
  assert.equal(cache.bytes, 0);
  assert.deepEqual(cache.toRecord(), {});
  for (const item of items) await assert.rejects(fetch(item.url));
});
