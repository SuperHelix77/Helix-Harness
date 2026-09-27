// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

function readStoredNumber(key, fallback, min, max) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    const value = Number(raw);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, Math.round(value)));
  } catch {
    return fallback;
  }
}

try {
  // v2.1 initially persisted the visually heavy "Balanced" tuple on first
  // launch. Move only that exact untouched tuple to the new translucent
  // default; custom, Airy, and Opaque choices remain authoritative.
  try {
    const materialVersionKey = "helix-material-profile-version";
    if (localStorage.getItem(materialVersionKey) !== "2") {
      const legacyDefaults = [
        ["helix-glass-blur", "28", "34"],
        ["helix-main-background-transparency", "34", "62"],
        ["helix-sidebar-transparency", "46", "58"],
        ["helix-chat-surface-transparency", "72", "84"],
        ["helix-composer-transparency", "18", "32"],
        ["helix-right-rail-transparency", "38", "52"],
      ];
      if (legacyDefaults.every(([key, previous]) => localStorage.getItem(key) === previous)) {
        for (const [key, , next] of legacyDefaults) localStorage.setItem(key, next);
      }
      localStorage.setItem(materialVersionKey, "2");
    }
  } catch {}

  let theme = "system";
  let palette = null;
  try {
    theme = localStorage.getItem("theme") || "system";
    palette = localStorage.getItem("helix-palette") || "glass";
  } catch {}
  const dark =
    theme === "dark" ||
    (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("light", !dark);
  root.style.colorScheme = dark ? "dark" : "light";

  const preferences = [
    ["glass-opacity", "glass-alpha", 36, 8, 88, "%", false],
    ["glass-blur", "glass-blur", 34, 0, 48, "px", false],
    ["main-background-transparency", "main-background-alpha", 62, 0, 100, "%", true],
    ["sidebar-transparency", "sidebar-material-alpha", 58, 0, 88, "%", true],
    ["chat-surface-transparency", "chat-material-alpha", 84, 0, 92, "%", true],
    ["composer-transparency", "composer-material-alpha", 32, 0, 72, "%", true],
    ["right-rail-transparency", "right-rail-material-alpha", 52, 0, 88, "%", true],
  ];
  for (const [key, property, fallback, min, max, unit, invert] of preferences) {
    const value = readStoredNumber(`helix-${key}`, fallback, min, max);
    root.style.setProperty(`--${property}`, `${invert ? 100 - value : value}${unit}`);
  }

  if (["classic", "minimal", "glass"].includes(palette)) {
    root.setAttribute("data-palette", palette);
  }
  if (palette === "glass") root.setAttribute("data-glass", "");
  root.setAttribute("data-ink", dark ? "light" : "dark");
} catch {}
