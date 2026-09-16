// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Apply the stored theme and palette before the bundle loads so the first paint is never the wrong
// mode. Loaded as an external classic script (it blocks parsing, like an inline script) because the
// backend CSP only allows script-src 'self'.
function readStoredNumber(key, fallback, min, max) {
  try {
    var raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    var value = Number(raw);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, Math.round(value)));
  } catch (e) {
    return fallback;
  }
}

try {
  // Storage reads get their own guards so a blocked localStorage (private
  // browsing) still resolves a mode from the OS preference.
  var theme = "system";
  var palette = null;
  try {
    theme = localStorage.getItem("theme") || "system";
    palette = localStorage.getItem("helix-palette") || "glass";
  } catch (e) {}
  var glassOpacity = readStoredNumber("helix-glass-opacity", 36, 8, 88);
  var glassBlur = readStoredNumber("helix-glass-blur", 28, 0, 48);
  var mainTransparency = readStoredNumber(
    "helix-main-background-transparency",
    0,
    0,
    100,
  );
  var dark =
    theme === "dark" ||
    (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  var root = document.documentElement;
  root.classList.toggle("dark", dark);
  root.classList.toggle("light", !dark);
  root.style.colorScheme = dark ? "dark" : "light";
  root.style.setProperty("--glass-alpha", glassOpacity + "%");
  root.style.setProperty("--glass-blur", glassBlur + "px");
  root.style.setProperty(
    "--main-background-alpha",
    100 - mainTransparency + "%",
  );
  if (palette === "classic" || palette === "minimal" || palette === "glass") {
    root.setAttribute("data-palette", palette);
  }
  if (palette === "glass") root.setAttribute("data-glass", "");
  root.setAttribute("data-ink", dark ? "light" : "dark");
} catch (e) {}
