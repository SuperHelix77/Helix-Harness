import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { readSrc } from "./helpers/kit.ts";

const themeStore = readSrc("features/settings/stores/theme-store.ts");
const controls = readSrc(
  "features/settings/components/appearance-custom-controls.tsx",
);
const css = readSrc("index.css");
const sidebar = readSrc("components/ui/sidebar.tsx");
const dashboard = readSrc("components/layout/dashboard-layout.tsx");
const titlebar = readSrc("components/tauri/window-titlebar.tsx");
const provider = readSrc("app/provider.tsx");
const chatPage = readSrc("features/chat/chat-page.tsx");
const hubPage = readSrc("features/hub/hub-page.tsx");
const studioPage = readSrc("features/studio/studio-page.tsx");
const recipeStudioPage = readSrc("features/recipe-studio/recipe-studio-page.tsx");

test("main background transparency is persisted and synchronized separately from glass surfaces", () => {
  assert.match(themeStore, /helix-main-background-transparency/);
  assert.match(themeStore, /--main-background-alpha/);
  assert.match(
    themeStore,
    /e\.key === GLASS_OPACITY_KEY[\s\S]*e\.key === GLASS_BLUR_KEY[\s\S]*e\.key === MAIN_BACKGROUND_TRANSPARENCY_KEY/,
  );
  assert.match(controls, /glass\.mainBackgroundTransparency/);
  assert.match(controls, /glass\.setMainBackgroundTransparency/);
  assert.match(controls, /MAIN_BACKGROUND_TRANSPARENCY_RANGE/);
});

test("main canvas alpha changes background paint without whole-element opacity", () => {
  assert.match(
    css,
    /--main-background-surface:\s*color-mix\(\s*in srgb,\s*var\(--background\)\s*var\(--main-background-alpha\),\s*transparent\s*\)/,
  );
  assert.match(
    css,
    /html\[data-palette=\"glass\"\] \.app-main-background\s*\{\s*background-color:\s*var\(--main-background-surface\);\s*\}/,
  );
  assert.match(css, /\.app-main-background\s*\{\s*background-color:\s*var\(--background\);\s*\}/);
  assert.doesNotMatch(css, /\.app-main-background\s*\{[^}]*\bopacity\s*:/s);
  for (const source of [
    sidebar,
    dashboard,
    titlebar,
    provider,
    chatPage,
    hubPage,
    studioPage,
    recipeStudioPage,
  ]) {
    assert.match(source, /app-main-background/);
  }
  assert.match(
    css,
    /html\[data-palette=\"glass\"\] body::before\s*\{[^}]*opacity:\s*var\(--main-background-alpha\)/s,
  );
  assert.match(css, /--sidebar:\s*rgb\(237 233 254 \/ var\(--glass-alpha\)\)/);
  assert.match(css, /--popover:\s*rgb\(245 243 255 \/ 0\.72\)/);
});

test("theme bootstrap applies persisted glass and main alpha before React", () => {
  const source = readFileSync(new URL("../public/theme-boot.js", import.meta.url), "utf8");
  const stored = new Map<string, string>([
    ["theme", "light"],
    ["helix-palette", "glass"],
    ["helix-glass-opacity", "999"],
    ["helix-glass-blur", "-9"],
    ["helix-main-background-transparency", "37.6"],
  ]);
  const vars = new Map<string, string>();
  const attrs = new Map<string, string>();
  const classes = new Map<string, boolean>();
  const style = {
    colorScheme: "",
    setProperty(name: string, value: string) {
      vars.set(name, value);
    },
  };
  const root = {
    style,
    classList: {
      toggle(name: string, value: boolean) {
        classes.set(name, value);
      },
    },
    setAttribute(name: string, value: string) {
      attrs.set(name, value);
    },
  };

  vm.runInNewContext(source, {
    document: { documentElement: root },
    localStorage: { getItem: (key: string) => stored.get(key) ?? null },
    matchMedia: () => ({ matches: false }),
  });

  assert.equal(vars.get("--glass-alpha"), "88%");
  assert.equal(vars.get("--glass-blur"), "0px");
  assert.equal(vars.get("--main-background-alpha"), "62%");
  assert.equal(style.colorScheme, "light");
  assert.equal(classes.get("light"), true);
  assert.equal(attrs.get("data-palette"), "glass");
});
