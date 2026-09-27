// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, useState } from "react";

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat";
import { useNavigate, useSearch } from "@tanstack/react-router";

const HelixWorkflowPanel = lazy(() =>
  import("./engine-panels").then((module) => ({ default: module.HelixWorkflowPanel })),
);
const HelixExecutionGraphPanel = lazy(() =>
  import("./engine-panels").then((module) => ({ default: module.HelixExecutionGraphPanel })),
);

type EngineView = "workflow" | "execution";

/** Dedicated entry to the same chat-scoped panels also available from Chat's right rail. */
export function HelixEnginePage() {
  const navigate = useNavigate();
  const { thread: routeThreadId } = useSearch({ from: "/engine" });
  const storeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  // The root clears Chat's transient store when leaving /chat. The route carries the
  // selected conversation explicitly so a normal Chat → Engine navigation survives it.
  const activeThreadId = routeThreadId ?? storeThreadId;
  const [view, setView] = useState<EngineView>("execution");
  const openChat = () =>
    void navigate({
      to: "/chat",
      search: activeThreadId ? { thread: activeThreadId } : {},
    });

  return (
    <main
      className="app-main-background flex h-full min-h-0 flex-col overflow-hidden"
      data-testid="helix-engine-page"
    >
      <header className="app-main-chrome-background flex min-h-[58px] shrink-0 flex-wrap items-center justify-between gap-3 border-b border-border/50 px-5 py-2">
        <div className="min-w-0">
          <h1 className="truncate font-heading text-sm font-semibold tracking-[0.08em]">
            HELIX ENGINE
          </h1>
          <p className="truncate text-xs text-muted-foreground">
            Workflow and evidence for the selected chat
          </p>
        </div>
        {activeThreadId ? (
          <div
            className="flex max-w-full shrink-0 items-center gap-1 overflow-x-auto rounded-xl border border-border/60 bg-background/35 p-1"
            role="group"
            aria-label="Helix Engine views"
          >
            <Button
              id="helix-engine-workflow-tab"
              type="button"
              size="sm"
              variant={view === "workflow" ? "secondary" : "ghost"}
              aria-pressed={view === "workflow"}
              onClick={() => setView("workflow")}
            >
              Workflow
            </Button>
            <Button
              id="helix-engine-execution-tab"
              type="button"
              size="sm"
              variant={view === "execution" ? "secondary" : "ghost"}
              aria-pressed={view === "execution"}
              onClick={() => setView("execution")}
            >
              Execution Graph
            </Button>
          </div>
        ) : null}
      </header>

      {activeThreadId ? (
        <div
          className="min-h-0 flex-1 overflow-hidden"
          data-testid="helix-engine-selected-chat-surface"
        >
          <Suspense fallback={<p className="p-4 text-sm text-muted-foreground">Loading Helix Engine…</p>}>
            {view === "workflow" ? (
              <HelixWorkflowPanel threadId={activeThreadId} onClose={openChat} />
            ) : (
              <HelixExecutionGraphPanel threadId={activeThreadId} onClose={openChat} />
            )}
          </Suspense>
        </div>
      ) : (
        <section className="flex min-h-0 flex-1 items-center justify-center p-6 text-center">
          <div className="max-w-md space-y-3">
            <h2 className="font-heading text-lg font-semibold">Choose a chat to inspect</h2>
            <p className="text-sm text-muted-foreground">
              Workflow and execution evidence stay scoped to the active conversation. Open Chat,
              select a conversation, then return here.
            </p>
            <Button type="button" onClick={openChat}>
              Open Chat
            </Button>
          </div>
        </section>
      )}
    </main>
  );
}
