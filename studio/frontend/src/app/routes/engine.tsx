// SPDX-License-Identifier: AGPL-3.0-only
import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const HelixEnginePage = lazyRouteComponent(
  () => import("@/features/helix-engine/engine-page"),
  "HelixEnginePage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/engine",
  staticData: { title: "Helix Engine" },
  beforeLoad: () => requireAuth(),
  validateSearch: (search: Record<string, unknown>): { thread?: string } => ({
    thread:
      typeof search.thread === "string" && search.thread.length > 0
        ? search.thread
        : undefined,
  }),
  component: HelixEnginePage,
});
