// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Helix product policy.
 *
 * Helix Harness is a fork, not the upstream project it derives from. The
 * inherited update surfaces compare the installed build against the *upstream*
 * release feed and then offer to replace the current install with an upstream
 * build. For a Helix-branded app that is not an update, it is a downgrade: it
 * would silently replace Helix's durable runtime, backend overlay, and product
 * layer with the upstream build.
 *
 * Helix therefore ships releases on its own channel, and these inherited
 * "an upstream release exists" surfaces stay off. Upgrading Helix is an
 * explicit operator action against the Helix release, not an in-app prompt
 * driven by someone else's release feed.
 */

/** Upstream-release-driven update surfaces (web update check, llama.cpp prebuilt nags). */
export const HELIX_UPSTREAM_UPDATE_NAGS_ENABLED = false;

/** Human-readable product name used in UI copy. */
export const HELIX_PRODUCT_NAME = "Helix Harness";
