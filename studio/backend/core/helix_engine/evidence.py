# SPDX-License-Identifier: AGPL-3.0-only
"""Evidence sufficiency: representation and deterministic claim adjudication."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any

from .schemas import CacheIntegrityReport, EvidenceClaim, EvidenceStatus
from .trajectory import Trajectory

_SPEED_RE = re.compile(r"(?:\b5\s*[x×]\b|five\s+times|speed|throughput|tok(?:en)?s?/s)", re.I)


@dataclass(frozen=True)
class _ResolvedEvidence:
    text: str
    authoritative: bool
    provenance: str
    bound_claim: str = ""


def _status(supporting: list[str], contradicting: list[str], missing: list[str]) -> EvidenceStatus:
    if contradicting:
        return EvidenceStatus.CONTRADICTED
    if supporting and missing:
        return EvidenceStatus.PARTIALLY_SUPPORTED
    if supporting:
        return EvidenceStatus.SUPPORTED
    if missing:
        return EvidenceStatus.UNVERIFIED
    return EvidenceStatus.UNVERIFIED


def validated_tool_verification(step):
    """Return a passed backend-owned typed verification receipt, else ``None``.

    The model controls tool arguments, including terminal/python command text. Those
    strings can never grant verifier authority. A verification receipt exists only
    when the backend explicitly tagged the tool execution at the capture boundary.
    """
    from .trajectory import (
        ToolVerificationReceipt,
        VerificationKind,
        VerificationStatus,
    )

    receipt = getattr(step, "verification", None)
    if not isinstance(receipt, ToolVerificationReceipt):
        return None
    if not isinstance(receipt.kind, VerificationKind):
        return None
    if receipt.provenance != "backend_tool_capture":
        return None
    if receipt.status != VerificationStatus.PASSED:
        return None
    return receipt


def tool_verification_receipt(step) -> str | None:
    """Return claim-addressable proof from backend-owned typed capture metadata.

    Authority and semantic relevance are separate. A successful verifier without a
    backend-supplied exact claim binding is useful trajectory context, but it cannot
    be cited as proof for arbitrary model-authored prose.
    """
    receipt = validated_tool_verification(step)
    if receipt is None or not str(receipt.claim or "").strip():
        return None
    detail = receipt.detail or str(getattr(step, "result", "") or "")
    return f"{receipt.kind.value} verification passed: {detail}"[:500]


def _normalized_claim(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _supports_claim(item: _ResolvedEvidence, claim: str) -> bool:
    if not item.authoritative or not item.bound_claim:
        return False
    return _normalized_claim(item.bound_claim) == _normalized_claim(claim)


def _objective_evidence(traj: Trajectory, cache: CacheIntegrityReport) -> dict[str, _ResolvedEvidence]:
    evidence: dict[str, _ResolvedEvidence] = {}
    extras = traj.extras if isinstance(traj.extras, dict) else {}
    if extras.get("objective_verified") is True:
        evidence["trajectory:objective_verified"] = _ResolvedEvidence(
            "objective verifier marked the trajectory successful",
            authoritative=True,
            provenance="objective_verifier",
            bound_claim="The requested task objective was achieved.",
        )
    for index, step in enumerate(traj.steps):
        if step.error:
            evidence[f"tool:{index}:error"] = _ResolvedEvidence(
                f"{step.name} failed: {step.error}"[:500],
                authoritative=False,
                provenance="tool_error",
            )
        else:
            # Result refs are context for an auditor, not admissible proof by themselves.
            evidence[f"tool:{index}:result"] = _ResolvedEvidence(
                f"{step.name}: {step.result}"[:500],
                authoritative=False,
                provenance="tool_result",
            )
            receipt = validated_tool_verification(step)
            verification = tool_verification_receipt(step)
            if receipt is not None and verification:
                evidence[f"tool:{index}:verification"] = _ResolvedEvidence(
                    verification,
                    authoritative=True,
                    provenance="typed_tool_verification",
                    bound_claim=receipt.claim,
                )
    telemetry = extras.get("telemetry", {})
    if isinstance(telemetry, dict) and telemetry.get("speed_benchmark_present"):
        ratio = telemetry.get("speed_ratio")
        evidence["benchmark:speed"] = _ResolvedEvidence(
            f"measured speed benchmark ratio={ratio}"[:500],
            authoritative=False,
            provenance="runtime_benchmark",
        )
    if cache.accepted_drafts > 0:
        evidence["speculation:accepted"] = _ResolvedEvidence(
            f"per-turn accepted_drafts={cache.accepted_drafts}",
            authoritative=False,
            provenance="runtime_telemetry",
        )
    return evidence


def evaluate_claim(
    raw: dict[str, Any],
    index: int = 0,
    *,
    objective_evidence: dict[str, str] | None = None,
) -> EvidenceClaim:
    objective = objective_evidence or {}
    claim_text = str(raw.get("claim") or "")[:2_000]
    reported = [str(x)[:500] for x in (raw.get("supporting_evidence") or raw.get("evidence") or []) if str(x).strip()]
    refs = [str(x)[:200] for x in (raw.get("evidence_refs") or []) if str(x).strip()]

    typed = {
        ref: item
        for ref, item in objective.items()
        if isinstance(item, _ResolvedEvidence)
    }
    supporting = [
        typed[ref].text
        for ref in refs
        if ref in typed and _supports_claim(typed[ref], claim_text)
    ]
    unresolved_refs = [ref for ref in refs if ref not in objective]
    untyped_refs = [ref for ref in refs if ref in objective and ref not in typed]
    context_only_refs = [
        ref for ref in refs if ref in typed and not typed[ref].authoritative
    ]
    semantically_unbound_refs = [
        ref
        for ref in refs
        if ref in typed
        and typed[ref].authoritative
        and not _supports_claim(typed[ref], claim_text)
    ]
    contradicting = [str(x)[:500] for x in (raw.get("contradicting_evidence") or []) if str(x).strip()]
    missing = [str(x)[:500] for x in (raw.get("missing_evidence") or raw.get("missing") or []) if str(x).strip()]
    if reported and not supporting:
        missing.append("model-reported support lacks a backend-resolved evidence reference")
    if unresolved_refs:
        missing.extend(f"unresolved evidence ref: {ref}" for ref in unresolved_refs)
    if untyped_refs:
        missing.extend(f"evidence ref lacks typed backend provenance: {ref}" for ref in untyped_refs)
    if context_only_refs and not supporting:
        missing.append("resolved context refs are not authoritative evidence")
    if semantically_unbound_refs:
        missing.extend(
            f"evidence ref is not semantically bound to this claim: {ref}"
            for ref in semantically_unbound_refs
        )
    claimed_status = raw.get("status")
    if claimed_status == EvidenceStatus.NOT_APPLICABLE.value:
        status = EvidenceStatus.NOT_APPLICABLE
    else:
        status = _status(supporting, contradicting, missing)
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return EvidenceClaim(
        claim_id=str(raw.get("claim_id") or f"claim-{index + 1}"),
        claim=claim_text,
        supporting_evidence=supporting,
        reported_supporting_evidence=reported,
        evidence_refs=refs,
        contradicting_evidence=contradicting,
        missing_evidence=list(dict.fromkeys(missing)),
        confidence=confidence,
        status=status,
    )

def build_evidence_claims(
    traj: Trajectory,
    cache: CacheIntegrityReport,
    raw_claims: list[dict[str, Any]] | None = None,
) -> list[EvidenceClaim]:
    objective = _objective_evidence(traj, cache)
    claims = [
        evaluate_claim(raw, i, objective_evidence=objective)
        for i, raw in enumerate(raw_claims or [])
        if isinstance(raw, dict)
    ]
    if not claims:
        verified = "trajectory:objective_verified" in objective
        claims.append(
            EvidenceClaim(
                claim_id="task-outcome",
                claim="The requested task objective was achieved.",
                supporting_evidence=[objective["trajectory:objective_verified"].text] if verified else [],
                evidence_refs=["trajectory:objective_verified"] if verified else [],
                missing_evidence=[] if verified else ["objective outcome verification"],
                confidence=0.9 if verified else 0.4,
                status=EvidenceStatus.SUPPORTED if verified else EvidenceStatus.UNVERIFIED,
            )
        )

    telemetry = traj.extras.get("telemetry", {}) if isinstance(traj.extras, dict) else {}
    ratio = telemetry.get("speed_ratio") if isinstance(telemetry, dict) else None
    benchmark_present = bool(isinstance(telemetry, dict) and telemetry.get("speed_benchmark_present"))
    for claim in claims:
        if not _SPEED_RE.search(claim.claim):
            continue
        if ratio is not None:
            try:
                measured_ratio = float(ratio)
            except (TypeError, ValueError):
                measured_ratio = None
            if measured_ratio is not None:
                benchmark_present = True
                evidence = f"measured speed ratio={measured_ratio:.3f}x"
                if "5" in claim.claim and measured_ratio < 5.0:
                    claim.contradicting_evidence.append(evidence)
                else:
                    claim.supporting_evidence.append(evidence)
        if "5" in claim.claim and cache.accepted_drafts <= 0:
            if benchmark_present:
                claim.contradicting_evidence.append("speculative accepted_drafts=0")
            else:
                claim.missing_evidence.append("real decode benchmark and nonzero accepted_drafts")
        elif cache.accepted_drafts > 0:
            claim.supporting_evidence.append(f"accepted_drafts={cache.accepted_drafts}")
        if not benchmark_present:
            claim.missing_evidence.append("real decode benchmark")
        claim.missing_evidence = list(dict.fromkeys(claim.missing_evidence))
        claim.supporting_evidence = list(dict.fromkeys(claim.supporting_evidence))
        claim.contradicting_evidence = list(dict.fromkeys(claim.contradicting_evidence))
        claim.status = _status(claim.supporting_evidence, claim.contradicting_evidence, claim.missing_evidence)
    return claims
