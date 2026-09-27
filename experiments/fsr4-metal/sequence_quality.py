"""Reference-side SDR sequence metrics, not a complete perceptual quality gate.

Evaluate baseline and candidate against the SAME high-resolution reference.
Current-to-previous pixel maps and ROI masks must come from the frozen fixture,
not the candidate. Integer maps are a smoke-test interface, not general optical
flow. Callers must verify capture provenance and actual render sizes separately.
"""
from dataclasses import dataclass
import math


class QualityError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    scene_id: str
    capture_sha256: str
    mode: str
    render_width: int
    render_height: int
    width: int
    height: int
    channels: int
    frame_ids: tuple
    resets: tuple

    def validate(self):
        if not self.scene_id or len(self.capture_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.capture_sha256):
            raise QualityError("Fixture identity is required")
        scale = {"performance": 2, "ultra_performance": 3}.get(self.mode)
        dims = (self.render_width, self.render_height, self.width, self.height)
        if scale is None or any(type(n) is not int or not 1 <= n <= 16384 for n in dims):
            raise QualityError("Unsupported mode or dimensions")
        if self.width != scale*self.render_width or self.height != scale*self.render_height:
            raise QualityError("Declared performance mode changed the render scale")
        if type(self.channels) is not int or self.channels not in (1, 3) or len(self.frame_ids) < 2:
            raise QualityError("At least two grayscale/RGB frames are required")
        if len(self.resets) != len(self.frame_ids) or not all(type(v) is bool for v in self.resets) or not self.resets[0]:
            raise QualityError("Explicit reset flags, including the first frame, are required")
        if all(self.resets):
            raise QualityError("A temporal sequence needs at least one consecutive non-reset transition")
        if any(type(i) is not int or i < 0 for i in self.frame_ids):
            raise QualityError("Invalid frame identity")
        for i in range(1, len(self.frame_ids)):
            if self.frame_ids[i] <= self.frame_ids[i-1]:
                raise QualityError("Duplicate or reordered frames")
            if self.frame_ids[i] != self.frame_ids[i-1]+1 and not self.resets[i]:
                raise QualityError("Frame gaps require history reset")


def require_disjoint_scenes(train, development, holdout):
    groups = [set(train), set(development), set(holdout)]
    if any(not g or any(type(s) is not str or not s for s in g) for g in groups):
        raise QualityError("Three nonempty scene populations are required")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise QualityError("Scene leakage across train/development/holdout")


def _frames(frames, spec):
    if len(frames) != len(spec.frame_ids):
        raise QualityError("Frame population mismatch")
    size = spec.width*spec.height*spec.channels
    for f in frames:
        if len(f) != size or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in f):
            raise QualityError("Expected finite normalized SDR samples of the exact size")


def evaluate(spec, baseline_spec, candidate_spec, reference, baseline, candidate,
             previous_pixel_maps, regions, tolerances=None):
    """Per-frame non-inferiority smoke check, including rare and temporal errors.

    No averages across frames can hide a regression. Tolerances (absolute metric
    deltas) must be frozen before evaluation; zero is the exact-preservation
    default. Missing semantic/HDR/LPIPS/visual evaluation never counts as PASS.
    """
    spec.validate()
    if spec != baseline_spec or spec != candidate_spec:
        raise QualityError("Capture, frame population, scale or reset mismatch")
    for frames in (reference, baseline, candidate):
        _frames(frames, spec)
    n = spec.width*spec.height
    count = len(spec.frame_ids)
    if len(previous_pixel_maps) != count or len(regions) != count:
        raise QualityError("Missing reference-side maps or regions")
    keys = ("mse", "max_abs", "edge_mse", "temporal_mse", "hud_mae", "disocclusion_mae")
    limits = dict.fromkeys(keys, 0.0) if tolerances is None else dict(tolerances)
    if set(limits) != set(keys) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in limits.values()):
        raise QualityError("Invalid or incomplete frozen tolerance policy")
    covered = set()
    for i in range(count):
        mapping = previous_pixel_maps[i]
        if spec.resets[i]:
            if mapping is not None:
                raise QualityError("No temporal correspondence may cross a reset")
        elif mapping is None or len(mapping) != n or any(type(j) is not int or j < -1 or j >= n for j in mapping):
            raise QualityError("Invalid reference-side current-to-previous pixel map")
        elif not any(j >= 0 for j in mapping):
            raise QualityError("Empty temporal coverage cannot prove stability")
        if set(regions[i]) != {"hud", "disocclusion"}:
            raise QualityError("Explicit HUD and disocclusion masks required")
        for name, mask in regions[i].items():
            if len(mask) != n or any(type(v) is not bool for v in mask):
                raise QualityError("Invalid region mask")
            if any(mask): covered.add(name)
    if covered != {"hud", "disocclusion"}:
        raise QualityError("The sequence does not cover both required stress regions")
    channels = spec.channels

    def metrics(frames, i):
        errors = [a-b for a, b in zip(frames[i], reference[i])]
        mse = sum(e*e for e in errors)/len(errors)
        edge_errors = []
        for y in range(spec.height):
            for x in range(spec.width):
                p = y*spec.width+x
                for q in ((p+1,) if x+1 < spec.width else ()) + ((p+spec.width,) if y+1 < spec.height else ()):
                    for c in range(channels):
                        edge_errors.append((errors[p*channels+c]-errors[q*channels+c])**2)
        result = {"mse": mse, "max_abs": max(abs(e) for e in errors),
                  "edge_mse": sum(edge_errors)/len(edge_errors), "temporal_mse": None}
        mapping = previous_pixel_maps[i]
        if mapping is not None:
            terms = []
            for p, q in enumerate(mapping):
                if q >= 0:
                    for c in range(channels):
                        old_error = frames[i-1][q*channels+c]-reference[i-1][q*channels+c]
                        terms.append((errors[p*channels+c]-old_error)**2)
            result["temporal_mse"] = sum(terms)/len(terms)
        for name, mask in regions[i].items():
            values = [abs(errors[p*channels+c]) for p in range(n) if mask[p] for c in range(channels)]
            result[name+"_mae"] = sum(values)/len(values) if values else None
        return result

    rows, regressions = [], []
    for i, frame_id in enumerate(spec.frame_ids):
        a, b = metrics(baseline, i), metrics(candidate, i)
        for key in keys:
            if a[key] is not None and b[key] > a[key]+limits[key]:
                regressions.append({"frame_id": frame_id, "metric": key,
                                    "baseline": a[key], "candidate": b[key]})
        rows.append({"frame_id": frame_id, "baseline": a, "candidate": b,
                     "temporal_valid_pixels": 0 if previous_pixel_maps[i] is None else sum(j >= 0 for j in previous_pixel_maps[i]),
                     "roi_pixels": {k: sum(v) for k, v in regions[i].items()}})
    return {"schema": "helix.fsr-metal.sequence-smoke.v1", "scene_id": spec.scene_id,
            "capture_sha256": spec.capture_sha256, "mode": spec.mode,
            "numeric_noninferiority_passed": not regressions, "regressions": regressions,
            "tolerances": limits, "frames": rows, "production_quality_admitted": False,
            "limitations": ["metadata equality is not live capture attestation",
                            "reference integer correspondence only; no subpixel flow interpolation",
                            "normalized SDR only; no HDR or perceptual/semantic/visual gate",
                            "no game timing or input-latency measurement"]}
