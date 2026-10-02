"""Offline metadata contract for a future temporal bridge; NOT a bridge implementation.

No Python object in this module proves a live texture, fence or imported handle.
Only the synthetic native-fixture descriptor lane is admitted. CrossOver remains
rejected until a separately tested native adapter exists.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Tuple


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class TextureInfo:
    token: str
    width: int
    height: int
    format: str
    device: str


@dataclass(frozen=True)
class FrameInfo:
    frame_id: int
    stream_id: str
    color: TextureInfo
    depth: TextureInfo
    motion: TextureInfo
    output: TextureInfo
    jitter_pixels: Tuple[float, float]
    motion_scale: Tuple[float, float]
    exposure: float
    pre_exposure: float
    delta_ms: float
    reset: bool
    motion_at_display_resolution: bool = False
    motion_contains_jitter: bool = False
    jitter_cancellation: bool = False
    inverted_depth: bool = True
    infinite_depth: bool = False
    color_space: str = "linear"
    motion_direction: str = "current_to_previous"
    backend: str = "native-fixture"


def validate_frame(frame: FrameInfo) -> None:
    """Validate declared metadata, not native resource ownership or synchronization."""
    if frame.backend != "native-fixture":
        raise ContractError("CrossOver/D3DMetal import is not implemented; no fallback substitution")
    if type(frame.frame_id) is not int or frame.frame_id < 0 or not frame.stream_id:
        raise ContractError("A nonnegative frame id and stream identity are required")
    textures = [frame.color, frame.depth, frame.motion, frame.output]
    if any(type(t.width) is not int or type(t.height) is not int or
           min(t.width, t.height) < 1 or max(t.width, t.height) > 16384 or
           not t.token or not t.device for t in textures):
        raise ContractError("Invalid texture dimensions or identity")
    if len({t.device for t in textures}) != 1:
        raise ContractError("Cross-device textures cannot share this timeline")
    if len({t.token for t in textures}) != len(textures):
        raise ContractError("Aliasing is unsupported")
    render = (frame.color.width, frame.color.height)
    display = (frame.output.width, frame.output.height)
    if frame.color.format != "rgba16float" or frame.output.format != "rgba16float":
        raise ContractError("The initial ABI accepts rgba16float color/output only")
    if frame.depth.format != "r32float" or frame.motion.format not in ("rg16float", "rg32float"):
        raise ContractError("Unsupported depth or motion format")
    if (frame.depth.width, frame.depth.height) != render:
        raise ContractError("Depth extent differs from rendered color")
    if (frame.motion.width, frame.motion.height) != (display if frame.motion_at_display_resolution else render):
        raise ContractError("Motion extent disagrees with its declared resolution")
    if any(d < r or d > 3*r for r, d in zip(render, display)):
        raise ContractError("Initial lane accepts 1x through 3x upscaling only")
    if frame.color_space != "linear" or frame.motion_direction != "current_to_previous":
        raise ContractError("Unconverted colorspace or motion convention")
    if len(frame.jitter_pixels) != 2 or len(frame.motion_scale) != 2:
        raise ContractError("Jitter and motion scale require exactly two components")
    if not all(isfinite(x) for x in (*frame.jitter_pixels, *frame.motion_scale,
                                    frame.exposure, frame.pre_exposure, frame.delta_ms)):
        raise ContractError("Nonfinite frame metadata")
    if min(frame.exposure, frame.pre_exposure, frame.delta_ms) <= 0:
        raise ContractError("Exposure and frame delta must be positive")
    if any(x == 0 for x in frame.motion_scale):
        raise ContractError("Zero motion scale destroys motion")
    if frame.motion_contains_jitter and not frame.jitter_cancellation:
        raise ContractError("Jittered motion requires explicit cancellation")


class HistoryGate:
    """Reference lifecycle: prepare -> GPU completion -> commit, one in flight.

    This is an offline state-machine specification. A future native adapter
    must call complete only from its verified GPU completion path.
    """
    def __init__(self) -> None:
        self._last = None
        self._pending = None
        self._generation = 0

    @staticmethod
    def _scope(f: FrameInfo):
        return (f.stream_id, f.color.device, f.color.width, f.color.height,
                f.output.width, f.output.height, f.color_space, f.inverted_depth,
                f.infinite_depth, f.motion_at_display_resolution,
                f.motion_contains_jitter, f.jitter_cancellation)

    def prepare(self, frame: FrameInfo) -> int:
        validate_frame(frame)
        if self._pending is not None:
            raise ContractError("A frame is already in flight")
        continuous = (self._last is not None and self._scope(frame) == self._scope(self._last)
                      and frame.frame_id == self._last.frame_id + 1)
        if not continuous and not frame.reset:
            raise ContractError("First frame, gap, resize or convention change requires reset")
        if self._last is not None and frame.stream_id == self._last.stream_id and frame.frame_id <= self._last.frame_id:
            raise ContractError("Duplicate or stale frame")
        self._generation += 1
        self._pending = (self._generation, frame)
        return self._generation

    def complete(self, ticket: int, *, gpu_succeeded: bool) -> None:
        if self._pending is None or ticket != self._pending[0]:
            raise ContractError("Unknown or already completed frame ticket")
        self._last = self._pending[1] if gpu_succeeded else None
        self._pending = None


def require_end_to_end_evidence(report: dict) -> None:
    """Prevent primitive receipts being accepted as FSR/game performance evidence."""
    if report.get("scope") != "crossover_end_to_end" or not all(
        report.get(k) is True for k in ("full_fsr_implemented", "crossover_interop_tested",
                                       "model_weights_loaded", "quality_gate_passed")
    ):
        raise ContractError("Not admissible as an end-to-end FSR/CrossOver result")
    raise ContractError("End-to-end evidence verification is not implemented in this bootstrap")
