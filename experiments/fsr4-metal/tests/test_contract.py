import unittest
from dataclasses import replace
from lab_contract import ContractError, FrameInfo, HistoryGate, TextureInfo, validate_frame, require_end_to_end_evidence


def frame(**changes):
    f = FrameInfo(0, "fixture", TextureInfo("c", 8, 8, "rgba16float", "gpu"),
                  TextureInfo("d", 8, 8, "r32float", "gpu"),
                  TextureInfo("m", 8, 8, "rg16float", "gpu"),
                  TextureInfo("o", 16, 16, "rgba16float", "gpu"),
                  (0.25, -0.25), (1.0, 1.0), 1.0, 1.0, 16.67, True)
    return replace(f, **changes)


class ContractTests(unittest.TestCase):
    def test_native_metadata(self):
        validate_frame(frame())

    def test_crossover_fails_closed(self):
        for backend in ("crossover-d3dmetal", "dxmt", "screen-capture", "unknown"):
            with self.subTest(backend=backend), self.assertRaises(ContractError):
                validate_frame(frame(backend=backend))

    def test_cross_device(self):
        with self.assertRaises(ContractError):
            validate_frame(frame(depth=replace(frame().depth, device="other")))

    def test_aliasing(self):
        with self.assertRaises(ContractError):
            validate_frame(frame(output=replace(frame().output, token="c")))

    def test_invalid_extent_and_format(self):
        for change in ({"width": 0}, {"width": 7}, {"format": "rgba8unorm"}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                validate_frame(frame(depth=replace(frame().depth, **change)))

    def test_motion_at_display(self):
        with self.assertRaises(ContractError):
            validate_frame(frame(motion_at_display_resolution=True))
        validate_frame(frame(motion_at_display_resolution=True,
                             motion=replace(frame().motion, width=16, height=16)))

    def test_colorspace_motion_and_jitter(self):
        for change in ({"color_space": "srgb"}, {"motion_direction": "previous_to_current"},
                       {"motion_contains_jitter": True}, {"motion_scale": (0.0, 1.0)}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                validate_frame(frame(**change))

    def test_nonfinite_and_zero(self):
        for change in ({"exposure": float("nan")}, {"jitter_pixels": (float("inf"), 0.0)},
                       {"pre_exposure": 0}, {"delta_ms": -1}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                validate_frame(frame(**change))

    def test_first_frame_requires_reset(self):
        with self.assertRaises(ContractError): HistoryGate().prepare(frame(reset=False))

    def test_successful_sequence(self):
        gate = HistoryGate()
        gate.complete(gate.prepare(frame()), gpu_succeeded=True)
        gate.complete(gate.prepare(frame(frame_id=1, reset=False)), gpu_succeeded=True)

    def test_pending_and_stale_completion(self):
        gate = HistoryGate(); ticket = gate.prepare(frame())
        with self.assertRaises(ContractError): gate.prepare(frame(frame_id=1))
        with self.assertRaises(ContractError): gate.complete(ticket+1, gpu_succeeded=True)
        gate.complete(ticket, gpu_succeeded=True)
        with self.assertRaises(ContractError): gate.complete(ticket, gpu_succeeded=True)
        with self.assertRaises(ContractError): gate.prepare(frame())

    def test_failure_requires_reset(self):
        gate = HistoryGate(); gate.complete(gate.prepare(frame()), gpu_succeeded=False)
        with self.assertRaises(ContractError): gate.prepare(frame(frame_id=1, reset=False))
        gate.complete(gate.prepare(frame(frame_id=1)), gpu_succeeded=True)

    def test_gap_resize_and_convention_change(self):
        for f in (frame(frame_id=2, reset=False),
                  frame(frame_id=1, reset=False, output=replace(frame().output, width=20)),
                  frame(frame_id=1, reset=False, inverted_depth=False)):
            gate = HistoryGate(); gate.complete(gate.prepare(frame()), gpu_succeeded=True)
            with self.assertRaises(ContractError): gate.prepare(f)
            gate.complete(gate.prepare(replace(f, reset=True)), gpu_succeeded=True)

    def test_invalid_input_does_not_consume_history(self):
        gate = HistoryGate()
        with self.assertRaises(ContractError): gate.prepare(frame(exposure=0))
        gate.complete(gate.prepare(frame()), gpu_succeeded=True)

    def test_microbenchmark_never_promotes(self):
        with self.assertRaises(ContractError):
            require_end_to_end_evidence({"scope": "synthetic_int8_convolution_microbenchmark"})

    def test_bare_attestations_do_not_prove_a_bridge(self):
        with self.assertRaises(ContractError):
            require_end_to_end_evidence(dict(scope="crossover_end_to_end", full_fsr_implemented=True,
                crossover_interop_tested=True, model_weights_loaded=True, quality_gate_passed=True))


if __name__ == "__main__": unittest.main()
