from dataclasses import replace
import unittest
from sequence_quality import QualityError, Spec, evaluate, require_disjoint_scenes


def fixture():
    spec = Spec("frozen-scene", "a"*64, "performance", 2, 2, 4, 4, 1,
                (0, 1, 2), (True, False, False))
    ref = [[0.5]*16 for _ in range(3)]
    maps = [None, tuple(range(16)), tuple(range(16))]
    masks = [{"hud": tuple(p < 4 for p in range(16)),
              "disocclusion": tuple(p >= 12 for p in range(16))} for _ in range(3)]
    return spec, ref, maps, masks


class SequenceTests(unittest.TestCase):
    def test_exact_sequence_is_not_production_approval(self):
        s, r, m, roi = fixture()
        report = evaluate(s, s, s, r, r, r, m, roi)
        self.assertTrue(report["numeric_noninferiority_passed"])
        self.assertFalse(report["production_quality_admitted"])

    def test_rare_bad_pixel_not_hidden_by_mean(self):
        s, r, m, roi = fixture()
        baseline = [[0.51]*16 for _ in range(3)]
        candidate = [list(f) for f in r]; candidate[1][0] = 0.52
        report = evaluate(s, s, s, r, baseline, candidate, m, roi)
        self.assertLess(report["frames"][1]["candidate"]["mse"], report["frames"][1]["baseline"]["mse"])
        self.assertIn("max_abs", {x["metric"] for x in report["regressions"]})
        self.assertFalse(report["numeric_noninferiority_passed"])

    def test_temporal_flicker_detected_at_equal_spatial_error(self):
        s, r, m, roi = fixture()
        base = [[0.625]*16 for _ in range(3)]
        candidate = [[v]*16 for v in (0.625, 0.375, 0.625)]
        report = evaluate(s, s, s, r, base, candidate, m, roi)
        self.assertEqual({x["metric"] for x in report["regressions"]}, {"temporal_mse"})

    def test_input_scale_cannot_be_secretly_raised(self):
        s, r, m, roi = fixture()
        with self.assertRaises(QualityError): evaluate(s, s, replace(s, render_width=3), r, r, r, m, roi)

    def test_capture_identity_and_population(self):
        s, r, m, roi = fixture()
        for other in (replace(s, capture_sha256="b"*64), replace(s, frame_ids=(0, 2, 3))):
            with self.assertRaises(QualityError): evaluate(s, s, other, r, r, r, m, roi)

    def test_nonfinite_wrong_shape_or_missing_frame(self):
        s, r, m, roi = fixture()
        for bad in ([[float("nan")]*16]*3, [[0.5]*15]*3, r[:2]):
            with self.assertRaises(QualityError): evaluate(s, s, s, r, r, bad, m, roi)

    def test_reset_cannot_use_history(self):
        s, r, m, roi = fixture(); s = replace(s, resets=(True, True, False))
        with self.assertRaises(QualityError): evaluate(s, s, s, r, r, r, m, roi)
        m[1] = None
        self.assertTrue(evaluate(s, s, s, r, r, r, m, roi)["numeric_noninferiority_passed"])

    def test_empty_temporal_coverage_rejected(self):
        s, r, m, roi = fixture(); m[1] = (-1,)*16
        with self.assertRaises(QualityError): evaluate(s, s, s, r, r, r, m, roi)

    def test_all_reset_sequence_cannot_claim_temporal_coverage(self):
        s, r, m, roi = fixture(); s = replace(s, resets=(True, True, True))
        with self.assertRaises(QualityError): evaluate(s, s, s, r, r, r, [None]*3, roi)

    def test_missing_stress_region_rejected(self):
        s, r, m, roi = fixture()
        for masks in roi: masks["hud"] = (False,)*16
        with self.assertRaises(QualityError): evaluate(s, s, s, r, r, r, m, roi)

    def test_bad_tolerance_policy(self):
        s, r, m, roi = fixture()
        with self.assertRaises(QualityError): evaluate(s, s, s, r, r, r, m, roi, {"mse": 1})

    def test_scene_disjointness(self):
        require_disjoint_scenes(["game1/scene1"], ["game1/scene2"], ["game2/scene1"])
        with self.assertRaises(QualityError): require_disjoint_scenes(["same"], ["dev"], ["same"])

    def test_ultra_performance_fixed_three_x(self):
        s, r, m, roi = fixture()
        replace(s, mode="ultra_performance", render_width=2, render_height=2, width=6, height=6).validate()
        with self.assertRaises(QualityError): replace(s, mode="ultra_performance").validate()


if __name__ == "__main__": unittest.main()
