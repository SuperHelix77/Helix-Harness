"""Portable tests for the FSR3-class Metal upscaler.

These check the parts that are pure logic and can run without a GPU: the
Lanczos2 polynomial transcribed from AMD's reference, the honesty of the
receipts, and that the ghosting metric is defined well enough to be able to
fail. The GPU results themselves are produced by native/UpscaleLab.swift.

Nothing here proves the upscaler is better than FSR, MetalFX or DLSS. Those
claims are not supported by this repository and these tests do not make them.
"""
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = sorted((ROOT / "fsr3metal" / "evidence").glob("upscale-*.json"))


def lanczos2_approx_sq_no_clamp(x2):
    """AMD's polynomial fit, from ffx_fsr3upscaler_sample.h."""
    a = (2.0 / 5.0) * x2 - 1.0
    b = (1.0 / 4.0) * x2 - 1.0
    return ((25.0 / 16.0) * a * a - (25.0 / 16.0 - 1.0)) * (b * b)


def lanczos2_approx(x):
    return lanczos2_approx_sq_no_clamp(min(x * x, 4.0))


def lanczos2_exact(x):
    x = min(abs(x), 2.0)
    if abs(x) < 1e-5:
        return 1.0
    return ((math.sin(math.pi * x) / (math.pi * x))
            * (math.sin(0.5 * math.pi * x) / (0.5 * math.pi * x)))


class LanczosPolynomial(unittest.TestCase):
    def test_unity_at_origin_and_support_edges(self):
        self.assertAlmostEqual(lanczos2_approx(0.0), 1.0, places=6)
        self.assertAlmostEqual(lanczos2_approx(1.0), 0.0, places=6)
        self.assertAlmostEqual(lanczos2_approx(2.0), 0.0, places=6)

    def test_tracks_true_lanczos2(self):
        # It is an approximation, not a copy, but it must be close enough to
        # be usable as a reconstruction filter.
        for x in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            with self.subTest(x=x):
                self.assertLess(abs(lanczos2_approx(x) - lanczos2_exact(x)), 0.06)

    def test_symmetric_and_clamped(self):
        for x in (0.2, 0.6, 1.2, 3.0):
            self.assertAlmostEqual(lanczos2_approx(x), lanczos2_approx(-x), places=6)
        self.assertAlmostEqual(lanczos2_approx(2.5), lanczos2_approx(2.0), places=6)


class ReceiptHonesty(unittest.TestCase):
    def test_receipts_exist(self):
        self.assertTrue(RECEIPTS, "no upscale receipt recorded")

    def test_no_competitive_claims(self):
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            for key in ("better_than_base_fsr_claimed", "better_than_metalfx_claimed",
                        "better_than_dlss_claimed", "real_game_content",
                        "real_fsr_weights", "crossover_interop_tested"):
                with self.subTest(receipt=path.name, key=key):
                    self.assertIn(key, data)
                    self.assertFalse(data[key], key + " must not be claimed")

    def test_ghosting_metric_has_discriminating_power(self):
        """A ghosting metric that cannot fail is not a test.

        This is the check that caught three broken versions of the metric,
        each of which would have passed a pipeline that does no reprojection.
        """
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            if "ghosting" not in data:
                continue
            g = data["ghosting"]
            with self.subTest(receipt=path.name):
                self.assertTrue(g["metric_has_discriminating_power"],
                                "ghosting control does not separate a correct "
                                "pipeline from a naive blend")
                self.assertGreater(g["naive_blend_control_trailing_energy"],
                                   g["trailing_energy"] + 0.01)

    def test_pipeline_is_at_background_level(self):
        """A reprojecting pipeline must leave no visible trail."""
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            if "ghosting" not in data:
                continue
            g = data["ghosting"]
            with self.subTest(receipt=path.name):
                self.assertLessEqual(g["trailing_energy"],
                                     g["ground_truth_background"] + 1e-4)

    def test_lanczos_beats_bilinear(self):
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            rows = {r["variant"]: r for r in data["results"]}
            if "bilinear_cpu" not in rows or "spatial_lanczos" not in rows:
                continue
            with self.subTest(receipt=path.name):
                self.assertLess(rows["spatial_lanczos"]["rmse"],
                                rows["bilinear_cpu"]["rmse"])


class SourceIntegrity(unittest.TestCase):
    def test_kernels_reference_provenance(self):
        for name in ("fsr3_spatial.metal", "fsr3_temporal.metal"):
            p = ROOT / "fsr3metal" / "kernels" / name
            self.assertTrue(p.exists(), p)
            text = p.read_text()
            with self.subTest(kernel=name):
                self.assertIn("60f4ea81909200d8542eca14dccb2628b763a9a3", text)
                self.assertIn("PROVENANCE.md", text)

    def test_no_amd_binaries_vendored(self):
        """The FSR4 DLL must never enter the repository."""
        for p in ROOT.rglob("*"):
            if ".git" in p.parts or not p.is_file():
                continue
            with self.subTest(path=str(p.relative_to(ROOT))):
                self.assertNotEqual(p.suffix.lower(), ".dll")
                if p.stat().st_size > 20_000_000:
                    self.fail(f"large binary committed: {p}")


if __name__ == "__main__":
    unittest.main()


class MetricIsWellPosed(unittest.TestCase):
    """Guard against a fixture that hides every real difference.

    An earlier fixture added a 2px checker to the high-resolution source. At 4x
    that becomes 0.5px in the low-resolution input, below Nyquist, so the
    information is destroyed before the upscaler runs. RMSE then carried a
    large constant term and bilinear and Lanczos scored within 1% of each
    other for reasons that had nothing to do with the filters.
    """

    def test_receipts_declare_the_fixture_is_band_limited(self):
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            with self.subTest(receipt=path.name):
                self.assertTrue(data.get("fixture_band_limited"))
                self.assertIn("reconstruction_floor_rmse", data)

    def test_variants_beat_the_memoryless_floor(self):
        """Every memoryless filter must beat the downsampled input itself."""
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            floor = data.get("reconstruction_floor_rmse")
            if floor is None:
                continue
            for row in data["results"]:
                if row["variant"] == "temporal_lock":
                    continue
                with self.subTest(receipt=path.name, variant=row["variant"]):
                    self.assertLess(
                        row["rmse"], floor,
                        "a memoryless filter cannot score worse than the input it read")

    def test_lanczos_beats_bilinear_by_a_real_margin(self):
        """Not just 'better', but better than the run-to-run precision of the
        metric. A 0.1% edge is a coin flip, not a result."""
        for path in RECEIPTS:
            data = json.loads(path.read_text())
            rows = {r["variant"]: r for r in data["results"]}
            if "bilinear_cpu" not in rows or "spatial_lanczos" not in rows:
                continue
            with self.subTest(receipt=path.name):
                b = rows["bilinear_cpu"]["rmse"]
                l = rows["spatial_lanczos"]["rmse"]
                self.assertLess(l, b)
                self.assertGreater((b - l) / b, 0.002,
                                   "improvement is too small to distinguish from noise")
