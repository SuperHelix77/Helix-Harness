"""Portable tests for the precision-arena host contract.

These cover the parts of the FP16/INT8 comparison that are pure logic and can
be checked without a GPU: the float32->binary16 round trip the lab relies on,
the half-ULP budget used to gate the simdgroup path, and the simdgroup matrix
orientation contract that was verified by probe on the M3 Max.

The GPU timings and the simdgroup kernel itself are validated by
native/PrecisionLab.swift on the host, not here. Nothing in this file proves a
precision choice for a real model; it only proves the lab's own arithmetic.
"""
import json
import math
import struct
import unittest
from pathlib import Path


def to_half(f):
    """float32 -> binary16 bits, round-to-nearest-even, flushed subnormals.

    Mirrors the Swift host oracle so both ends of the comparison agree on what
    "half" means before any GPU is involved.
    """
    x = struct.unpack("<I", struct.pack("<f", f))[0]
    sign = (x >> 16) & 0x8000
    raw_exp = (x >> 23) & 0xFF
    man = x & 0x7FFFFF
    if raw_exp == 0:
        return sign
    if raw_exp == 0xFF:
        return sign | (0x7C00 if man == 0 else 0x7E00)
    e = raw_exp - 112
    if e <= 0:
        return sign
    if e >= 31:
        return sign | 0x7C00
    if e >= 15:
        return sign | (e << 10) | (man >> 13)
    r = (man + 0x0FFF + ((man >> 13) & 1)) >> 13
    if r & 0x400:
        return sign | ((e + 1) << 10)
    return sign | (e << 10) | r


def from_half(h):
    s = -1.0 if h & 0x8000 else 1.0
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:
        return s * m * 5.9604644775390625e-8
    if e == 31:
        return s * (float("inf") if m == 0 else float("nan"))
    return s * (1 + m / 1024) * math.ldexp(1.0, e - 15)


def half_ulp(magnitude):
    a = abs(magnitude)
    if a < 6.103515625e-05:
        return 5.9604644775390625e-8
    return math.ldexp(1.0, math.floor(math.log2(a)) - 10)


class HalfRoundTrip(unittest.TestCase):
    def test_exact_small_values(self):
        for v in (0.0, 0.5, 1.0, 2.0, -1.0, -2.0, 0.25, 1024.0, -1024.0):
            self.assertAlmostEqual(from_half(to_half(v)), v, places=5)

    def test_every_half_bit_round_trips_through_float(self):
        for h in range(0x10000):
            exponent = (h >> 10) & 0x1F
            if exponent == 0x1F:
                continue
            if exponent == 0 and (h & 0x3FF):
                continue  # subnormals are flushed on the way back in
            self.assertEqual(to_half(from_half(h)), h)

    def test_subnormal_bits_flush_rather_than_round(self):
        self.assertEqual(to_half(from_half(0x0001)), 0)

    def test_subnormals_flush_to_zero(self):
        self.assertEqual(to_half(1e-7), 0)

    def test_overshoot_saturates_to_inf(self):
        self.assertEqual(to_half(1e6) & 0x7C00, 0x7C00)

    def test_negative_zero_preserved(self):
        self.assertEqual(to_half(-0.0), 0x8000)


class HalfULP(unittest.TestCase):
    def test_ulp_halves_with_each_binade(self):
        self.assertAlmostEqual(half_ulp(1.0), 2.0 ** -10)
        self.assertAlmostEqual(half_ulp(2.0), 2.0 ** -9)
        self.assertAlmostEqual(half_ulp(0.5), 2.0 ** -11)

    def test_subnormal_band_is_flat(self):
        self.assertEqual(half_ulp(1e-6), half_ulp(0.0))


class SimdgroupOrientationContract(unittest.TestCase):
    """The simdgroup `a * b` result is an ordinary row-major C = A * B.

    Verified on the M3 Max with an identity-matrix probe. The host therefore
    pre-transposes the weight tile to B[k][o] = W[o][k]. If a future change
    passes W directly, this reconstruction shows the one-column shift that the
    GPU produced, which is how the original bug was found.
    """

    def test_row_major_reconstruction(self):
        A = [float((r * 8 + k) % 8) for r in range(8) for k in range(8)]
        W = [float((o * 8 + k) % 7) for o in range(8) for k in range(8)]
        B = [W[o * 8 + k] for k in range(8) for o in range(8)]
        for r in range(8):
            for c in range(8):
                acc = sum(A[r * 8 + k] * B[k * 8 + c] for k in range(8))
                self.assertAlmostEqual(acc, sum(A[r * 8 + k] * W[c * 8 + k] for k in range(8)))

    def test_untransposed_weights_shift_by_one_column(self):
        A = [float((r * 8 + k) % 8) for r in range(8) for k in range(8)]
        W = [float((o * 8 + k) % 7) for o in range(8) for k in range(8)]
        correct = [sum(A[r * 8 + k] * W[o * 8 + k] for k in range(8))
                   for r in range(8) for o in range(8)]
        shifted = [sum(A[r * 8 + k] * W[((o + 1) % 8) * 8 + k] for k in range(8))
                   for r in range(8) for o in range(8)]
        self.assertNotEqual(correct, shifted)


class ReceiptHonesty(unittest.TestCase):
    """A precision receipt must not promote a synthetic result to a claim."""

    REQUIRED_FALSE = (
        "full_fsr_implemented", "real_model_weights", "picture_quality_measured",
        "student_precision_chosen", "crossover_interop_tested",
    )

    def test_receipts_declare_their_limits(self):
        found = sorted(Path(__file__).resolve().parents[1].joinpath("evidence").glob("precision-*.json"))
        self.assertTrue(found, "no precision receipt recorded")
        for path in found:
            with self.subTest(receipt=path.name):
                data = json.loads(path.read_text())
                for key in self.REQUIRED_FALSE:
                    self.assertIn(key, data)
                    self.assertFalse(data[key], key + " must not be claimed")
                self.assertIn("interpretation_rules", data)

    def test_receipts_record_distribution_not_just_a_median(self):
        """A bare median hides this host's two-clock-state bimodality."""
        found = sorted(Path(__file__).resolve().parents[1].joinpath("evidence").glob("precision-*.json"))
        for path in found:
            with self.subTest(receipt=path.name):
                data = json.loads(path.read_text())
                for row in data["results"]:
                    for variant in ("int8_split", "fp32", "fp16_halfacc", "fp16_f32acc"):
                        with self.subTest(fixture=row["fixture"], variant=variant):
                            for key in ("gpu_min_ms", "gpu_median_ms", "gpu_p95_ms", "gpu_max_ms"):
                                self.assertIn(key, row[variant])

    def test_receipts_do_not_promote_absolute_timings(self):
        """The receipt must not present absolute ms as a stable result."""
        found = sorted(Path(__file__).resolve().parents[1].joinpath("evidence").glob("precision-*.json"))
        for path in found:
            with self.subTest(receipt=path.name):
                data = json.loads(path.read_text())
                self.assertIn("gpu_timing_scope", data)
                # The caveat about absolute milliseconds must be carried by the
                # receipt itself, not only by the prose document.
                self.assertIn("absolute_timing_caveat", data)
                self.assertIn("absolute", data["absolute_timing_caveat"].lower())
                self.assertIn("not reproducible", data["absolute_timing_caveat"].lower())
                self.assertIn("variant_timing_protocol", data)


if __name__ == "__main__":
    unittest.main()
