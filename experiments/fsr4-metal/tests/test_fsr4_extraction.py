"""Tests for the FSR4 extraction evidence.

These guard two things that are easy to get wrong and expensive to get wrong
silently:

  1. the numbers claimed about the artifact, and
  2. the boundary of what is NOT known, so a partial result cannot be read as
     a complete one.

No weight bytes are read or written by any test. The DLL is never executed and
is not present in the repository.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"
BINARY_SHA = "4e7dc37aebea3a90e3d3cc43e24cb2b54176b2535315f20dbe63b3b7cfc56b1e"


def load(name):
    p = EVID / name
    if not p.exists():
        raise unittest.SkipTest(f"{name} not present")
    return json.loads(p.read_text())


class BinaryInventory(unittest.TestCase):
    def test_identifies_the_vendor_product(self):
        d = load("fsr4-binary-inventory-20260927.json")
        i = d["identification"]
        self.assertIn("Advanced Micro Devices", i["vendor"])
        self.assertIn("FSR4", i["product"])
        self.assertIn("v07", i["product"])

    def test_records_that_the_binary_was_never_executed(self):
        d = load("fsr4-binary-inventory-20260927.json")
        self.assertFalse(d["status"]["executed"])
        self.assertFalse(d["status"]["weights_extracted"])
        self.assertEqual(d["status"]["license_review"], "not_cleared")

    def test_corrects_the_shader_label_misreading(self):
        """The fsr4_model_v07_* strings are DXBC/DXIL labels, not weights."""
        d = load("fsr4-binary-inventory-20260927.json")
        note = d["identification"]["embedded_model_blobs_note"]
        self.assertIn("shader", note.lower())
        self.assertTrue(d["corrections"], "the earlier misreading must stay on record")

    def test_dxbc_shader_containers_dominate_rdata(self):
        d = load("fsr4-binary-inventory-20260927.json")
        comp = d["identification"]["composition_analysis"]
        self.assertGreater(comp["bytes_claimed_by_shader_containers"],
                           comp["rdata_bytes"] * 0.5)
        self.assertGreater(comp["dxil_dxbc_shader_containers"], 1000)


class Manifest(unittest.TestCase):
    def test_shape_is_never_claimed(self):
        d = load("fsr4-manifest-extraction-20260927.json")
        blob = json.dumps(d).lower()
        for banned in ("out_channels\": ", "in_channels\": ", "kernel_size\": "):
            self.assertNotIn(banned, blob, "a shape field was invented")
        self.assertTrue(d["NOT_established"])

    def test_manifest_looks_like_a_conv_net(self):
        d = load("fsr4-manifest-extraction-20260927.json")
        e = d["established"]
        self.assertGreater(e["tensor_count"], 100)
        self.assertGreater(e["total_tensor_bytes"], 10 * 1024 * 1024)
        hist = e["size_histogram_kib"]
        # per-channel scales plus many 3x3 convolutions
        self.assertGreater(hist.get("8", 0), 10)
        self.assertGreater(hist.get("24", 0) + hist.get("32", 0), 50)

    def test_multiple_tables_reported_not_hidden(self):
        d = load("fsr4-manifest-extraction-20260927.json")
        self.assertGreaterEqual(d["method"]["tables_matching_fingerprint"], 2)


class Graph(unittest.TestCase):
    def test_layer_count_and_contiguity(self):
        d = load("fsr4-graph-extraction-20260927.json")
        g = d["established"]["graph"]
        for res, info in g.items():
            with self.subTest(res=res):
                self.assertEqual(info["layer_count"], 191)
                self.assertTrue(info["contiguous"])
                self.assertEqual(info["missing_indices"], [])
                self.assertEqual(info["index_max"] - info["index_min"] + 1, 191)

    def test_layer_names_match_layers_times_resolutions(self):
        d = load("fsr4-graph-extraction-20260927.json")
        b = d["per_layer_bindings"]
        self.assertEqual(len(b), 382)
        self.assertEqual(len(b), 191 * 2)

    def test_shapes_are_still_declared_unknown(self):
        d = load("fsr4-graph-extraction-20260927.json")
        joined = " ".join(d["NOT_established"]).lower()
        self.assertIn("shape", joined)
        self.assertIn("not guessed", joined)

    def test_same_binary_across_all_reports(self):
        for name in ("fsr4-binary-inventory-20260927.json",
                     "fsr4-manifest-extraction-20260927.json",
                     "fsr4-graph-extraction-20260927.json"):
            d = load(name)
            with self.subTest(receipt=name):
                # the binary inventory predates the shared schema and keeps the
                # hash under file/sha256; the extraction reports use
                # source_binary/sha256. Both must name the same artifact.
                src = d.get("source_binary") or d["file"]
                self.assertEqual(src["sha256"], BINARY_SHA)
                if "source_binary" in d:
                    self.assertFalse(d["source_binary"]["executed"])


class NothingVendored(unittest.TestCase):
    def test_no_dll_and_no_large_blobs_in_the_repo(self):
        for p in ROOT.rglob("*"):
            if ".git" in p.parts or not p.is_file():
                continue
            with self.subTest(path=str(p.relative_to(ROOT))):
                self.assertNotEqual(p.suffix.lower(), ".dll")
                self.assertLess(p.stat().st_size, 8 * 1024 * 1024,
                                "a large binary or weight blob is committed")

    def test_extraction_tools_do_not_write_weights(self):
        for tool in ("extract_fsr4_manifest.py", "extract_fsr4_graph.py"):
            p = ROOT / "scripts" / tool
            self.assertTrue(p.exists(), tool)
            text = p.read_text()
            with self.subTest(tool=tool):
                self.assertIn("executed", text)
                # the only write target is the JSON report path from argv
                self.assertNotIn("wb", text.replace("w b", ""))
                self.assertNotIn(".write_bytes", text)
                self.assertNotIn("shutil", text)


if __name__ == "__main__":
    unittest.main()
