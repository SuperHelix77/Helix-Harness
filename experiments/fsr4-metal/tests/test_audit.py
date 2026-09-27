import tempfile
import unittest
from pathlib import Path
from scripts.audit_sources import inventory


class AuditTests(unittest.TestCase):
    def test_hash_and_lexical_scan_not_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/"pass.hlsl").write_text("dot4add_i8packed(a,b,0); groupshared int x;")
            (root/"initializers.bin").write_bytes(b"\x00\x01")
            (root/"LICENSE").write_text("test fixture, not a grant")
            result = inventory(root, "a"*40)
            self.assertEqual(len(result["files"]), 3)
            self.assertEqual(result["license_review"], "not_cleared")
            self.assertFalse(result["model_graph_recovered"])
            self.assertFalse(result["revision_verified_against_upstream"])
            self.assertEqual(result["files"][2]["lexical_tokens"]["dot4add_i8packed"], 1)
            self.assertEqual(result, inventory(root, "a"*40))

    def test_branch_not_accepted_as_revision(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            inventory(Path(d), "main")

    def test_empty_tree_rejected(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            inventory(Path(d), "a"*40)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/"a.hlsl").write_text("test")
            (root/"b.hlsl").symlink_to(root/"a.hlsl")
            with self.assertRaises(ValueError): inventory(root, "a"*40)


if __name__ == "__main__": unittest.main()
