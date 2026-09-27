"""Inventory a separately obtained source tree without executing or copying it.

Lexical counts are research hints, NOT a model graph or dependency closure.
Presence of a license file is not license clearance. Symlinks are refused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

TOKENS = ("dot4add_i8packed", "AmdWaveMatrix", "WaveActive", "groupshared",
          "GroupMemoryBarrierWithGroupSync", "ResourceDescriptorHeap", "Conv2D",
          "Quantize", "Dequantize")
EXTENSIONS = {".hlsl", ".hlsli", ".cpp", ".h", ".hpp", ".bin", ".onnx"}


def inventory(root: Path, revision: str) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Expected a real source directory")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Record an exact lowercase 40-character Git commit, not a branch")
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {".git", ".build"})
        if any((Path(directory)/d).is_symlink() for d in dirs):
            raise ValueError("Symlink directories are not audited")
        for name in sorted(names):
            path = Path(directory)/name
            if path.is_symlink(): raise ValueError("Symlink files are not audited")
            is_license = name.lower().startswith(("license", "copying"))
            if path.suffix.lower() not in EXTENSIONS and not is_license: continue
            size = path.stat().st_size
            if size > 128*1024*1024: raise ValueError("File exceeds 128 MiB audit cap")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            item = {"path": path.relative_to(root).as_posix(), "bytes": size,
                    "sha256": digest.hexdigest(), "license_candidate": is_license}
            if path.suffix.lower() not in {".bin", ".onnx"}:
                if size > 8*1024*1024: raise ValueError("Text exceeds 8 MiB lexical scan cap")
                text = path.read_text(encoding="utf-8", errors="strict")
                item["lexical_tokens"] = {token: text.count(token) for token in TOKENS if token in text}
            files.append(item)
    if not files: raise ValueError("No supported source or weight files found")
    return {"schema": "helix.fsr-metal.source-inventory.v1", "declared_revision": revision,
            "revision_verified_against_upstream": False, "license_review": "not_cleared",
            "model_graph_recovered": False, "files": sorted(files, key=lambda item: item["path"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inventory(args.root, args.revision)
        # Exclusive create avoids overwriting the supplied source or prior evidence.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(1, f"source-audit: {error}\n")


if __name__ == "__main__": main()
