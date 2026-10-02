"""Deterministic synthetic storage experiment. No model-quality inference."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_codec import byte_ledger, decode, encode


def run() -> dict:
    rows = []
    for seed in (11, 23, 29):
        for kind in ("small_signed", "small_plus_1pct_outliers", "uniform_int8"):
            r = random.Random(seed)
            data = bytearray()
            for i in range(32768):
                v = r.randrange(-128, 128) if kind == "uniform_int8" else r.randrange(-8, 8)
                if kind == "small_plus_1pct_outliers" and i % 100 == 0:
                    v = -128 if i % 200 else 127
                data.append(v & 255)
            raw = bytes(data); archive = encode(raw)
            assert decode(archive) == raw
            rows.append(dict(fixture=kind, seed=seed, **byte_ledger(raw, archive)))
    return {"schema": "helix.fsr-metal.lossless-storage.v1", "scope": "synthetic_codec_only",
            "full_fsr_implemented": False, "real_model_weights": False,
            "quality_improvement_measured": False, "rows": rows}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_codec_lab.py NEW_RECEIPT.json")
    result = run()
    with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    for row in result["rows"]:
        print(row["fixture"], row["seed"], row["mode"], row["archive_bytes"], "bytes; exact")
