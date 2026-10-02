#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .build evidence
xcrun swiftc -O -framework Metal -framework CryptoKit native/KernelLab.swift -o .build/kernel-lab
.build/kernel-lab kernels/conv_i8.metal "${1:-evidence/native-kernels.json}"
python3 - "${1:-evidence/native-kernels.json}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
path = Path(sys.argv[1])
report = json.loads(path.read_text())
report["source_and_binary_sha256"] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in map(Path, ("native/KernelLab.swift", "kernels/conv_i8.metal", ".build/kernel-lab"))
}
report["swift_toolchain"] = subprocess.check_output(["xcrun", "swift", "--version"], text=True).strip()
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
