#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .build evidence
xcrun swiftc -O -framework Metal -framework CryptoKit native/PrecisionLab.swift -o .build/precision-lab
.build/precision-lab "${1:-evidence/precision-20260927-a01.json}"
