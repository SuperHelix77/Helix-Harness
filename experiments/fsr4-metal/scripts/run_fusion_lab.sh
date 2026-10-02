#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p .build evidence
xcrun swiftc -O -framework Metal -framework CryptoKit native/FusionLab.swift -o .build/fusion-lab
.build/fusion-lab "${1:-evidence/fusion-20260927-a01.json}"
