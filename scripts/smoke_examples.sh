#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v python3.13 || command -v python3)}"

"$PYTHON_BIN" -m ibm650_it.cli smoke-examples --output "$ROOT/artifacts/smoke_examples"
