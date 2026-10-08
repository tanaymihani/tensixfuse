#!/usr/bin/env bash
# Download the exported GPU-CorruptNet data (release data-v1) and check it.
# Regenerate it instead with model/corruptnet_export.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data"
cd "$ROOT/data"
if [ ! -f corruptnet.npz ]; then
  curl -fsSL --retry 3 -o corruptnet.npz \
    https://github.com/tanaymihani/tensixfuse/releases/download/data-v1/corruptnet.npz
fi
(sha256sum -c "$ROOT/data.sha256" 2>/dev/null || shasum -a 256 -c "$ROOT/data.sha256") >&2
