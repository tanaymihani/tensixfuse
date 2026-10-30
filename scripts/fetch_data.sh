#!/usr/bin/env bash
# Download the exported GPU-CorruptNet data (release data-v2) and check it.
# Regenerate it instead with model/corruptnet_export.py and
# model/corruptnet_bn2_export.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data"
cd "$ROOT/data"
for f in corruptnet.npz corruptnet_bn2.npz; do
  if [ ! -f "$f" ]; then
    curl -fsSL --retry 3 -o "$f" \
      "https://github.com/tanaymihani/tensixfuse/releases/download/data-v2/$f"
  fi
done
(sha256sum -c "$ROOT/data.sha256" 2>/dev/null || shasum -a 256 -c "$ROOT/data.sha256") >&2
