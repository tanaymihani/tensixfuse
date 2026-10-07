#!/usr/bin/env bash
# Install the pinned SFPI toolchain where the ttnn wheel looks for it
# (<ttnn>/runtime/sfpi), so no root is needed. Pass a directory to override.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/sfpi-version"

DEST="${1:-$(python3 -c 'import os, ttnn; print(os.path.join(os.path.dirname(ttnn.__file__), "runtime", "sfpi"))')}"
STAMP="$DEST/.tensixfuse-sfpi-$sfpi_version"
if [ -f "$STAMP" ]; then
  echo "sfpi $sfpi_version already in $DEST" >&2
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
archive="sfpi_${sfpi_version}_x86_64_debian.txz"
curl -fsSL --retry 3 -o "$tmp/$archive" \
  "https://github.com/tenstorrent/sfpi/releases/download/$sfpi_version/$archive"
echo "$sfpi_x86_64_debian_txz_hash  $tmp/$archive" | sha256sum -c - >&2

mkdir -p "$tmp/x"
tar -xJf "$tmp/$archive" -C "$tmp/x"
entries=("$tmp"/x/*)
if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ]; then
  src="${entries[0]}"   # the archive has one top-level directory
else
  src="$tmp/x"
fi
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$src" "$DEST"
touch "$STAMP"
echo "sfpi $sfpi_version installed in $DEST ($(du -sh "$DEST" | cut -f1))" >&2
