#!/usr/bin/env bash
# Download the pinned ttsim release, check it, and point tt-metal at it.
#
#   scripts/setup_ttsim.sh blackhole            # prints the exports to eval
#   eval "$(scripts/setup_ttsim.sh wormhole)"
#
# Inside GitHub Actions the variables go to $GITHUB_ENV instead.
# Needs ttnn (the wheel or a tt-metal source tree via TT_METAL_HOME) installed,
# because the SoC descriptor ttsim reads ships with tt-metal.
set -euo pipefail

ARCH="${1:-blackhole}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "$ROOT/ttsim-version")"
SIM_DIR="${2:-$HOME/.cache/tensixfuse/ttsim/$VERSION/$ARCH}"

case "$ARCH" in
  blackhole) LIB=libttsim_bh.so;    SOC=blackhole_140_arch.yaml ;;
  wormhole)  LIB=libttsim_wh.so;    SOC=wormhole_b0_80_arch.yaml ;;
  n300)      LIB=libttsim_wh_x2.so; SOC=wormhole_b0_80_arch.yaml ;;
  *) echo "unknown arch '$ARCH' (blackhole | wormhole | n300)" >&2; exit 2 ;;
esac

mkdir -p "$SIM_DIR"
if [ ! -f "$SIM_DIR/$LIB" ]; then
  curl -fsSL --retry 3 -o "$SIM_DIR/$LIB.part" \
    "https://github.com/tenstorrent/ttsim/releases/download/$VERSION/$LIB"
  mv "$SIM_DIR/$LIB.part" "$SIM_DIR/$LIB"
fi

expected="$(awk -v f="$LIB" '$2 == f {print $1}' "$ROOT/ttsim.sha256")"
actual="$(sha256sum "$SIM_DIR/$LIB" 2>/dev/null || shasum -a 256 "$SIM_DIR/$LIB")"
actual="${actual%% *}"
if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
  echo "checksum mismatch for $LIB ($VERSION): expected '$expected', got '$actual'" >&2
  rm -f "$SIM_DIR/$LIB"
  exit 1
fi

# ttsim looks for soc_descriptor.yaml next to the .so.
soc_path=""
candidates=("${TT_METAL_HOME:-}")
if ttnn_root="$(python3 -c 'import os, ttnn; print(os.path.dirname(os.path.dirname(ttnn.__file__)))' 2>/dev/null)"; then
  candidates+=("$ttnn_root")
fi
for base in "${candidates[@]}"; do
  [ -n "$base" ] && [ -d "$base" ] || continue
  soc_path="$(find "$base" -path '*soc_descriptors*' -name "$SOC" 2>/dev/null | head -n 1)"
  [ -n "$soc_path" ] && break
done
if [ -z "$soc_path" ]; then
  echo "could not find $SOC in TT_METAL_HOME or the ttnn install" >&2
  exit 1
fi
cp "$soc_path" "$SIM_DIR/soc_descriptor.yaml"

vars=(
  "TT_METAL_SIMULATOR=$SIM_DIR/$LIB"
  "TT_METAL_SLOW_DISPATCH_MODE=1"
  "TT_METAL_DISABLE_SFPLOADMACRO=1"
  "TENSIXFUSE_ARCH=$ARCH"
)
if [ -n "${GITHUB_ENV:-}" ]; then
  printf '%s\n' "${vars[@]}" >> "$GITHUB_ENV"
  echo "ttsim $VERSION ($ARCH) ready in $SIM_DIR" >&2
else
  for v in "${vars[@]}"; do echo "export $v"; done
fi
