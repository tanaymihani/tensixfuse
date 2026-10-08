#!/bin/sh
# Pick the simulated chip from TENSIXFUSE_ARCH (blackhole or wormhole).
case "${TENSIXFUSE_ARCH:-blackhole}" in
  wormhole) export TT_METAL_SIMULATOR=/opt/ttsim/wormhole/libttsim_wh.so ;;
  blackhole) export TT_METAL_SIMULATOR=/opt/ttsim/blackhole/libttsim_bh.so ;;
  *) echo "TENSIXFUSE_ARCH must be blackhole or wormhole" >&2; exit 2 ;;
esac
exec "$@"
