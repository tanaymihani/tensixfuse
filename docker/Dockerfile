# TensixFuse on Tenstorrent's simulators: TT-Lang's functional simulator,
# ttnn (tt-metal), the SFPI kernel compiler and ttsim for Wormhole and Blackhole,
# all pinned to the versions in this repo. Nothing here needs Tenstorrent hardware.
#
#   docker run --rm ghcr.io/tanaymihani/tensixfuse pytest -m ttsim tests/test_ttsim.py
#   docker run --rm -e TENSIXFUSE_ARCH=wormhole ghcr.io/tanaymihani/tensixfuse tensixfuse-doctor
FROM ubuntu:24.04

ARG TTNN_VERSION=0.79.0
ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv ca-certificates curl xz-utils \
      libatomic1 libnuma1 libhwloc15 libgomp1 \
      libmpc3 libmpfr6 libgmp10 libisl23 \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv \
 && pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install "ttnn==${TTNN_VERSION}" "tt-lang-sim==1.1.6" pytest

WORKDIR /opt/tensixfuse
COPY pyproject.toml README.md ttsim-version ttsim.sha256 sfpi-version data.sha256 ./
COPY scripts ./scripts
COPY src ./src
RUN pip install --no-deps -e . \
 && bash scripts/install_sfpi.sh \
 && bash scripts/setup_ttsim.sh blackhole /opt/ttsim/blackhole > /dev/null \
 && bash scripts/setup_ttsim.sh wormhole /opt/ttsim/wormhole > /dev/null
COPY kernels ./kernels
COPY bench ./bench
COPY model ./model
COPY tests ./tests
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV TT_METAL_SLOW_DISPATCH_MODE=1 \
    TT_METAL_DISABLE_SFPLOADMACRO=1 \
    TENSIXFUSE_ARCH=blackhole
LABEL org.opencontainers.image.source="https://github.com/tanaymihani/tensixfuse" \
      org.opencontainers.image.description="TensixFuse kernels and benchmarks on Tenstorrent's ttsim simulator"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "scripts/ttsim_smoke.py"]
