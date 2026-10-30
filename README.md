# TensixFuse

[![ci](https://github.com/tanaymihani/tensixfuse/actions/workflows/ci.yml/badge.svg)](https://github.com/tanaymihani/tensixfuse/actions/workflows/ci.yml)
[![ttsim](https://github.com/tanaymihani/tensixfuse/actions/workflows/ttsim.yml/badge.svg)](https://github.com/tanaymihani/tensixfuse/actions/workflows/ttsim.yml)
[![metalium](https://github.com/tanaymihani/tensixfuse/actions/workflows/metalium.yml/badge.svg)](https://github.com/tanaymihani/tensixfuse/actions/workflows/metalium.yml)

How much DRAM traffic does kernel fusion actually save on a Tenstorrent chip? And how far can a model's weights be pushed into Tenstorrent's block-float formats before the model notices? I wrote fused kernels for Tensix cores to find out, in TT-Lang (Tenstorrent's Python kernel language) and by hand in TT-Metalium C++, and ran two real models through them: GPT-2's MLP blocks and the classifier head of [GPU-CorruptNet](https://github.com/tanaymihani/gpu-corruptnet), my GPU-corruption detector.

None of this ran on silicon. Device numbers come from Tenstorrent's two simulators: TT-Lang's functional simulator, which logs every tile that moves between DRAM, L1 and other cores, and [ttsim](https://github.com/tenstorrent/ttsim), a full-system simulator of Wormhole and Blackhole chips that is designed to be bit-exact with the hardware. So correctness and data movement are real. Timing isn't, because ttsim doesn't model it, and I don't report any device latency.

The repo also runs like a small hardware lab with simulated chips: CI on simulated Wormhole and Blackhole, a C++ program built against Tenstorrent's released SDK, a pinned container, the benchmark sweep fanned out across hosts and as a Kubernetes JobSet, and an Ansible playbook that turns a bare Ubuntu machine into a checked simulator node.

TensixFuse is listed in Tenstorrent's [tt-awesome](https://github.com/tenstorrent/tt-awesome/blob/main/entries/kernels/tensixfuse.json) registry of community projects. The fused TT-Lang kernel starts from TT-Lang's own matmul tutorial and the multicast version from tt-lang's SUMMA benchmark; [Builds on](#builds-on) says what came from where and what's mine.

## In brief

- **Fusion always saves the same 8 MiB. What changes is how much that's worth.** Fusing matmul, bias and ReLU at 1024³ removes two intermediate tensors, 8 MiB written to DRAM and read back. Next to a naive matmul that's 6% of all traffic. Once the matmul reads each input only once, it's half. TT-Lang's functional simulator agreed with my analytic model tile for tile on all 48 runs.
- **Data reuse mattered more than fusion.** Bigger blocks cut DRAM traffic from 132 to 20 MiB, and multicasting input panels over the NoC on a 64-core grid took it to 8 MiB, 16.5x below where I started.
- **Block-float weights on GPT-2.** bfloat8_b cuts the MLP weights 47% and the worst of the 12 blocks still has PCC 0.99982. bfloat4_b cuts 72% and the worst block falls to 0.961. If you keep one layer in bfloat8_b, it should be `c_fc`, not `c_proj`, which is the opposite of what I expected.
- **The format, not the chip's arithmetic, is where the error comes from.** A NumPy copy of tt-metal's block-float packer (bit-exact with it on a million values) predicts the simulated chip's PCC to the 5th decimal. The same reasoning says LoFi math fidelity should cost nothing with bfloat4_b weights, whose 3 mantissa bits fit in a single multiplier pass, and it measures that way.
- **Whole-model accuracy, against other formats.** With every linear weight in bfloat8_b, GPT-2 small's WikiText-2 perplexity moves 0.24%. At 4 bits, bfloat4_b costs 21% where NVFP4 costs 6.7% at the same 4.5 bits per weight, and about half of that gap is which axis the 16-value groups run along. On Qwen2.5-0.5B, tt_transformers' performance recipe (FF1/FF3 in bfloat4_b) costs 8.0%, its default (all bfloat8_b) 0.13%.
- **My own model on a simulated Blackhole.** GPU-CorruptNet's head runs as one fused `sigmoid(x @ W + b)`. With bfloat8_b weights, 2 of 4,160 frames get a different label set from PyTorch, and macro-F1 is unchanged (0.9112 seen, 0.8756 unseen).
- **A TT-Metalium program by hand.** Fused BatchNorm + ReLU on a real ResNet-50 layer's input, 3.2 million values over 8 cores, with host code and reader, compute and writer kernels. The fp32 version lands every value within one bf16 step of NumPy, with identical results on simulated Wormhole and Blackhole.

## How a tile moves

Every op on a Tensix core is the same loop. Each core has five small RISC-V processors: two move data (reader and writer) and three drive the compute engine (unpack, math, pack). They hand 32×32 tiles to each other through circular buffers in the core's 1.4 MiB of SRAM (L1), and the cores talk to DRAM and to each other over two NoCs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/tile-trip-dark.svg">
  <img alt="A tile's trip: DRAM, NoC read, reader, a circular buffer in L1, unpack, math on the FPU or SFPU, pack, another circular buffer, writer, NoC write, back to DRAM. Fused, the bias add and the activation happen in the math stage." src="docs/img/tile-trip-light.svg" width="100%">
</picture>

Unfused, each op is one trip around this loop, and each op's output goes back to DRAM before the next op reads it in again. Fused, the bias add and the activation happen in the math stage before pack, so those intermediates never leave the core.

## Results

Every table here is regenerated from the JSON in [`results/`](results) by `python bench/report.py`, which writes [`docs/RESULTS.md`](docs/RESULTS.md) with all sizes and every run. Each section says which simulator it came from. CI re-runs a study whenever its code or model changes; across runs every table holds to the precision shown, except that GPT-2 PCCs can move in the 6th decimal (and once in the 5th), because the GPT-2 activations are captured with PyTorch on the runner's CPU, which isn't bit-reproducible across CPU types.

### 1. Fusion and data movement

`y = relu(a @ b + c)`, M = K = N = 1024, bf16, on TT-Lang's simulator (tt-lang-sim 1.1.6). "Unfused" is the same matmul followed by separate add and ReLU operations, written in TT-Lang with the same blocking, so fusion is the only difference. DRAM traffic is tiles read plus tiles written, from the simulator's trace, at 2 KiB per tile.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/staircase-dark.png">
  <img alt="Bar chart of DRAM traffic at 1024 cubed: fused traffic falls from 132 MiB with 1x1 tile blocks to 68, 36 and 20 MiB with bigger blocks on one core, 36 MiB on 64 cores, and 8 MiB with multicast, the compulsory floor. Unfusing adds the same 8 MiB to every bar." src="docs/img/staircase-light.png" width="100%">
</picture>

| Matmul data reuse | Cores | Unfused | Fused | Fusion saves |
|---|---|---|---|---|
| 1×1 tile blocks | 1 | 140 MiB | 132 MiB | 6% |
| 2×2 tile blocks | 1 | 76 MiB | 68 MiB | 11% |
| 4×4 tile blocks | 1 | 44 MiB | 36 MiB | 18% |
| 8×8 tile blocks | 1 | 28 MiB | 20 MiB | 29% |
| 4×4 blocks per core, 8×8 grid | 64 | 44 MiB | 36 MiB | 18% |
| same, with A and B multicast | 64 | 16 MiB | 8 MiB | 50% |

On 64 cores without multicast, every core reads its own copy of the A and B blocks it needs, so each panel of A is read once per column of cores. With multicast, one core per row reads A and one per column reads B, and they pass the blocks to the rest of their row or column over the NoC:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/summa-dark.svg">
  <img alt="A 4 by 4 grid of cores. The cores in the first column read rows of A blocks from DRAM and multicast them along their rows; the cores in the first row read columns of B blocks and multicast them down their columns. At 1024 cubed on 64 cores, fused traffic drops from 36 MiB to 8 MiB." src="docs/img/summa-light.svg" width="100%">
</picture>

**What that could be worth in time (modelled, not measured).** ttsim doesn't model time, but Tenstorrent publishes peaks for the Wormhole n150: 74 TFLOPS at 16 bits over 72 cores, and 288 GB/s of DRAM bandwidth ([specifications](https://docs.tenstorrent.com/aibs/wormhole/specifications.html)). On the 64 cores this grid uses, the 1024³ multiply needs at least 33 µs of compute. Moving 36 MiB needs at least 131 µs, so without multicast the kernel would be DRAM-bound about 4× over. With multicast, 8 MiB needs 29 µs, just under the compute time: DRAM stops being the limit below about 9 MiB, so the floor is about where more reuse stops paying. Fusion's 8 MiB is worth the most there: unfused with multicast is 16 MiB and 58 µs, fused is 33 µs. These are bounds from two published numbers, with no NoC contention, dispatch or kernel overhead, and no claim that the kernel reaches either peak.

**Through the real compiler: not yet.** The functional simulator runs a kernel as Python, so it accepts things the TT-Lang compiler won't lower. A CI job ([`ttlang-compile.yml`](.github/workflows/ttlang-compile.yml)) installs the tt-lang 1.1.6 compiler with the ttnn, ttsim and SFPI versions it needs and runs the same kernels on a simulated Blackhole. The toolchain works: TT-Lang's own matmul tutorial (step 1, at 64³) compiles, and its generated kernels run on the simulated chip with PCC 0.999995. Mine don't compile yet. The fused variants stop with `Cannot Compare Non-Integer Values` and the unfused ones with `Cell is empty`, both at the `@ttl.operation`, most likely from the Python-level flags (`if fused:`) and the closure variables that only exist in the fused variant, which the functional simulator just runs as Python. Rewriting the kernels in the form the compiler takes is next.

Before running anything I wrote an analytic model of how many tiles each kernel should move ([`traffic.py`](src/tensixfuse/traffic.py)). The simulator agrees with it tile for tile on all 48 runs: 4 sizes from 256 to 2048, 6 reuse levels, fused and unfused, and [`tests/test_sim_traffic.py`](tests/test_sim_traffic.py) keeps it that way. Every run that computed real values matched PyTorch with PCC 1.000000. That's expected, because the functional simulator does its math in fp32: it checks that the right tiles went to the right place, not the chip's arithmetic. The chip's arithmetic comes from ttsim, in everything below.

### 2. Block-float weights on GPT-2's MLPs

GPT-2 small has 12 MLP blocks: `c_fc` (768 to 3072), GELU, then `c_proj` (3072 to 768). I captured each block's real input on the first 128 tokens of the WikiText-2 test set and ran the blocks with TTNN on simulated Blackhole at HiFi4, weights in each format and activations in bf16. PCC is against the fp32 PyTorch block. The matmuls here are TTNN's own `ttnn.linear` kernels, not mine: what I built is the capture, the block-float emulator the results are checked against, and the comparison.

| Weights | MLP weights, 12 blocks | PCC, median block | PCC, worst block |
|---|---|---|---|
| bf16 | 108.0 MiB | 0.99999 | 0.99997 |
| bfloat8_b | 57.4 MiB | 0.99987 | 0.99982 |
| bfloat4_b | 30.4 MiB | 0.97698 | 0.96066 |
| bfloat4_b `c_fc`, bfloat8_b `c_proj` | 43.9 MiB | 0.98623 | 0.97500 |
| bfloat8_b `c_fc`, bfloat4_b `c_proj` | 43.9 MiB | 0.99050 | 0.98332 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/blockfloat-dark.svg">
  <img alt="A 32x32 tile is four 16x16 faces; the 16 values in a face row share one exponent. Per tile: bfloat16 2,048 bytes; bfloat8_b 1,088 bytes (64 exponent bytes plus signs and 7-bit mantissas); bfloat4_b 576 bytes (64 exponent bytes plus signs and 3-bit mantissas)." src="docs/img/blockfloat-light.svg" width="100%">
</picture>

bfloat8_b and bfloat4_b store one 8-bit exponent for every 16 values, and each value keeps a sign and 7 or 3 bits of mantissa (hidden bit included) measured against the largest value in its group. A 32×32 tile is 1,024 values plus 64 shared exponents: 1,088 bytes for bfloat8_b and 576 for bfloat4_b, against 2,048 for bf16. In bfloat4_b, a value 16x or more smaller than the largest in its group is always stored as zero, and between 15 and 19% of the weights in every layer here end up as zero.

Math fidelity is the other knob: the matrix engine multiplies in 1 to 4 passes (LoFi, HiFi2, HiFi3, HiFi4), trading mantissa bits for speed. On block 6, the one bfloat4_b hurts most:

| Weights | LoFi | HiFi2 | HiFi3 | HiFi4 |
|---|---|---|---|---|
| bf16 | 0.99841 | 0.99993 | 0.99998 | 0.99998 |
| bfloat8_b | 0.99882 | 0.99979 | 0.99982 | 0.99982 |
| bfloat4_b | 0.96070 | 0.96070 | 0.96066 | 0.96066 |

With bfloat4_b weights, LoFi is as good as HiFi4: the weights only carry 3 mantissa bits, and the first pass already uses them all. With bf16 weights LoFi visibly costs, and HiFi3 already matches HiFi4. ttsim can't say how much faster LoFi would run, but one pass instead of four is the point of it.

**The whole model: WikiText-2 perplexity.** PCC per block doesn't say what the model loses end to end, so I also measured GPT-2 small's perplexity on the whole WikiText-2 test set (287,644 tokens, 280 windows of 1,024) with its weights rounded the way each format stores them: bfloat8_b and bfloat4_b through the same bit-exact emulation of tt-metal's packer, next to OCP's MXFP8 and MXFP4, NVIDIA's NVFP4, and INT4 with one scale per 128 weights. It's weight-only (activations stay fp32), on CPU in CI ([`bench/perplexity.py`](bench/perplexity.py)); the write-up is in [`docs/PERPLEXITY.md`](docs/PERPLEXITY.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/perplexity-dark.png">
  <img alt="Bar chart of GPT-2 small's WikiText-2 perplexity above fp32 (29.95) with every linear weight rounded: bfloat16 +0.0%, MXFP8 +0.1%, bfloat8_b +0.2%, NVFP4 +6.7%, INT4 g128 +11.0%, bfloat4_b with groups along K +14.0%, MXFP4 +15.9%, bfloat4_b +21.0%." src="docs/img/perplexity-light.png" width="100%">
</picture>

| Weights rounded | bfloat8_b | MXFP8 | bfloat4_b | NVFP4 | INT4 g128 | MXFP4 |
|---|---|---|---|---|---|---|
| bits per weight | 8.5 | 8.25 | 4.5 | 4.5 | 4.16 | 4.25 |
| MLP only | +0.06% | -0.43% | +8.4% | +3.1% | +5.4% | +2.4% |
| every linear weight | +0.24% | +0.11% | +21.0% | +6.7% | +11.0% | +15.9% |

At 8 bits nothing matters: no 8-bit format costs more than 0.24% over fp32's 29.95. At 4 bits, bfloat4_b is the weakest format here, at the same 4.5 bits as NVFP4: NVFP4's 16-value groups get a fine-grained FP8 scale and floating-point elements, where bfloat4_b's get a power-of-two exponent and 3 bits of mantissa, so small neighbors of a large weight round to zero. Part of the gap is the axis: Tenstorrent's tiles group 16 *output* features per exponent, and the same format grouped along the input dimension instead (the weight stored transposed) cuts the damage from +21.0% to +14.0%. The mixed MLP splits agree with the per-block PCCs: bfloat8_b `c_fc` with bfloat4_b `c_proj` costs 2.6%, the other way around 5.0%. MXFP8 on the MLP came out slightly below fp32, which I read as luck, not a gain.

**tt_transformers' recipes on a Llama-style model.** Tenstorrent's `tt_transformers` keeps every weight in bfloat8_b by default, and its performance setting moves FF1 and FF3 (the gate and up projections) to bfloat4_b, except on Qwen2.5-7B, which its config says degrades. On Qwen2.5-0.5B, a small model of that family with a gated SiLU MLP, the same emulation gives ([`bench/perplexity_llama.py`](bench/perplexity_llama.py)):

| Recipe | Bits per weight | Perplexity | vs fp32 |
|---|---|---|---|
| fp32 (the checkpoint is bf16, so bf16 is the same) | 32 | 14.651 | |
| default: every linear weight bfloat8_b | 8.5 | 14.669 | +0.13% |
| performance: FF1/FF3 bfloat4_b, the rest bfloat8_b | 6.16 | 15.816 | +7.96% |
| every MLP weight bfloat4_b | 4.99 | 16.749 | +14.3% |

So the small Qwen degrades under the performance setting too, by 8% in perplexity, while the default costs 0.13%.

### 3. GPU-CorruptNet's head on a simulated Blackhole

GPU-CorruptNet is a ResNet-50 that flags GPU rendering artifacts in frames, multi-label over 10 artifact types. Its head is one linear layer from 2,048 pooled features to 10 logits, read through a sigmoid with a `>= 0.5` threshold. I ran the backbone on CPU once for the 1,560 seen-content and 2,600 unseen-content test frames (recomputing the head from those features gives GPU-CorruptNet's published 0.911 and 0.876 macro-F1), and ran only the head on the simulator, fused, with the calibration temperature folded into the weights.

| Weights | Label sets match PyTorch | Frames that differ | Logit PCC | Macro-F1, seen | Macro-F1, unseen |
|---|---|---|---|---|---|
| PyTorch fp32 | | | | 0.9112 | 0.8756 |
| bf16 | 99.93% | 3 | 0.99996 | 0.9112 | 0.8754 |
| bfloat8_b | 99.95% | 2 | 0.99996 | 0.9112 | 0.8756 |
| bfloat4_b | 99.47% | 22 | 0.99984 | 0.9113 | 0.8768 |

The frames that flip are the ones whose logits sit right at zero, so bfloat4_b's slightly higher F1 is luck, not an improvement.

### 4. TT-Metalium C++: fused BatchNorm + ReLU

At inference time BatchNorm is a per-channel scale and shift, so the step after each conv in ResNet-50 is `y = relu(x * s + b)`. I wrote it as a TT-Metalium program in [`kernels/metalium/bn_relu`](kernels/metalium/bn_relu): host code that sets up the device, DRAM buffers, circular buffers and runtime arguments, and three kernels per core. The reader pulls activation tiles over the NoC (and each column's scale and shift once), compute does the math, and the writer sends finished tiles back. The input is the real input to GPU-CorruptNet's `layer4.2.bn3` for 32 frames, channels last: 1,568 × 2,048 values, 3,136 tiles over 8 cores. It's built in CI against tt-metal 0.79.0's released SDK packages and run on simulated Wormhole and Blackhole.

One honest caveat: in ResNet-50, `bn3` is followed by the residual add, not by ReLU, so BN + ReLU on its input is a stand-in, and ReLU zeros 93% of its outputs. Zeros match trivially, so the rates that count are the ones on the 6.6% of outputs (211,954) that stay nonzero:

| Kernel | How | Bit-exact, all | Bit-exact, nonzero | Within 1 bf16 step, nonzero | PCC |
|---|---|---|---|---|---|
| `fpu` | matrix engine, row broadcast, bf16 intermediate in L1 | 95.40% | 30.28% | 64.68% | 0.999986 |
| `sfpu` | vector engine, everything in fp32 registers, one rounding at the end | 99.78% | 96.70% | 100.00% | 1.000000 |

`layer4.2.bn2` is followed by ReLU in the real network; [`model/corruptnet_bn2_export.py`](model/corruptnet_bn2_export.py) exports it for the same 32 frames (75% of its outputs are still zero after ReLU), and running the kernels on it is the next step.

Both give identical results on Wormhole and Blackhole, and the same bits on every run. "NumPy" here is the same bf16 inputs in fp32, rounded to bf16 once.

### 5. Infrastructure

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/infra-dark.svg">
  <img alt="A push starts GitHub Actions: lint and tests on Linux and macOS, the ttsim matrix, the C++ build, the studies, the sharded sweep, the JobSet on kind and the Ansible role. They produce results, tested images pushed to GHCR, and the PCC and DRAM-tile checks that gate pull requests." src="docs/img/infra-light.svg" width="100%">
</picture>

| Piece | What it does | Numbers |
|---|---|---|
| CI ([`ci.yml`](.github/workflows/ci.yml)) | Lint and 72 tests (model, block-float and MX/NVFP4/INT4 emulation, simulator-vs-model, regression gate) on Linux and macOS | on every push |
| ttsim matrix ([`ttsim.yml`](.github/workflows/ttsim.yml)) | 6 jobs, Wormhole and Blackhole × bf16, bfloat8_b and bfloat4_b, with a pinned simulator and PCC floors in [`bench/baselines.json`](bench/baselines.json) | 7 tests per chip, split by format |
| Regression gate | A PR fails if a PCC drops below its floor or a kernel's DRAM tile count goes above its baseline | |
| C++ build ([`metalium.yml`](.github/workflows/metalium.yml)) | Installs tt-metal's SDK packages, builds the program with CMake, runs both kernels on both chips | 3.2M values checked per run |
| Containers ([`docker/`](docker)) | Two pinned images, each tested before it's pushed to GHCR: `tensixfuse` with ttnn, SFPI and ttsim for both chips, and `tensixfuse-sweep` with TT-Lang's simulator | 2.1 GB and 1.1 GB; 7 ttsim tests per chip in under 20 s |
| Sharded sweep ([`sweep.yml`](.github/workflows/sweep.yml)) | The 48-run fusion sweep split over 8 GitHub-hosted runners with cost-balanced shards, merged and checked | 184 s on one runner, 76 s across 8 |
| Kubernetes ([`deploy/helm`](deploy/helm/tensixfuse-sweep)) | A Helm chart that runs the sweep as a [JobSet](https://github.com/kubernetes-sigs/jobset), one simulator pod per shard; CI runs it on a kind cluster | 4 pods finish in under 25 s on kind; all 24 runs match the model |
| Ansible ([`deploy/ansible`](deploy/ansible)) | `site.yml` turns a bare Ubuntu 24.04 host into a simulator node; `health_check` runs known-answer kernels on every simulated chip and fails the play on a mismatch. Tested with Molecule | under 100 s on a fresh VM; a second run changes nothing |

The simulator and the SFPI compiler are pinned to the versions tt-metal 0.79.0 itself pins, with checksums ([`ttsim-version`](ttsim-version), [`sfpi-version`](sfpi-version)). When the simulator changes, I want to know that the simulator changed, not wonder whether my kernel did.

## What I found

**Fusion's savings depend on everything else.** Fusing removes the same two intermediates at every row of the fusion table: 8 MiB at 1024³. With 1×1 blocks the matmul re-reads A and B so many times that 8 MiB is 6% of the traffic, and fusion barely shows. I expected fusion to be the headline. It turned out to matter most after the boring work of data reuse was done, when it's half of everything left.

**Spreading the work over 64 cores doesn't cut DRAM traffic by itself.** Each core still fetches its own A and B panels, so 64 cores with 4×4 blocks move exactly as much as one core with 4×4 blocks: 36 MiB. What helps is multicast. One core per row reads its A panel from DRAM and sends it over the NoC to the other seven in the row, one core per column does the same for B, and every input tile leaves DRAM once.

**L1 sets the block size.** With every buffer double-buffered, an 8×8 block takes 1 MiB of buffers and a 16×16 block takes 3,670,016 bytes, and the simulator warns that this is over its per-core limit of 1,466,368 bytes (1,432 KiB).

**I guessed the wrong layer.** Tenstorrent's `tt_transformers` performance mode puts the up and gate projections (FF1, FF3) in bfloat4_b with LoFi and keeps the down projection (FF2) in bfloat8_b. For GPT-2 that's bfloat4_b `c_fc` with bfloat8_b `c_proj`, so I expected that split to win. It's the worse one for GPT-2 small: at the same 43.9 MiB, keeping `c_fc` in bfloat8_b instead gives a worst block of 0.983 against 0.975. Llama's MLP is gated and uses SiLU, so this says nothing about Llama, only that the split is worth measuring per model. On Qwen2.5-0.5B, which is gated, the recipe as tt_transformers ships it costs 8% in perplexity (section 2); I haven't tried the opposite split there.

**The error is in the format, not the arithmetic.** I wrote the block-float packer again in NumPy from tt-metal's source and checked it against ttnn on about a million values, outliers and rounding ties included: zero mismatches. Running the GPT-2 blocks on CPU with weights rounded that way gives the same PCC as the simulated chip to the 5th decimal in every configuration. So at HiFi4 the chip adds nothing measurable; the loss is the format itself. That's also why LoFi is free for bfloat4_b weights.

**Padding is easy to get wrong in a multi-label head.** 10 outputs don't fill a 32-wide tile, so the weights are padded to 32 columns. On the device, the padded columns come out as exactly sigmoid(0) = 0.5, and with a `>= 0.5` threshold that's a prediction: thresholding before slicing them off gives 22.92 labels per frame instead of 0.92.

**My first BatchNorm kernel rounded twice.** The FPU version stores `x * s` in L1 as bf16 before adding `b`, so it rounds twice, and the matrix engine reads operands at 16 or 19 bits. Against a reference that rounds the same way it matches 98%, not 95%, which pinned down where the rest came from. The SFPU version keeps everything in fp32 registers and lands every element within one bf16 step. On silicon the FPU version should be the faster one; ttsim can't say by how much.

**Getting the stack to run took more than the kernels did.** The `ttnn` wheel doesn't bundle SFPI, the RISC-V compiler tt-metal uses to build kernels at runtime, so that's pinned and installed separately. The SDK's `libtt_metal.so` needs Tenstorrent's OpenMPI build with ULFM fault tolerance, which the packages don't pull in, so linking failed on `MPIX_Comm_revoke` until I installed it. A slim Ubuntu image also needs `libatomic` and GCC's math libraries for SFPI's compiler. TT-Lang's simulator and the real `ttnn` can't share a Python environment either: with both installed, the simulator picks up the real `ttnn` and the kernels fail, which I found when the Kubernetes pods crashed. So the Dockerfile has two targets. And tt-lang-sim 1.1.6 replaces any kernel error with a `FileNotFoundError` for `ttl/ttl/diagnostics.py`, a file the wheel doesn't ship; [`kernels/ttlang/simfix.py`](kernels/ttlang/simfix.py) prints the real message. Its help text also says the per-core L1 limit is 1,336 KiB, while the code enforces 1,432. I reported both upstream, with repros and proposed fixes: [tt-lang#1192](https://github.com/tenstorrent/tt-lang/issues/1192) and [tt-lang#1193](https://github.com/tenstorrent/tt-lang/issues/1193).

**One run sets the floor for sharding.** Spread over 8 runners, the sweep is bounded by its largest single run, the unblocked 2048³ dry run, which takes about a minute on a hosted runner by itself. Balancing shards by predicted cost keeps the other big runs off that shard, but nothing gets the slowest shard below that one run.

## Builds on

- **TT-Lang's matmul tutorial** (steps 1 to 4). The fused `relu(a @ b + c)` kernel in [`kernels/ttlang/ops.py`](kernels/ttlang/ops.py) is that tutorial's kernel, generalized to other block sizes, grids and activations. The unfused baseline and the elementwise ops are mine.
- **tt-lang's SUMMA benchmark** ([`benchmarks/matmul/summa_kernel.py`](https://github.com/tenstorrent/tt-lang/blob/main/benchmarks/matmul/summa_kernel.py)). My multicast variant uses its row and column `PipeNet` pattern, with the fused bias and activation added. tt-metal's [`matmul_multicore_reuse_mcast`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast) example does the same thing in TT-Metalium.
- **tt-metal's `eltwise_binary` and `eltwise_sfpu` programming examples**, for the host and kernel structure of [`kernels/metalium/bn_relu`](kernels/metalium/bn_relu), and tt-metal's FP32 accuracy guide for its SFPU version.
- **tt-metal's block-float packer** (`blockfloat_common.cpp` at v0.79.0), which [`blockfloat.py`](src/tensixfuse/blockfloat.py) reimplements in NumPy.
- **TTNN**, whose `ttnn.linear` runs the GPT-2 and GPU-CorruptNet studies on ttsim.
- **The other formats' definitions**, for [`formats.py`](src/tensixfuse/formats.py): the OCP Microscaling (MX) spec v1.0 for MXFP8 and MXFP4, NVIDIA's two-level NVFP4 scheme, and the INT4 group-of-128 layout GPTQ and AWQ store weights in (without their calibration).
- **tt_transformers' precision settings** (`models/tt_transformers/tt/model_config.py` in tt-metal), the recipes the Qwen study tests, and **Tenstorrent's published card specs** for the modelled roofline.

What's mine: the fusion-vs-reuse measurement and the analytic traffic model it's checked against, the block-float emulator and the numerics and perplexity studies built on it, the BatchNorm program, and the CI, container, Kubernetes and Ansible setup.

## Limitations

- No silicon, so no device latency or throughput. DRAM traffic here counts tiles. It says nothing about bandwidth, NoC contention, or how much compute hides behind data movement.
- TT-Lang's functional simulator computes with PyTorch in fp32 and stores bfloat8_b as bf16, so it can check data movement but not numerics. Every accuracy number comes from ttsim.
- GPU-CorruptNet's head is small (20,490 parameters), and its backbone ran on CPU. GPT-2 small is a small model, and bigger LLMs have larger outliers, so bfloat4_b would likely do worse on them.
- The sweep's fan-out speedup is real but modest, because the sweep is short and one run dominates it.
- The perplexity numbers are CPU emulation of the weights only: activations stay fp32, and math fidelity (LoFi) isn't modelled. The other formats are plain round-to-nearest, without the calibration GPTQ, AWQ or NVIDIA's recipes would add.
- My TT-Lang kernels have only run in the functional simulator. They don't get through the TT-Lang compiler yet (TT-Lang's own tutorial kernel does, in the same CI job).

## What I would do next

- Run the same kernels on a real Wormhole or Blackhole card and see how much of the DRAM saving turns into time, and what the SFPU BatchNorm costs next to the FPU one.
- Write the bfloat8_b linear layer myself in TT-Metalium (matmul with bfloat8_b weights, bias and GELU in one kernel) and match `ttnn.linear` bit for bit, instead of comparing against TTNN's kernel.
- Fuse the whole MLP (`c_fc`, GELU, `c_proj`) so the 128 × 3072 hidden activations stay in L1 across cores and never touch DRAM.
- Put LoFi into the perplexity table by running the MLPs on ttsim for a slice of WikiText-2. The CPU emulation can't model math fidelity.
- Move to tt-metal 0.81.1 (it pins ttsim v1.12.1 and SFPI 7.89.0) and re-run every study, as its own change.
- Finish the tensor-parallel GPT-2 MLP on ttsim's 2-chip N300 build ([`bench/tp_mlp_n300.py`](bench/tp_mlp_n300.py)). Both simulated chips come up and the Ethernet fabric between them initializes, but host writes to the remote chip time out with tt-metal 0.79.0, which matches ttsim's note that multichip support isn't fully merged into mainline yet. My first attempt also stopped on ttsim's strict checks, with UndefinedBehavior in an unpacker reading and writing Float32, which I didn't get to the bottom of.

## Reproduce

A simulated Blackhole chip in a container: the health check, then the ttsim tests ([`docs/demo.tape`](docs/demo.tape), recorded in CI):

![Terminal: docker run of the TensixFuse image runs tensixfuse-doctor on a simulated Blackhole, then pytest runs the 7 ttsim tests, which pass in about 22 seconds](docs/img/demo.gif)

**TT-Lang simulator (macOS or Linux):**

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.11 or newer
pip install -e ".[dev]"                                  # includes tt-lang-sim 1.1.6
pytest                                                   # 72 tests
python bench/fusion.py                                   # the fusion sweep, ~3 min
python bench/report.py                                   # docs/RESULTS.md
```

**ttsim (Linux x86_64), in the container or directly:**

```bash
docker run --rm -e TENSIXFUSE_ARCH=wormhole ghcr.io/tanaymihani/tensixfuse tensixfuse-doctor

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ttnn==0.79.0 && pip install --no-deps -e .
bash scripts/install_sfpi.sh                  # pinned SFPI, checksum checked
eval "$(bash scripts/setup_ttsim.sh blackhole)"
pytest -m ttsim tests/test_ttsim.py
python model/gpt2_capture.py && python bench/quant_gpt2.py
bash scripts/fetch_data.sh && python bench/corruptnet_head.py
```

**Perplexity (CPU, any OS):**

```bash
pip install torch transformers datasets && pip install -e .
python bench/perplexity.py --configs fp32,all:bfloat8_b,all:bfloat4_b,all:nvfp4
python bench/perplexity_llama.py --configs fp32,attn=bfloat8_b+ff13=bfloat4_b+ff2=bfloat8_b
```

**The C++ program** needs tt-metal 0.79.0's SDK packages; [`metalium.yml`](.github/workflows/metalium.yml) has the exact install and build steps.

**The sweep on Kubernetes** (with the JobSet controller installed):

```bash
helm install sweep deploy/helm/tensixfuse-sweep --set shards=8
kubectl wait --for=condition=Completed jobset/sweep --timeout=30m
```

**A simulator node with Ansible:**

```bash
cd deploy/ansible && ansible-playbook site.yml   # then cat /var/lib/tensixfuse/health.json
```

## Layout

```
kernels/ttlang/        fused and unfused matmul, multicast variant, elementwise ops, runner
kernels/metalium/      bn_relu: host program, reader/writer kernels, FPU and SFPU compute kernels
src/tensixfuse/        traffic model, block-float and MX/NVFP4/INT4 emulation, metrics, simulator runner, doctor
model/                 GPU-CorruptNet exports, GPT-2 MLP capture
bench/                 fusion sweep, GPT-2, perplexity and head studies, BatchNorm check, TT-Lang compiler run, merge, report
tests/                 model, emulation, simulator-vs-model, regression gate, ttsim
pins/                  ttsim and SFPI pins for the tt-lang 1.1.6 compiler job (its ttnn is 0.74.0)
docker/                pinned simulator image
deploy/helm/           JobSet sweep chart
deploy/ansible/        site.yml, roles/ttsim_node, roles/health_check, molecule/
.github/workflows/     ci, ttsim, metalium, docker, sweep, kubernetes, ansible, studies, perplexity, ttlang-compile, scaleout
results/               every result the tables come from
docs/RESULTS.md        all the tables, generated
docs/make_figures.py   the figures in docs/img, drawn from results/
```

## License

MIT
