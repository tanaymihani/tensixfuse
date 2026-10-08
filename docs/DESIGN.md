# Design notes

For each decision: what I picked, what it costs, what could break, and how I'd find out.

## Measuring data movement

**Choice.** DRAM traffic comes from TT-Lang's functional simulator, which logs every tile copy with where the tensor lives, checked against an analytic model ([`traffic.py`](../src/tensixfuse/traffic.py)). ttsim can't do this job: it's bit-exact, but it doesn't count bytes and its cycle counters aren't meaningful.

**Cost.** The functional simulator does its math in PyTorch fp32, so its PCC says nothing about the chip's arithmetic. That's why every accuracy number comes from ttsim instead.

**What could break.** A tt-lang-sim update could change how copies are traced, or a kernel edit could quietly add reads. [`tests/test_sim_traffic.py`](../tests/test_sim_traffic.py) requires the measured count to equal the model exactly, and [`tests/test_regression_gate.py`](../tests/test_regression_gate.py) fails if a count goes above its baseline even when the model was changed to match.

## A fair unfused baseline

**Choice.** The unfused chain (matmul, add, ReLU) is written in TT-Lang with the same blocking as the fused kernel. Comparing against TTNN's ops would mix in a different matmul implementation, and then the difference wouldn't be fusion.

**What it shows.** The only difference between the variants is two intermediate tensors, 8 MiB at 1024³, in every row of the table.

## Block size and L1

**Choice.** Output-stationary blocks, like the TT-Lang tutorial: each block of output tiles streams its A row panel and B column panel. Bigger blocks re-read less.

**Limit.** Each buffer is double-buffered, so a block of bm × bn with k-step bk needs 2 × (bm·bk + bk·bn + 3·bm·bn) tiles. 8×8×4 is 1 MiB; 16×16×4 is 3.5 MiB, over the simulator's 1,432 KiB per core. Single-buffering C, or a smaller k-step, would buy room for 8×16.

## Multicast

**Choice.** SUMMA-style: the core in column 0 of each row reads A from DRAM and sends it along the row with a `PipeNet`; row 0 does the same for B down each column. Every core in the row, the sender included, receives through the pipe.

**What could break.** It needs the output blocks to split evenly over the grid ([`MatmulConfig`](../src/tensixfuse/traffic.py) refuses otherwise). With more than one block per core, the sender re-sends A for each block column, which the model accounts for (the 2048³ grid row isn't at the floor for that reason).

## Accuracy from ttsim, with an emulation next to it

**Choice.** Every PCC, label agreement and F1 comes from ttsim, which is designed to match silicon bit for bit. Next to each GPT-2 number is the same computation on CPU with weights rounded the way tt-metal packs them ([`blockfloat.py`](../src/tensixfuse/blockfloat.py)).

**Why.** If the two agree, the loss is the format. If they disagree, it's the chip's arithmetic (or a bug). They agreed to the 5th decimal everywhere at HiFi4.

**What could break.** If tt-metal changes its packer (rounding, grouping), the emulation goes stale. [`tests/test_ttsim.py`](../tests/test_ttsim.py) checks it bit for bit against ttnn on every ttsim run, outliers included.

## GPU-CorruptNet's head

**Choices.** The calibration temperature is folded into W and b so the calibrated head is still one op. The 10 outputs are padded to 32 and sliced off **before** thresholding: zero-padded columns come out as exactly 0.5 on the device, and the model's threshold is `>= 0.5`. The batch is the two test splits together, 4,160 frames, which is exactly 130 tile rows.

**What could break.** Thresholding before slicing adds 22 labels to every frame. A test runs that case on ttsim and checks the padded columns really are 0.5.

## Two BatchNorm + ReLU kernels

**FPU (`bn_relu.cpp`).** Row broadcast on the matrix engine, `x * scale` into an L1 buffer as bf16, then `+ shift` and ReLU. It uses the fast engine, but it rounds twice and the matrix engine reads operands at 16 or 19 bits: 95.4% bit-exact, 97.7% within one bf16 step.

**SFPU (`bn_relu_sfpu.cpp`).** Everything in fp32 destination registers on the vector engine, one rounding when packing: 99.78% bit-exact and every element within one step. The cost is that scale and shift come as full tiles (the SFPU ops have no broadcast), and on silicon the vector engine would be slower; ttsim can't say how much.

**Both** read each column's scale and shift once and reuse them for the whole column, and give the same bits on every run.

## Pinning

ttnn 0.79.0, ttsim v1.10.8 and SFPI 7.78.0 are the versions tt-metal 0.79.0 pins for itself, each with a checksum. The N300 cluster file for the (manual, not yet working) two-chip run comes from the tt-umd commit tt-metal pins. When a number moves, it should be because the code moved. A simulator bump is its own commit.

## CI tiers

- Every push: lint and the 54 fast tests on Linux and macOS (functional simulator, emulation, model, regression gate).
- ttsim: 6 jobs over simulated Wormhole and Blackhole and the three weight formats, with PCC floors.
- C++: built against Tenstorrent's released SDK packages, both kernels on both chips, every element checked.
- Studies (GPT-2, CorruptNet head) run when their code changes and write the JSON the tables come from. The two-chip N300 run is manual until host writes to the remote chip work.

## Sharding the sweep

Shards are assigned longest-first to the least-loaded shard, from a cost estimate based on the number of block matmuls. It keeps big runs apart, but it can't split one: the slowest shard is never faster than the largest single run (the unblocked 2048³ dry run, about a minute on a hosted runner).

## Provisioning

The Ansible roles are idempotent: a second run changes nothing, which CI checks on a fresh VM and Molecule checks in a container. The health check runs known-answer kernels on every simulated chip and fails the play on any mismatch; its JSON report holds timings, so writing it never counts as a change.
