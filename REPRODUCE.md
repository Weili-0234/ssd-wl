# Reproducing SSD (Speculative Speculative Decoding) Results

This guide walks through reproducing the Qwen-3-32B experiments from the
[SSD paper](https://arxiv.org/abs/2603.03251) (ICLR 2026). It is written for
collaborators who may not be familiar with speculative decoding.

> **Note**: The original SSD repo does not include a benchmark comparison script.
> We added `bench/run_qwen_bench.sh` and `bench/analyze_results.py` to automate
> running AR / SD / SSD experiments and analyzing the results.

> **Note**: All commands in this guide assume your terminal is **directly
> attached to GPUs** (i.e., you are on a SLURM compute node, not a head/login
> node). If using SLURM, first get an interactive session or submit via `srun` /
> `sbatch`.

---

## 1. Background: What Are We Benchmarking?

### Autoregressive Decoding (AR)

Standard LLM inference generates one token at a time. Each token requires a full
forward pass of the target model (Qwen-3-32B, ~32 billion parameters). On 4
H100 GPUs this takes about 10ms per token, giving ~97 tok/s.

### Speculative Decoding (SD)

A small **draft model** (Qwen-3-0.6B, ~600M parameters) quickly guesses the
next K tokens. Then the large target model **verifies** all K guesses in a
single forward pass (the same cost as generating 1 token). Correctly guessed
tokens are accepted; on the first wrong guess, the target samples a corrected
"bonus token" and the round ends.

This works because verification is cheap (1 forward pass regardless of K), while
the draft's K forward passes are fast due to its small size. Net result: multiple
tokens per target forward pass.

In SD, draft and target run on the **same GPUs**, taking turns sequentially.

### Speculative Speculative Decoding (SSD / "Saguaro")

SSD puts the draft model on a **separate GPU** so it can speculate **in parallel**
with the target's verification. While the target verifies round T, the draft
predicts what the verification outcome will be and pre-computes speculations for
round T+1. These pre-computed results are stored in a "speculation cache."

When verification finishes:
- **Cache hit**: the predicted outcome matches reality, and the next speculation
  is returned instantly (zero draft latency).
- **Cache miss**: fallback to just-in-time (JIT) speculation (same as SD).

SSD is **lossless** -- it produces the exact same token distribution as the
target model.

### Key Parameter: K (Speculative Lookahead)

K is the number of tokens the draft guesses per round. Larger K means more
potential tokens per round, but the draft takes longer (K forward passes for SD,
or K tree-decode steps for SSD).

**In SSD, K has a critical constraint that doesn't exist in SD**: the draft's
tree-decode time (K x ~2.5ms on Qwen-0.6B/H100) must fit within the target's
verify time (~11.5ms on Qwen-32B/H100). If it overflows, the target idles
waiting for the draft, and the async benefit vanishes. See Section 5 for details.

---

## 2. Hardware and Software Requirements

### Hardware
- **Minimum**: 5 NVIDIA GPUs with >=80GB HBM each (4 for target + 1 for draft)
  - Tested on: H100 80GB HBM3
  - Should also work on: A100 80GB (set `SSD_CUDA_ARCH=8.0`)
- **Recommended**: 8 GPUs on a single node (allows running AR, SD, SSD in parallel)
- All GPUs must be on the same node with NVLink interconnect

### Software
- Python 3.11+
- CUDA >= 12.8
- Dependencies pinned in `pyproject.toml` (PyTorch 2.8.0, FlashInfer 0.5.2, etc.)

### Setup

```bash
git clone https://github.com/Weili-0234/ssd-wl.git
cd ssd-wl
```

**Important**: `uv sync` installs PyTorch and other packages that depend on CUDA
/ `nvcc`, which are only available on compute nodes. Run it inside an interactive
SLURM session, not on a login node:

```bash
# Get an interactive session on a compute node with GPUs first
sinfo # see the available partitions and node names 
salloc -p [partition name] \
    -N [# of nodes, e.g. 1] \
    --exclusive \
    --gres=gpu:[# of gpus per node, e.g. 4 or 8] \
    -J [job name, e.g. ssd-qwen-8b] \
    --time=4-0:00:00 \
    -w [node name]
srun --partition=[depends on your cluster] --gpus=8 --pty bash

# Then install (inside the srun session)
uv sync
uv run python -c "from ssd import LLM; print('ok')"
```

---

## 3. Environment Configuration

Create your local environment file (this file is in `.gitignore` and will not be
committed):

```bash
cp env_vars.sh.example env_vars.sh
```

Edit `env_vars.sh` and fill in your paths:

```bash
export SSD_HF_CACHE=/path/to/your/hf_cache       # HuggingFace model snapshots
export HF_DATASETS_CACHE=/path/to/your/datasets   # Raw HuggingFace datasets
export SSD_DATASET_DIR=/path/to/processed_datasets # Processed JSONL benchmark files
export SSD_CUDA_ARCH=9.0                           # 9.0=H100, 8.0=A100, or others
export HF_TOKEN=your_hf_token_here                 # HuggingFace token (for gated models like llama 3.3)
```

### Download Models

```bash
source env_vars.sh

# Qwen models (target + draft)
uv run python scripts/download_from_hf.py qwen

# Llama models (optional, for Llama experiments)
uv run python scripts/download_from_hf.py llama
```

### Download and Process Datasets

```bash
source env_vars.sh
uv run python scripts/get_data_from_hf.py
```

Verify everything is ready:
```bash
ls $SSD_HF_CACHE/models--Qwen--Qwen3-32B/snapshots/*/config.json
ls $SSD_HF_CACHE/models--Qwen--Qwen3-0.6B/snapshots/*/config.json
ls $SSD_DATASET_DIR/{humaneval,gsm8k,alpaca,ultrafeedback}/*_data_10000.jsonl
```

---

## 4. Running the Benchmark

### Quick Start: Full Comparison (AR + SD + SSD)

```bash
source env_vars.sh
bash bench/run_qwen_bench.sh
```

This runs 3 modes across 4 datasets (12 runs total). Each mode reloads the model
(different GPU configurations), so the full run takes about 90 minutes on H100.

| Mode | What it does | GPUs | Key flags |
|------|-------------|------|-----------|
| AR | Autoregressive baseline | 4 | `--gpus 4` |
| SD | Synchronous speculative decoding | 4 | `--gpus 4 --spec --k 6` |
| SSD | Async speculative decoding (Saguaro) | 5 | `--gpus 5 --spec --async --k 4 --f 3` |

Results are saved to `bench/results/` as individual log files per mode/dataset.

**Important**: the script uses `python -O` automatically. The `-O` flag disables
Python debug assertions that add significant overhead to every inference step.

### Running Individual Modes

```bash
source env_vars.sh

# AR -- autoregressive baseline, 4 GPUs
uv run python -O bench/bench.py --qwen --size 32 --gpus 4 \
    --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# SD -- synchronous speculative decoding, 4 GPUs, K=6
uv run python -O bench/bench.py --qwen --size 32 --gpus 4 --spec --k 6 \
    --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# SSD -- async speculative decoding (Saguaro), 5 GPUs, K=4, fan-out=3
uv run python -O bench/bench.py --qwen --size 32 --gpus 5 --spec --async --k 4 --f 3 \
    --b 1 --temp 0 --numseqs 128 --output_len 512 --all
```

Replace `--all` with a dataset flag to run a single dataset:

| Flag | Dataset |
|------|---------|
| `--humaneval` | Code completion (HumanEval) |
| `--alpaca` | Instruction following (Alpaca) |
| `--ultrafeedback` | Chat (UltraFeedback) |
| _(no flag)_ | Math (GSM8k, default) |

### Key Command-Line Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--qwen` / `--llama` | Model family | `--llama` |
| `--size 32` | Target model size in billions | `70` |
| `--gpus N` | Total GPUs (AR/SD: 4, SSD: 5) | `1` |
| `--spec` | Enable speculative decoding | off |
| `--async` | Enable async speculation (SSD) | off |
| `--k K` | Speculative lookahead (tokens per round) | `6` |
| `--f F` | Fan-out factor (SSD cache breadth) | `3` |
| `--b B` | Batch size | `1` |
| `--temp T` | Sampling temperature (0 = greedy) | `0.0` |
| `--numseqs N` | Sequences per dataset | `128` |
| `--output_len N` | Tokens to generate per sequence | `512` |
| `--all` | Run all 4 datasets | off |
| `--backup jit` | Cache miss strategy: `jit` or `fast` | `jit` |

---

## 5. Checking Progress and Analyzing Results

### Checking If an Experiment Is Still Running

```bash
# Check status of a running (or completed) experiment
uv run python bench/analyze_results.py bench/results/ssd_k4_full.log --status
```

Output for a running experiment:
```
  ssd_k4_full: 2/4 datasets complete
    humaneval          224.5 tok/s  [done]
    ultrafeedback      181.9 tok/s  [done]
    alpaca             87/128 sequences  [running]
    gsm                [pending]
```

Output when complete:
```
  ssd_k4_full: 4/4 datasets complete
    humaneval          224.5 tok/s  [done]
    ultrafeedback      181.9 tok/s  [done]
    alpaca             192.4 tok/s  [done]
    gsm                244.8 tok/s  [done]
    AVERAGE            210.9 tok/s
```

### Comparing Results Across Modes

```bash
# Compare AR, SD, SSD side by side
uv run python bench/analyze_results.py \
    bench/results/ar_full.log \
    bench/results/sd_full.log \
    bench/results/ssd_k4_full.log

# Include paper reference numbers
uv run python bench/analyze_results.py \
    bench/results/ar_full.log \
    bench/results/sd_full.log \
    bench/results/ssd_k4_full.log \
    --paper
```

Example output:
```
========================================================================
  Qwen-3-32B / Qwen-3-0.6B  --  Benchmark Results (tok/s)
========================================================================
Config          HumanEval     UltraFB      Alpaca       GSM8k     Average
------------------------------------------------------------------------
AR                 96.9        97.3        97.7        97.6        97.4
SD                152.7       121.7       127.2       164.1       141.4
SSD K=4           224.5       181.9       192.4       244.8       210.9
Paper SSD         222.0       174.0       185.0       234.0       203.8
------------------------------------------------------------------------

  Speedups (vs AR avg=97.4, SD avg=141.4):
    SSD K=4       SSD/AR=2.17x   SSD/SD=1.49x
```

### Comparing K Ablation Results

```bash
uv run python bench/analyze_results.py \
    bench/results/ssd_k{4,5,6,7}_full.log \
    --paper
```

---

## 6. Our Results: Qwen-3-32B / Qwen-3-0.6B on H100

### Main Results (K=4 for SSD, K=6 for SD)

Decode throughput (tok/s). Batch size 1, greedy decoding, 128 sequences per
dataset, 512 output tokens each.

| Mode | HumanEval | UltraFB | Alpaca | GSM8k | **Average** |
|------|-----------|---------|--------|-------|-------------|
| AR (ours) | 96.9 | 97.3 | 97.7 | 97.6 | 97.4 |
| AR (paper) | 88.8 | 88.8 | 88.8 | 88.8 | 88.8 |
| SD (ours, K=6) | 152.7 | 121.7 | 127.2 | 164.1 | 141.4 |
| SD (paper, K=5) | 146 | 122 | 127 | 152 | 136.8 |
| **SSD (ours, K=4)** | **224.5** | **182.0** | **192.4** | **244.8** | **210.9** |
| SSD (paper) | 222 | 174 | 185 | 234 | 203.8 |

Speedups:

| Ratio | Ours | Paper |
|-------|------|-------|
| **SSD / SD** | **1.49x** | **1.49x** |
| SSD / AR | 2.17x | 2.29x |

The SSD-over-SD speedup of **1.49x matches the paper exactly**.

### K Ablation for SSD

The speculative lookahead K has a dramatic effect on SSD throughput:

| K | HumanEval | UltraFB | Alpaca | GSM8k | **Average** | SSD/SD |
|---|-----------|---------|--------|-------|-------------|--------|
| **4** | **224.5** | **182.0** | **192.4** | **244.8** | **210.9** | **1.49x** |
| 5 | 210.1 | 162.7 | 174.1 | 227.5 | 193.6 | 1.37x |
| 6 | 193.9 | 146.0 | 156.9 | 204.4 | 175.3 | 1.24x |
| 7 | 169.0 | 131.4 | 141.1 | 192.7 | 158.6 | 1.12x |

#### Why K=4 Is Optimal for Qwen-32B/0.6B on H100

In SSD, the draft performs **tree-decode** on a separate GPU in parallel with the
target's verification. Each tree-decode step takes ~2.54ms for Qwen-0.6B on H100.
The target's verify takes ~11.5ms for Qwen-32B on H100.

The async benefit only works when the draft finishes before the target:

```
K=4:  draft tree-decode = 4 x 2.54ms = 10.2ms  < verify 11.5ms  --> fits, async works
K=5:  draft tree-decode = 5 x 2.54ms = 12.7ms  > verify 11.5ms  --> overflows by 1.2ms
K=7:  draft tree-decode = 7 x 2.54ms = 17.8ms  >> verify 11.5ms --> overflows by 6.3ms
```

When the draft overflows the verify window, the target **waits** for the draft
to finish. Each millisecond of overflow directly adds to step latency.

> **Note for A100 users**: Both the verify and tree-decode times will be
> different on A100. You should re-run the K ablation (K=3 through K=7) to find
> the optimal K for your hardware. The rule of thumb:
> `optimal K = floor(target_verify_time / per_step_tree_decode_time)`.

#### How We Discovered This

The README suggests `--k 7` for SSD, but we initially got only 169 tok/s (vs.
the paper's 222). Profiling with `SSD_PROFILE=1` revealed:

```
Target full step:  23.81ms  (handshake 11.36ms + verify 11.45ms + postprocess 0.04ms)
Draft tree-decode: 19.04ms  (7 steps x 2.54ms)
```

The handshake was 11.36ms because the target had to wait ~7ms for the draft to
finish tree-decode after verify completed. Reducing K to 4 brought tree-decode
to ~10.2ms, fitting within the 11.5ms verify window and eliminating the wait.

The paper's Appendix B.2 mentions "proposing 5 draft tokens per step" but this
refers to the SD baselines (SGLang/vLLM), not their SSD experiments. The paper
does not explicitly state the K used for SSD.

---

## 7. Profiling (Advanced)

For diagnosing throughput issues on new hardware:

```bash
source env_vars.sh

# Target-side breakdown: handshake / verify / postprocess per step
SSD_PROFILE=1 uv run python -O bench/bench.py --qwen --size 32 --gpus 5 \
    --spec --async --k 4 --f 3 --b 1 --temp 0 --numseqs 32 --output_len 512 --humaneval

# Draft-side breakdown: service / build_tree / decode_tree / populate
SSD_PROFILE_DRAFT=1 uv run python -O bench/bench.py ...

# Both together (most useful)
SSD_PROFILE=1 SSD_PROFILE_DRAFT=1 uv run python -O bench/bench.py ...
```

Output per step:
```
[PROFILE target] handshake=10.76ms verify=11.45ms postprocess=0.04ms total=22.25ms hits=1/1 toks=5
[PROFILE draft]  service=0.71ms build_tree=2.83ms decode_tree=10.16ms populate=0.05ms total=13.75ms
```

**Key diagnostic**: if `decode_tree` > `verify`, reduce K.

---

## 8. Adapting for A100 GPUs

1. Set `SSD_CUDA_ARCH=8.0` in your `env_vars.sh`
2. Run the K ablation to find A100-optimal K:

```bash
source env_vars.sh
for K in 3 4 5 6 7; do
    echo "=== K=$K ==="
    uv run python -O bench/bench.py --qwen --size 32 --gpus 5 --spec --async \
        --k $K --f 3 --b 1 --temp 0 --numseqs 128 --output_len 512 --all \
        2>&1 | tee bench/results/ssd_k${K}_a100.log
done

# Compare results
uv run python bench/analyze_results.py bench/results/ssd_k{3,4,5,6,7}_a100.log --paper
```

A100 is slower than H100, so both verify time and tree-decode time will increase.
If verify time increases proportionally more than tree-decode, a higher K (e.g.,
K=5 or K=6) may become optimal.

---

## 9. Troubleshooting

**Stale GPU processes** after a crash:
```bash
pkill -9 -f "bench.py"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

**Triton cache errors** (`OSError: Stale file handle`):
```bash
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/torchinductor_cache
```

**Dataset not found**: ensure `SSD_DATASET_DIR` points to the directory that
*contains* `humaneval/`, `gsm8k/`, etc. (not a parent directory).

**Model loading fails**: ensure `SSD_HF_CACHE` points to the HuggingFace cache
root containing `models--Qwen--Qwen3-32B/` etc.

---

## 10. TODO

- [ ] **K ablation for SD**: Sweep K=3..7 for synchronous speculative decoding.
  SD is less sensitive to K than SSD (no verify-window constraint), but there may
  be a sweet spot. Our current SD uses K=6; the paper baselines use K=5.
- [ ] **Llama-3.1-70B / Llama-3.2-1B experiments**: Reproduce the Llama results
  from the paper (Table B.3).
- [ ] **Temperature sweep**: Reproduce Figures 4/12 (temp=0, 0.3, 0.7, 1.0) to
  test geometric vs uniform fan-out at higher temperatures.
- [ ] **Batch size sweep**: Reproduce Figures 6/13 (b=1, 2, 4, 8, 16) to test
  fast vs neural backup at different batch sizes.
- [ ] **A100 experiments**: Full benchmark suite on A100 with re-tuned K.
