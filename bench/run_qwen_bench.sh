#!/usr/bin/env bash
# Reproduce Qwen-3-32B paper results: AR, SD, SSD across 4 datasets.
# Usage: bash bench/run_qwen_bench.sh
#
# Paper numbers (Table B.3):
#   AR  avg 88.8 tok/s | SD avg 136.8 tok/s | SSD avg 203.8 tok/s
#
# Results saved to bench/results/qwen_results.json and printed as a table.

set -euo pipefail
cd "$(dirname "$0")/.."
source env_vars.sh

RESULTS_DIR="bench/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/qwen_results.json"

# Initialize results JSON
echo '{}' > "$RESULTS_FILE"

DATASETS=("humaneval" "alpaca" "gsm" "ultrafeedback")
DATASET_FLAGS=("--humaneval" "--alpaca" "" "--ultrafeedback")
# gsm is the default (no flag needed)

NUMSEQS=128
OUTPUT_LEN=512
TEMP=0

# Helper: run one bench config and extract throughput
run_bench() {
    local mode="$1"
    local dataset="$2"
    local dataset_flag="$3"
    local extra_args="$4"

    echo ""
    echo "================================================================"
    echo "  MODE=$mode  DATASET=$dataset"
    echo "================================================================"

    # Build command
    local cmd="python -O bench/bench.py --qwen --size 32 --temp $TEMP --b 1"
    cmd+=" --numseqs $NUMSEQS --output_len $OUTPUT_LEN"
    if [ -n "$dataset_flag" ]; then
        cmd+=" $dataset_flag"
    fi
    cmd+=" $extra_args"

    echo "CMD: $cmd"
    echo ""

    # Run and capture output
    local logfile="$RESULTS_DIR/${mode}_${dataset}.log"
    $cmd 2>&1 | tee "$logfile"

    # Extract throughput from the "Total Throughput: XXX.XX tok/s" line
    local throughput
    throughput=$(grep -oP 'Total Throughput: \K[0-9.]+' "$logfile" | tail -1)

    if [ -z "$throughput" ]; then
        echo "WARNING: Could not extract throughput for $mode/$dataset"
        throughput="0"
    fi

    echo ""
    echo ">>> $mode / $dataset: $throughput tok/s"
    echo ""

    # Append to results JSON
    python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
data.setdefault('$mode', {})['$dataset'] = float('$throughput')
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Kill any stale workers before starting
pkill -9 -f "bench.py" 2>/dev/null || true
sleep 2

echo "============================================================"
echo "  Qwen-3-32B Reproducibility Benchmark"
echo "  Datasets: ${DATASETS[*]}"
echo "  NumSeqs: $NUMSEQS | OutputLen: $OUTPUT_LEN | Temp: $TEMP"
echo "============================================================"

# ── 1. Autoregressive baseline (4 GPUs) ──
for i in "${!DATASETS[@]}"; do
    run_bench "AR" "${DATASETS[$i]}" "${DATASET_FLAGS[$i]}" "--gpus 4"
done

# ── 2. Synchronous Speculative Decoding (4 GPUs, k=6) ──
for i in "${!DATASETS[@]}"; do
    run_bench "SD" "${DATASETS[$i]}" "${DATASET_FLAGS[$i]}" "--gpus 4 --spec --k 6"
done

# ── 3. SSD / Saguaro (5 GPUs, k=4, f=3, async, JIT backup) ──
# NOTE: K=4 is optimal for Qwen-32B/0.6B. The README suggests --k 7, but
# profiling shows draft tree-decode (K*2.54ms) must fit within the target
# verify window (~11.5ms). K=4 (10.2ms) fits; K=7 (17.8ms) overflows,
# destroying the async overlap benefit. See REPRODUCE.md for details.
for i in "${!DATASETS[@]}"; do
    run_bench "SSD" "${DATASETS[$i]}" "${DATASET_FLAGS[$i]}" "--gpus 5 --spec --async --k 4 --f 3"
done

# ── Print summary table ──
echo ""
echo ""
python3 - "$RESULTS_FILE" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

# Paper reference numbers
paper = {
    "AR":  {"humaneval": 88.8, "alpaca": 88.8, "gsm": 88.8, "ultrafeedback": 88.8},
    "SD":  {"humaneval": 146,  "alpaca": 127,  "gsm": 152,  "ultrafeedback": 122},
    "SSD": {"humaneval": 222,  "alpaca": 185,  "gsm": 234,  "ultrafeedback": 174},
}

datasets = ["humaneval", "ultrafeedback", "alpaca", "gsm"]
modes = ["AR", "SD", "SSD"]

print("=" * 90)
print(f"  Qwen-3-32B / Qwen-3-0.6B  —  Reproducibility Results")
print("=" * 90)
print(f"{'Mode':<6} {'Dataset':<15} {'Measured':>10} {'Paper':>10} {'Ratio':>8}  {'SSD/SD':>8}  {'SSD/AR':>8}")
print("-" * 90)

for mode in modes:
    measured_vals = []
    paper_vals = []
    for ds in datasets:
        m = data.get(mode, {}).get(ds, 0)
        p = paper.get(mode, {}).get(ds, 0)
        ratio = f"{m/p:.2f}x" if p > 0 and m > 0 else "—"
        measured_vals.append(m)
        paper_vals.append(p)
        print(f"{mode:<6} {ds:<15} {m:>9.1f}  {p:>9.1f}  {ratio:>8}")

    m_avg = sum(measured_vals) / len(measured_vals) if measured_vals else 0
    p_avg = sum(paper_vals) / len(paper_vals) if paper_vals else 0
    ratio = f"{m_avg/p_avg:.2f}x" if p_avg > 0 and m_avg > 0 else "—"
    print(f"{mode:<6} {'AVERAGE':<15} {m_avg:>9.1f}  {p_avg:>9.1f}  {ratio:>8}")
    print("-" * 90)

# Compute speedups
ar_avg = sum(data.get("AR", {}).get(ds, 0) for ds in datasets) / len(datasets)
sd_avg = sum(data.get("SD", {}).get(ds, 0) for ds in datasets) / len(datasets)
ssd_avg = sum(data.get("SSD", {}).get(ds, 0) for ds in datasets) / len(datasets)

print()
if ar_avg > 0 and sd_avg > 0 and ssd_avg > 0:
    print(f"  Measured SSD/AR:  {ssd_avg/ar_avg:.2f}x  (paper: 2.29x)")
    print(f"  Measured SSD/SD:  {ssd_avg/sd_avg:.2f}x  (paper: 1.49x)")
    print(f"  Measured SD/AR:   {sd_avg/ar_avg:.2f}x")
print()
PYEOF

echo "Results saved to $RESULTS_FILE"
