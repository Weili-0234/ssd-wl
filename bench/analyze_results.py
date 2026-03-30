#!/usr/bin/env python3
"""Parse and summarize benchmark log files.

Usage examples:

  # Check if an experiment is done (shows progress per dataset)
  python bench/analyze_results.py bench/results/ssd_k4_full.log --status

  # Parse throughput from a single log
  python bench/analyze_results.py bench/results/ssd_k4_full.log

  # Compare multiple logs side-by-side
  python bench/analyze_results.py bench/results/ar_full.log bench/results/sd_full.log bench/results/ssd_k4_full.log

  # Compare with paper numbers included
  python bench/analyze_results.py bench/results/ar_full.log bench/results/sd_full.log bench/results/ssd_k4_full.log --paper

  # K ablation: compare multiple SSD configs
  python bench/analyze_results.py bench/results/ssd_k{4,5,6,7}_full.log --paper
"""
import argparse
import re
import sys
from pathlib import Path

# Paper reference numbers for Qwen-3-32B / Qwen-3-0.6B
PAPER = {
    "Paper AR":  {"humaneval": 88.8, "ultrafeedback": 88.8, "alpaca": 88.8, "gsm": 88.8},
    "Paper SD":  {"humaneval": 146,  "ultrafeedback": 122,  "alpaca": 127,  "gsm": 152},
    "Paper SSD": {"humaneval": 222,  "ultrafeedback": 174,  "alpaca": 185,  "gsm": 234},
}

DATASETS = ["humaneval", "ultrafeedback", "alpaca", "gsm"]

THROUGHPUT_RE = re.compile(r"Total Throughput:\s+([0-9.]+)\s*tok/s")
HEADER_RE = re.compile(r"=== (.+?) / (\w+) ===")
GENERATING_RE = re.compile(r"Generating:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)")


def parse_log(path: str) -> dict:
    """Parse a benchmark log file. Returns {dataset: throughput}."""
    results = {}
    current_dataset = None
    with open(path) as f:
        for line in f:
            m = HEADER_RE.search(line)
            if m:
                current_dataset = m.group(2)
            m = THROUGHPUT_RE.search(line)
            if m and current_dataset:
                results[current_dataset] = float(m.group(1))
    return results


def check_status(path: str):
    """Print progress status for a running experiment."""
    results = parse_log(path)
    current_dataset = None
    last_progress = None

    with open(path) as f:
        for line in f:
            m = HEADER_RE.search(line)
            if m:
                current_dataset = m.group(2)
            m = GENERATING_RE.search(line)
            if m:
                last_progress = (int(m.group(2)), int(m.group(3)))

    name = Path(path).stem
    done = len(results)
    total = 4
    print(f"\n  {name}: {done}/{total} datasets complete")

    for ds in DATASETS:
        if ds in results:
            print(f"    {ds:<15} {results[ds]:>8.1f} tok/s  [done]")
        elif ds == current_dataset and last_progress:
            cur, tot = last_progress
            print(f"    {ds:<15} {cur}/{tot} sequences  [running]")
        elif done < total:
            status = "[pending]"
            print(f"    {ds:<15} {status}")

    if done == total:
        avg = sum(results.values()) / len(results)
        print(f"    {'AVERAGE':<15} {avg:>8.1f} tok/s")
    print()


def label_from_path(path: str) -> str:
    """Derive a short label from a log filename."""
    stem = Path(path).stem
    # ssd_k4_full -> SSD K=4
    # ar_full -> AR
    # sd_full -> SD
    stem = stem.replace("_full", "")
    if stem.startswith("ssd_k"):
        k = stem.replace("ssd_k", "")
        return f"SSD K={k}"
    return stem.upper()


def main():
    parser = argparse.ArgumentParser(
        description="Parse and compare SSD benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("logs", nargs="+", help="Log file(s) to analyze")
    parser.add_argument("--status", action="store_true",
                        help="Show progress status instead of results table")
    parser.add_argument("--paper", action="store_true",
                        help="Include paper reference numbers in comparison")
    args = parser.parse_args()

    if args.status:
        for log in args.logs:
            check_status(log)
        return

    # Parse all logs
    entries = {}
    for log in args.logs:
        label = label_from_path(log)
        results = parse_log(log)
        if results:
            entries[label] = results
        else:
            print(f"  WARNING: no throughput data in {log}", file=sys.stderr)

    if args.paper:
        entries.update(PAPER)

    if not entries:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Print table
    col_w = 12
    label_w = max(len(l) for l in entries) + 2

    ds_labels = {"humaneval": "HumanEval", "ultrafeedback": "UltraFB", "alpaca": "Alpaca", "gsm": "GSM8k"}
    header = f"{'Config':<{label_w}}" + "".join(f"{ds_labels.get(ds, ds):>{col_w}}" for ds in DATASETS) + f"{'Average':>{col_w}}"
    print()
    print("=" * len(header))
    print("  Qwen-3-32B / Qwen-3-0.6B  --  Benchmark Results (tok/s)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, results in entries.items():
        vals = [results.get(ds, 0) for ds in DATASETS]
        non_zero = [v for v in vals if v > 0]
        avg = sum(non_zero) / len(non_zero) if non_zero else 0
        row = f"{label:<{label_w}}"
        for v in vals:
            row += f"{v:>{col_w}.1f}" if v > 0 else f"{'--':>{col_w}}"
        row += f"{avg:>{col_w}.1f}" if avg > 0 else f"{'--':>{col_w}}"
        print(row)

    print("-" * len(header))

    # Compute speedups if we have AR and SD baselines
    ar = entries.get("AR", entries.get("Paper AR"))
    sd = entries.get("SD", entries.get("Paper SD"))
    if ar and sd:
        ar_avg = sum(ar.get(ds, 0) for ds in DATASETS) / len(DATASETS)
        sd_avg = sum(sd.get(ds, 0) for ds in DATASETS) / len(DATASETS)
        if ar_avg > 0 and sd_avg > 0:
            print(f"\n  Speedups (vs AR avg={ar_avg:.1f}, SD avg={sd_avg:.1f}):")
            for label, results in entries.items():
                if label.startswith("Paper") or label in ("AR", "SD"):
                    continue
                vals = [results.get(ds, 0) for ds in DATASETS]
                non_zero = [v for v in vals if v > 0]
                avg = sum(non_zero) / len(non_zero) if non_zero else 0
                if avg > 0:
                    print(f"    {label:<{label_w}} SSD/AR={avg/ar_avg:.2f}x   SSD/SD={avg/sd_avg:.2f}x")
    print()


if __name__ == "__main__":
    main()
