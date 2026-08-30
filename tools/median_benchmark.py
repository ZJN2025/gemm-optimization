#!/usr/bin/env python3
"""把 benchmark 跑 N 次取中位数，抑制笔记本 GPU 时钟漂移带来的抖动。

本机（RTX 5060 Laptop）不支持 nvidia-smi 锁频，因此用"重复运行取中位数"替代。
benchmark 每次运行会把结果写进 --out 指向的文件，本脚本跑完一次就把它另存为
results/<name>_run<i>.csv，最后取每个 (kernel, 尺寸) 的中位数写回 --out。

用法（在项目根目录执行）:
    python tools/median_benchmark.py                              # benchmark_gemm，3 次
    python tools/median_benchmark.py --cmd ./build/benchmark_cutlass \
        --out results/benchmark_cutlass.csv
"""

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIELDS = ["kernel", "M", "N", "K", "latency_ms", "tflops"]


def run_once(cmd: list[str], out_path: Path) -> list[dict]:
    """执行一次 benchmark（结果由它自己写进 out_path），返回其 CSV 行。"""
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"benchmark failed with exit code {result.returncode}")

    with out_path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", default="./build/benchmark_gemm",
                        help="benchmark 可执行文件（默认 ./build/benchmark_gemm）")
    parser.add_argument("--out", default="results/benchmark.csv",
                        help="合并后的输出 CSV，同时是 benchmark 自己的输出路径")
    parser.add_argument("--runs", type=int, default=3, help="重复次数（默认 3）")
    args = parser.parse_args()

    out_path = ROOT / args.out
    stem = Path(args.cmd).stem

    rows: dict[tuple, dict[str, list]] = defaultdict(lambda: {"ms": [], "tflops": []})
    key_order: list[tuple] = []

    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs}: {args.cmd}", flush=True)
        run_rows = run_once(args.cmd.split(), out_path)

        run_path = ROOT / "results" / f"{stem}_run{i + 1}.csv"
        run_path.parent.mkdir(exist_ok=True)
        with run_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(run_rows)

        for row in run_rows:
            key = (row["kernel"], row["M"], row["N"], row["K"])
            if key not in rows:
                key_order.append(key)
            rows[key]["ms"].append(float(row["latency_ms"]))
            rows[key]["tflops"].append(float(row["tflops"]))

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for key in key_order:
            writer.writerow([*key,
                             f"{statistics.median(rows[key]['ms']):.6g}",
                             f"{statistics.median(rows[key]['tflops']):.6g}"])

    print(f"median of {args.runs} runs saved -> {out_path}")


if __name__ == "__main__":
    main()
