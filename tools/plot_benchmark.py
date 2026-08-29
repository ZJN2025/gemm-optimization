#!/usr/bin/env python3
"""把 results/benchmark.csv（及可选的 results/benchmark_cutlass.csv）画成对比图。

用法（在项目根目录执行）:
    python tools/plot_benchmark.py
    # 或
    .venv/bin/python tools/plot_benchmark.py

输出:
    results/benchmark_analysis.png  三张面板：TFLOPS / 延迟 / 相对 naive 的加速比
    跑过 ./build/benchmark_cutlass 之后，CUTLASS 参考线（灰色虚线/点线）会自动叠加进来。
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "results" / "benchmark.csv"
CUTLASS_CSV_PATH = BASE / "results" / "benchmark_cutlass.csv"
OUT_PATH = BASE / "results" / "benchmark_analysis.png"

# 手写 kernel 的类别色（dataviz 参考调色板，按 CSV 里的出现顺序依次分配；同一 kernel 颜色恒定）
# 8 个类别槽位是 CVD 安全的硬上限，第 9 个借用了同一调色板的蓝色序列深色档（step 550），
# 不再是严格意义上的第 9 个类别色——超过 9 个 kernel 时建议分面或分组，而不是继续加颜色
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#1c5cab",  # dark blue（sequential ramp step 550）
]

# CUTLASS 参考线：中性灰 + 虚线/点线，与手写 kernel 的彩色实线区分
REFERENCE_STYLES = {
    "cutlass_sgemm_simt": ("#898781", (0, (2, 2))),      # dotted
    "cutlass_sgemm_tf32": ("#52514e", (0, (6, 2))),      # dashed
    "cutlass_hgemm": ("#0b0b0b", (0, (6, 2, 2, 2))),     # dash-dot
}

SURFACE = "#fcfcfb"  # 图表底色
INK = "#0b0b0b"  # 标题
SECONDARY = "#52514e"  # 轴标题/图例
MUTED = "#898781"  # 刻度
GRIDLINE = "#e1e0d9"  # 网格线
BASELINE = "#c3c2b7"  # 轴线


def load_results(csv_path: Path) -> tuple[list[str], list[int], dict[str, dict[int, tuple[float, float]]]]:
    """读取 CSV，返回 (kernel 顺序列表, 尺寸列表, kernel -> {size -> (tflops, ms)})。文件不存在时返回空。"""
    kernels: list[str] = []
    sizes: set[int] = set()
    data: dict[str, dict[int, tuple[float, float]]] = {}

    if not csv_path.exists():
        return kernels, [], data

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["kernel"]
            size = int(row["M"])
            tflops = float(row["tflops"])
            ms = float(row["latency_ms"])
            if name not in data:
                kernels.append(name)
                data[name] = {}
            data[name][size] = (tflops, ms)
            sizes.add(size)

    return kernels, sorted(sizes), data


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=1.0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1.0)


def plot_lines(ax, kernels, sizes, data, get_y, styles, xlim=(230, 4400)):
    """在 ax 上为每个 kernel 画一条 2px 折线（圆点 + 底色描边）。"""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, labels=[str(s) for s in sizes])
    ax.set_xlim(*xlim)
    for kernel in kernels:
        color, linestyle, marker_size = styles[kernel]
        xs = [s for s in sizes if s in data[kernel]]
        ys = [get_y(data[kernel][s]) for s in xs]
        ax.plot(xs, ys, color=color, linewidth=2, linestyle=linestyle, marker="o",
                markersize=marker_size, markeredgewidth=2, markeredgecolor=SURFACE,
                solid_capstyle="round", solid_joinstyle="round", label=kernel, zorder=3)


def main() -> None:
    kernels, sizes, data = load_results(CSV_PATH)
    if not kernels:
        raise SystemExit(f"no data found in {CSV_PATH} - run benchmark first")

    cutlass_kernels, cutlass_sizes, cutlass_data = load_results(CUTLASS_CSV_PATH)
    if not cutlass_kernels:
        print("note: results/benchmark_cutlass.csv not found - run ./build/benchmark_cutlass "
              "to add CUTLASS reference lines")
    for name in cutlass_kernels:
        if name not in data:
            kernels.append(name)
        data.setdefault(name, {}).update(cutlass_data[name])
    sizes = sorted(set(sizes) | set(cutlass_sizes))

    # 线型分配：手写 kernel 按出现顺序取类别色（实线），cutlass 用灰色虚线
    styles = {}
    color_index = 0
    for kernel in kernels:
        if kernel in REFERENCE_STYLES:
            color, linestyle = REFERENCE_STYLES[kernel]
            styles[kernel] = (color, linestyle, 6)
        else:
            styles[kernel] = (SERIES_COLORS[color_index % len(SERIES_COLORS)], "solid", 9)
            color_index += 1

    # 以 gemm_naive 为基准的加速比（TFLOPS 比值 == 延迟比值）
    speedup = {}
    if "gemm_naive" in data:
        for kernel in kernels:
            speedup[kernel] = {
                s: data[kernel][s][0] / data["gemm_naive"][s][0]
                for s in sizes
                if s in data[kernel] and s in data["gemm_naive"]
            }

    # 轴范围随数据动态调整（cutlass 可能比手写 kernel 快一个数量级）
    all_tflops = [v[0] for kd in data.values() for v in kd.values()]
    all_ms = [v[1] for kd in data.values() for v in kd.values()]
    max_tflops = max(all_tflops)
    min_ms = min(all_ms)
    max_ms = max(all_ms)
    max_speedup = max((v for kd in speedup.values() for v in kd.values()), default=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("CUDA GEMM kernel comparison — RTX 5060 Laptop", color=INK,
                 fontsize=15, fontweight="bold", y=0.975)

    # 面板 1：TFLOPS
    ax = axes[0]
    style_axes(ax)
    ax.set_title("TFLOPS vs problem size (M=N=K)", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("problem size", color=SECONDARY, fontsize=10.5)
    ax.set_ylabel("TFLOPS", color=SECONDARY, fontsize=10.5)
    ax.set_ylim(0, max_tflops * 1.1)
    plot_lines(ax, kernels, sizes, data, lambda v: v[0], styles)
    footnote = "flops = 2·M·N·K; fp16-input kernels (tensor core) use half"
    if cutlass_kernels:
        footnote += "; dashed/dotted = CUTLASS reference"
    ax.text(0.99, 0.03, footnote, transform=ax.transAxes, ha="right",
            color=MUTED, fontsize=8.5)

    # 面板 2：延迟
    ax = axes[1]
    style_axes(ax)
    ax.set_title("Latency vs problem size", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("problem size", color=SECONDARY, fontsize=10.5)
    ax.set_ylabel("latency (ms, log)", color=SECONDARY, fontsize=10.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(min_ms * 0.5, max_ms * 1.6)
    plot_lines(ax, kernels, sizes, data, lambda v: v[1], styles)

    # 面板 3：相对 naive 的加速比
    ax = axes[2]
    style_axes(ax)
    ax.set_title("Speedup vs gemm_naive", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("problem size", color=SECONDARY, fontsize=10.5)
    ax.set_ylabel("speedup (×)", color=SECONDARY, fontsize=10.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, labels=[str(s) for s in sizes])
    ax.set_xlim(230, 4400)
    ax.set_ylim(0, max_speedup * 1.15)
    best_kernel, best_size, best_value = "gemm_naive", sizes[0], 1.0
    for kernel in kernels:
        if kernel not in speedup:
            continue
        color, linestyle, marker_size = styles[kernel]
        xs = [s for s in sizes if s in speedup[kernel]]
        ys = [speedup[kernel][s] for s in xs]
        ax.plot(xs, ys, color=color, linewidth=2, linestyle=linestyle, marker="o",
                markersize=marker_size, markeredgewidth=2, markeredgecolor=SURFACE,
                solid_capstyle="round", solid_joinstyle="round", label=kernel, zorder=3)
        for x, y in zip(xs, ys):
            if y > best_value:
                best_kernel, best_size, best_value = kernel, x, y
    ax.annotate(f"{best_kernel} ×{best_value:.1f}", xy=(best_size, best_value),
                xytext=(0, -16), textcoords="offset points", ha="center",
                color=SECONDARY, fontsize=9)

    legend_rows = (len(kernels) + 3) // 4
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4,
               frameon=False, fontsize=9, labelcolor=SECONDARY,
               handlelength=2.5, columnspacing=1.8)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.92,
                        bottom=0.03 + 0.035 * legend_rows, hspace=0.35)
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
