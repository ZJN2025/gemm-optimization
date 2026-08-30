#!/usr/bin/env python3
"""把 results/benchmark.csv 画成对比图。

输出两张图：
- results/benchmark_analysis.png   主线 9 个 kernel（演进路线）
- results/benchmark_controlled.png 对照实验组（单变量控制，6 个配置）

用法（在项目根目录执行）:
    python tools/plot_benchmark.py
    # 或
    .venv/bin/python tools/plot_benchmark.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "benchmark.csv"
OUT_MAIN = Path(__file__).resolve().parent.parent / "results" / "benchmark_analysis.png"
OUT_CONTROL = Path(__file__).resolve().parent.parent / "results" / "benchmark_controlled.png"

# 主线 kernel：演进路线，画在主图里
MAIN_KERNELS = {
    "gemm_naive",
    "gemm_shared_tiling",
    "gemm_vectorized",
    "gemm_async_copy",
    "gemm_double_buffer",
    "gemm_async_vectorized",
    "gemm_simt_128x128",
    "gemm_tensor_core",
    "gemm_tensor_core_optimized",
    "gemm_tensor_core_mma",
}

# 对照实验组：同一骨架只改一个变量，单独一张图（避免和主线混在一起产生混杂对比）
CONTROL_KERNELS = {
    "gemm_tile_32x32",
    "gemm_tile_64x64",
    "gemm_tile_128x128",
    "gemm_rblock_4x8",
    "gemm_stages2_128x128",
    "gemm_stages4_128x128",
}

# 类别色（dataviz 参考调色板，按本图内的出现顺序依次分配；同一 kernel 颜色恒定）
# 8 个类别槽位是 CVD 安全的硬上限，第 9 个借用了同一调色板的蓝色序列深色档（step 550）
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
    "#3987e5",  # light blue（sequential ramp step 400；第 10 个系列已越过类别色安全上限）
]

SURFACE = "#fcfcfb"  # 图表底色
INK = "#0b0b0b"  # 标题
SECONDARY = "#52514e"  # 轴标题/图例
MUTED = "#898781"  # 刻度
GRIDLINE = "#e1e0d9"  # 网格线
BASELINE = "#c3c2b7"  # 轴线


def load_results(csv_path: Path) -> tuple[list[str], list[int], dict[str, dict[int, tuple[float, float]]]]:
    """读取 CSV，返回 (kernel 顺序列表, 尺寸列表, kernel -> {size -> (tflops, ms)})。"""
    kernels: list[str] = []
    sizes: set[int] = set()
    data: dict[str, dict[int, tuple[float, float]]] = {}

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


def plot_lines(ax, kernels, sizes, data, get_y, colors):
    """在 ax 上为每个 kernel 画一条 2px 折线（9px 圆点 + 底色描边）。"""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, labels=[str(s) for s in sizes])
    ax.set_xlim(230, 4400)
    for kernel, color in zip(kernels, colors):
        xs = [s for s in sizes if s in data[kernel]]
        ys = [get_y(data[kernel][s]) for s in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=9,
                markeredgewidth=2, markeredgecolor=SURFACE, solid_capstyle="round",
                solid_joinstyle="round", label=kernel, zorder=3)


def make_figure(selected_kernels, data, sizes, title, out_path):
    """画一张三面板图（TFLOPS / 延迟 / 相对 naive 的加速比）。"""
    kernels = [k for k in selected_kernels if k in data]
    if not kernels:
        raise SystemExit(f"no data found in {CSV_PATH} for {title} - run benchmark first")

    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(kernels))]

    # 纵轴范围按本图数据动态计算（留 15% 余量），分数再高也不会画出界
    max_tflops = max(v[0] for k in kernels for v in data[k].values())
    min_ms = min(v[1] for k in kernels for v in data[k].values())
    max_ms = max(v[1] for k in kernels for v in data[k].values())

    # 加速比统一以 gemm_naive 为基准（无论它是否在本图内）
    speedup = {}
    if "gemm_naive" in data:
        for kernel in kernels:
            speedup[kernel] = {
                s: data[kernel][s][0] / data["gemm_naive"][s][0]
                for s in sizes
                if s in data[kernel] and s in data["gemm_naive"]
            }

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, color=INK, fontsize=15, fontweight="bold", y=0.975)

    # 面板 1：TFLOPS
    ax = axes[0]
    style_axes(ax)
    ax.set_title("TFLOPS vs problem size (M=N=K)", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("problem size", color=SECONDARY, fontsize=10.5)
    ax.set_ylabel("TFLOPS", color=SECONDARY, fontsize=10.5)
    ax.set_ylim(0, max_tflops * 1.15)
    plot_lines(ax, kernels, sizes, data, lambda v: v[0], colors)
    ax.text(0.99, 0.03, "flops = 2·M·N·K; fp16-input kernels (tensor core) use half",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)

    # 面板 2：延迟
    ax = axes[1]
    style_axes(ax)
    ax.set_title("Latency vs problem size", color=INK, fontsize=12.5,
                 fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("problem size", color=SECONDARY, fontsize=10.5)
    ax.set_ylabel("latency (ms, log)", color=SECONDARY, fontsize=10.5)
    ax.set_yscale("log")
    ax.set_yticks([t for t in (0.01, 0.1, 1, 10, 100, 1000)
                   if min_ms / 3 <= t <= max_ms * 1.5])
    ax.set_ylim(min_ms / 3, max_ms * 1.5)
    plot_lines(ax, kernels, sizes, data, lambda v: v[1], colors)

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
    max_speedup = max(v for k in speedup for v in speedup[k].values()) if speedup else 1.0
    ax.set_ylim(0, max(2.0, max_speedup * 1.15))
    best_kernel, best_size, best_value = kernels[0], sizes[0], 0.0
    for kernel, color in zip(kernels, colors):
        if kernel not in speedup:
            continue
        xs = [s for s in sizes if s in speedup[kernel]]
        ys = [speedup[kernel][s] for s in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=9,
                markeredgewidth=2, markeredgecolor=SURFACE, solid_capstyle="round",
                solid_joinstyle="round", label=kernel, zorder=3)
        for x, y in zip(xs, ys):
            if y > best_value:
                best_kernel, best_size, best_value = kernel, x, y
    if best_value > 0:
        ax.annotate(f"{best_kernel} ×{best_value:.1f}", xy=(best_size, best_value),
                    xytext=(0, -16), textcoords="offset points", ha="center",
                    color=SECONDARY, fontsize=9)

    # 图例独占画布最底部：按条目行数动态预留高度（ncol=4），
    # 面板整体上移，保证图例与最下面板的横轴刻度之间有空隙
    n_legend_rows = (len(kernels) + 3) // 4

    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4,
               frameon=False, fontsize=9, labelcolor=SECONDARY,
               handlelength=2.5, columnspacing=1.8)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.92,
                        bottom=0.055 + 0.045 * n_legend_rows, hspace=0.35)
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main() -> None:
    kernels, sizes, data = load_results(CSV_PATH)
    if not kernels:
        raise SystemExit(f"no data found in {CSV_PATH} - run benchmark first")

    # 主图：只画主线 kernel（对照实验组单独成图，避免混杂变量干扰对比）
    main_kernels = [k for k in kernels if k in MAIN_KERNELS]
    control_kernels = [k for k in kernels if k in CONTROL_KERNELS]

    make_figure(main_kernels, data, sizes,
                "CUDA GEMM kernels — RTX 5060 Laptop", OUT_MAIN)

    if control_kernels:
        make_figure(control_kernels, data, sizes,
                    "Controlled experiments (single variable) — RTX 5060 Laptop",
                    OUT_CONTROL)


if __name__ == "__main__":
    main()
