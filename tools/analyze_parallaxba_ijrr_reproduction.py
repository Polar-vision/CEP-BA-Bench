#!/usr/bin/env python3
"""Analyze the mechanism-level reproduction of the IJRR ParallaxBA study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    "A0-XYZ-W",
    "A1-SphInvRange-Aw",
    "A2-Parallax-Mw",
    "A2-Parallax-Mc",
)
METHOD_LABELS = {
    "A0-XYZ-W": "XYZ",
    "A1-SphInvRange-Aw": "Inverse range",
    "A2-Parallax-Mw": "Parallax-Mw",
    "A2-Parallax-Mc": "Parallax-Mc",
}
COLORS = {
    "A0-XYZ-W": "#4C78A8",
    "A1-SphInvRange-Aw": "#F58518",
    "A2-Parallax-Mw": "#54A24B",
    "A2-Parallax-Mc": "#B279A2",
}
STRESS_ORDER = (
    "distant-features",
    "forward-motion-two-view",
    "relative-scale-1.2",
    "relative-scale-1.5",
)
STRESS_LABELS = {
    "distant-features": "Distant features",
    "forward-motion-two-view": "Forward two-view",
    "relative-scale-1.2": "Pose scale [1, 1.2]",
    "relative-scale-1.5": "Pose scale [1, 1.5]",
}


def median_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["stress", "method"], as_index=False)
        .agg(
            runs=("status", "size"),
            successful_runs=("status", lambda values: int((values == "ok").sum())),
            initial_rmse_spread=("initial_rmse_px", lambda values: values.max() - values.min()),
            median_initial_rmse_px=("initial_rmse_px", "median"),
            median_final_rmse_px=("final_rmse_px", "median"),
            median_iterations=("iterations", "median"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_linear_solver_time_sec=("linear_solver_time_sec", "median"),
            median_final_gradient_max_norm=("final_gradient_max_norm", "median"),
        )
        .sort_values(["stress", "method"])
    )


def read_point_conditioning(diagnostic_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for path in diagnostic_root.rglob("point_block_conditioning.csv"):
        relative = path.relative_to(diagnostic_root).parts
        dataset = relative[0]
        method = relative[2]
        frame = pd.read_csv(path)
        frame["condition_number"] = pd.to_numeric(
            frame["condition_number"],
            errors="coerce",
        )
        frame["dataset"] = dataset
        frame["method"] = method
        samples.append(frame)
        finite = frame["condition_number"].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "samples": len(frame),
                "finite_samples": len(finite),
                "median_condition_number": finite.median(),
                "p90_condition_number": finite.quantile(0.9),
                "maximum_condition_number": finite.max(),
                "rank_deficient_samples": int((frame["numerical_rank"] < 3).sum()),
            }
        )
    return pd.concat(samples, ignore_index=True), pd.DataFrame(summaries)


def plot_iterations(summary: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(STRESS_ORDER))
    width = 0.19
    for index, method in enumerate(METHODS):
        indexed = summary[summary["method"].eq(method)].set_index("stress")
        values = [indexed.loc[stress, "median_iterations"] for stress in STRESS_ORDER]
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    axis.set_xticks(x, [STRESS_LABELS[item] for item in STRESS_ORDER])
    axis.set_ylabel("Median LM iterations")
    axis.set_title("IJRR-style synthetic reproduction: convergence effort")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, frameon=False, loc="upper center")
    figure.tight_layout()
    figure.savefig(output / "ijrr_reproduction_iterations.png", dpi=220)
    plt.close(figure)


def plot_convergence(diagnostic_root: Path, output: Path) -> None:
    datasets = (
        "IJRR-S1-distant-s00",
        "IJRR-S2-forward-two-view-s00",
        "IJRR-S3-scale1p2-s00",
        "IJRR-S3-scale1p5-s00",
    )
    titles = (
        "Simulation 1: distant features",
        "Simulation 2: forward two-view features",
        "Simulation 3A: relative scales [1, 1.2]",
        "Simulation 3B: relative scales [1, 1.5]",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharey=True)
    for axis, dataset, title in zip(axes.flat, datasets, titles):
        curves: dict[str, pd.DataFrame] = {}
        final_cost = np.inf
        for method in METHODS:
            path = (
                diagnostic_root
                / dataset
                / f"{dataset}-quality"
                / method
                / "convergence.txt"
            )
            curve = pd.read_csv(path)
            curves[method] = curve
            final_cost = min(final_cost, float(curve["cost"].iloc[-1]))
        for method, curve in curves.items():
            denominator = max(float(curve["cost"].iloc[0]) - final_cost, 1e-20)
            relative_gap = np.maximum(
                (curve["cost"].to_numpy() - final_cost) / denominator,
                1e-14,
            )
            axis.semilogy(
                curve["iteration"],
                relative_gap,
                marker="o",
                markersize=2.8,
                linewidth=1.5,
                label=METHOD_LABELS[method],
                color=COLORS[method],
            )
        axis.set_title(title)
        axis.set_xlabel("LM iteration")
        axis.set_ylabel("Normalized objective gap")
        axis.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, frameon=False, loc="upper center")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output / "ijrr_reproduction_convergence.png", dpi=220)
    plt.close(figure)


def plot_conditioning(samples: pd.DataFrame, output: Path) -> None:
    datasets = (
        "IJRR-S1-distant-s00",
        "IJRR-S2-forward-two-view-s00",
    )
    titles = (
        "Simulation 1: distant features",
        "Simulation 2: forward two-view features",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for axis, dataset, title in zip(axes, datasets, titles):
        values = []
        for method in METHODS:
            selected = samples[
                samples["dataset"].eq(dataset) & samples["method"].eq(method)
            ]["condition_number"].replace([np.inf, -np.inf], np.nan).dropna()
            values.append(selected.to_numpy())
        boxes = axis.boxplot(
            values,
            tick_labels=[METHOD_LABELS[method] for method in METHODS],
            showfliers=True,
            patch_artist=True,
        )
        for patch, method in zip(boxes["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.75)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("Point-block condition number")
        axis.grid(axis="y", alpha=0.25, which="both")
        axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(output / "ijrr_reproduction_point_conditioning.png", dpi=220)
    plt.close(figure)


def markdown_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Stress case | Method | Median iterations | Median solver time (s) | Median final RMSE (px) |",
        "|---|---:|---:|---:|---:|",
    ]
    for stress in STRESS_ORDER:
        selected = summary[summary["stress"].eq(stress)].set_index("method")
        for method in METHODS:
            row = selected.loc[method]
            lines.append(
                f"| {STRESS_LABELS[stress]} | {METHOD_LABELS[method]} | "
                f"{row.median_iterations:.1f} | "
                f"{row.median_solver_time_sec:.3f} | "
                f"{row.median_final_rmse_px:.6f} |"
            )
    return lines


def write_report(
    summary: pd.DataFrame,
    conditioning: pd.DataFrame,
    output: Path,
) -> None:
    indexed = summary.set_index(["stress", "method"])
    condition_index = conditioning.set_index(["dataset", "method"])
    s1_xyz = indexed.loc[("distant-features", "A0-XYZ-W")]
    s1_inv = indexed.loc[("distant-features", "A1-SphInvRange-Aw")]
    s1_pa = indexed.loc[("distant-features", "A2-Parallax-Mw")]
    s2_xyz = indexed.loc[("forward-motion-two-view", "A0-XYZ-W")]
    s2_inv = indexed.loc[("forward-motion-two-view", "A1-SphInvRange-Aw")]
    s2_pa = indexed.loc[("forward-motion-two-view", "A2-Parallax-Mw")]
    s1_xyz_cond = condition_index.loc[
        ("IJRR-S1-distant-s00", "A0-XYZ-W")
    ]
    s1_pa_cond = condition_index.loc[
        ("IJRR-S1-distant-s00", "A2-Parallax-Mw")
    ]
    s2_xyz_cond = condition_index.loc[
        ("IJRR-S2-forward-two-view-s00", "A0-XYZ-W")
    ]
    s2_inv_cond = condition_index.loc[
        ("IJRR-S2-forward-two-view-s00", "A1-SphInvRange-Aw")
    ]
    s2_pa_cond = condition_index.loc[
        ("IJRR-S2-forward-two-view-s00", "A2-Parallax-Mw")
    ]

    report = [
        "# ParallaxBA IJRR 优势复现实验",
        "",
        "## 复现范围",
        "",
        "- 这是对 Zhao 等人 IJRR ParallaxBA 论文 Simulation 1--3 的机理级复现，不是原始二进制和原始随机数据的逐数值复刻。",
        "- 相机内参为 800 x 800 像幅、焦距 400 px、主点 (400, 400)，影像噪声为 0.1 px。",
        "- 三种参数化使用完全相同的相机初值、带噪观测和三角化物点初值。",
        "- 求解器统一为 Ceres 2.3 LM + SPARSE_SCHUR，最多 200 次迭代，并固定单目规范自由度。",
        "- 原论文 ParallaxBA 对应 `A2-Parallax-Mw`；`A2-Parallax-Mc` 仅作为本文扩展版本。",
        "",
        "## 定量结果",
        "",
        *markdown_table(summary),
        "",
        "## 是否复现了论文声称的优势",
        "",
        (
            f"1. **远距离点相对 XYZ 的优势得到复现。** "
            f"`A2-Parallax-Mw` 的中位迭代数为 {s1_pa.median_iterations:.0f}，"
            f"`A0-XYZ-W` 为 {s1_xyz.median_iterations:.0f}，"
            f"迭代数降低约 {100.0 * (1.0 - s1_pa.median_iterations / s1_xyz.median_iterations):.1f}%。"
            f"两者最终 RMSE 相同。"
        ),
        (
            f"2. **Simulation 1 中 Parallax 并不优于逆深度。** "
            f"逆深度中位迭代数为 {s1_inv.median_iterations:.0f}，"
            f"低于 Parallax 的 {s1_pa.median_iterations:.0f}。"
            "这与原论文的文字结论一致：远景点场景中逆深度也能有效处理退化。"
        ),
        (
            f"3. **沿运动方向、仅两次观测的点上，Parallax 相对逆深度的优势得到复现。** "
            f"Parallax、逆深度和 XYZ 的中位迭代数分别为 "
            f"{s2_pa.median_iterations:.0f}、{s2_inv.median_iterations:.0f} 和 "
            f"{s2_xyz.median_iterations:.0f}。"
        ),
        (
            "4. **较差姿态尺度初值下没有普遍优势。** "
            "尺度扰动 [1, 1.2] 和 [1, 1.5] 中四种方法均收敛到相同 RMSE，"
            "迭代差异较小。这支持原论文 Simulation 3 的限定："
            "良好的物点参数化不能消除姿态初值本身造成的非线性困难。"
        ),
        "",
        "## 条件数证据",
        "",
        (
            f"- Simulation 1 中，XYZ 点块条件数的 90% 分位数为 "
            f"{s1_xyz_cond.p90_condition_number:.3g}，"
            f"Parallax 为 {s1_pa_cond.p90_condition_number:.3g}。"
        ),
        (
            f"- Simulation 2 中，XYZ 与逆深度点块最大条件数分别达到 "
            f"{s2_xyz_cond.maximum_condition_number:.3g} 和 "
            f"{s2_inv_cond.maximum_condition_number:.3g}，"
            f"Parallax 最大值为 {s2_pa_cond.maximum_condition_number:.3g}。"
        ),
        "- 该诊断是每个物点局部正规块的条件数，不等同于原论文绘制的完整信息矩阵条件数，但直接对应参数化对单点深度方向可观性的影响。",
        "",
        "## 实现核验",
        "",
        "- 修复前 `A2-Parallax-Mw` 将方向角和视差角注册为两个 Ceres 参数块，破坏了标准 Schur 的三维物点块结构。",
        "- 现已将其改为连续的 `[azimuth, elevation, parallax]` 三维参数块，与原论文定义一致。",
        "- 修复后远景场景中 `A2-Parallax-Mw` 的中位求解时间为 "
        f"{s1_pa.median_solver_time_sec:.3f} s，XYZ 为 {s1_xyz.median_solver_time_sec:.3f} s；"
        "此前观察到的 Mw 异常慢主要是实现问题，而不是参数化的必然代价。",
        "",
        "## 结论",
        "",
        "> 本复现实验支持 ParallaxBA 的条件性优势：当场景包含远距离点，尤其包含沿相机运动方向、深度几乎不可观且观测次数很少的点时，Parallax 参数化能够显著改善局部条件数，并减少 LM 达到同一重投影误差所需的迭代次数。该优势不是全局性的；普通远景点可由逆深度同样有效地处理，而较差的相机姿态初值仍会同时限制所有物点参数化。",
        "",
        "## 输出文件",
        "",
        "- `aggregate_summary.csv`: 三个随机重复的汇总结果。",
        "- `point_conditioning_summary.csv`: 代表场景的点块条件数。",
        "- `ijrr_reproduction_convergence.png`: 归一化目标函数下降曲线。",
        "- `ijrr_reproduction_iterations.png`: 迭代数对比。",
        "- `ijrr_reproduction_point_conditioning.png`: 点块条件数分布。",
    ]
    (output / "conclusions_zh.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\zuo\projects\CEP\parallaxba_ijrr_reproduction"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(root / "scenario_manifest.csv")
    clean = pd.read_csv(root / "results_clean" / "summary.csv")
    numeric_columns = (
        "initial_rmse_px",
        "final_rmse_px",
        "iterations",
        "solver_time_sec",
        "linear_solver_time_sec",
        "final_gradient_max_norm",
    )
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.merge(
        manifest[["base_dataset", "stress", "paper_simulation", "seed"]],
        on="base_dataset",
        how="left",
    )
    summary = median_summary(clean)
    samples, conditioning = read_point_conditioning(root / "results_diagnostic")

    clean.to_csv(output / "paired_results.csv", index=False)
    summary.to_csv(output / "aggregate_summary.csv", index=False)
    conditioning.to_csv(output / "point_conditioning_summary.csv", index=False)
    samples.to_csv(output / "point_conditioning_samples.csv", index=False)
    plot_iterations(summary, output)
    plot_convergence(root / "results_diagnostic", output)
    plot_conditioning(samples, output)
    write_report(summary, conditioning, output)
    print(f"Analysis: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
