#!/usr/bin/env python3
"""Analyze controlled synthetic CEP benchmark results and select diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "A0-XYZ-W",
    "A0-XYInvZ-W",
    "A0-SphRange-W",
    "A0-SphInvRange-W",
    "A1-XYZ-Ac",
    "A1-XYInvZ-Ac",
    "A1-SphRange-Ac",
    "A1-SphInvRange-Ac",
    "A2-Parallax-Mc",
    "A1-XYZ-Aw",
    "A1-XYInvZ-Aw",
    "A1-SphRange-Aw",
    "A1-SphInvRange-Aw",
    "A2-Parallax-Mw",
]

FOCUS_METHODS = [
    "A0-XYZ-W",
    "A0-XYInvZ-W",
    "A1-XYInvZ-Ac",
    "A1-XYInvZ-Aw",
    "A2-Parallax-Mc",
    "A2-Parallax-Mw",
]

LEGACY_METHOD_ALIASES = {
    "A1-IDP-Civera-Ac": "A1-XYInvZ-Ac",
}


def load_results(root: Path) -> pd.DataFrame:
    results = pd.read_csv(root / "clean_results" / "summary.csv")
    results["method"] = results["method"].replace(LEGACY_METHOD_ALIASES)
    manifest = pd.read_csv(root / "scenario_manifest.csv")
    data = results.merge(
        manifest,
        on=["base_dataset", "quality_dataset"],
        how="left",
        validate="many_to_one",
    )
    numeric = [
        "wall_time_sec",
        "solver_time_sec",
        "linear_solver_time_sec",
        "initial_rmse_px",
        "final_rmse_px",
        "iterations",
        "accepted_steps",
        "rejected_steps",
        "termination_type",
        "target_parallax_deg",
        "parallax_median_deg",
        "depth_scale",
        "rotation_noise_deg",
        "translation_noise_ratio",
        "track_length",
        "coordinate_offset",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    grouped = data.groupby("quality_dataset", sort=False)
    data["best_rmse"] = grouped["final_rmse_px"].transform("min")
    data["rmse_gap_pct"] = 100.0 * (data["final_rmse_px"] / data["best_rmse"] - 1.0)
    data["solution_success"] = (
        data["status"].eq("ok")
        & np.isfinite(data["final_rmse_px"])
        & data["rmse_gap_pct"].le(1.0)
    )
    data["ceres_converged"] = data["termination_type"].eq(0)
    data["best_solver_time"] = grouped["solver_time_sec"].transform("min")
    data["time_ratio_best"] = data["solver_time_sec"] / data["best_solver_time"]
    data["best_iterations"] = grouped["iterations"].transform("min")
    data["iteration_gap"] = data["iterations"] - data["best_iterations"]
    data["iteration_winner"] = data["iteration_gap"].eq(0)
    data["rmse_winner"] = data["rmse_gap_pct"].le(1e-8)
    data["time_winner"] = data["time_ratio_best"].le(1.000001)
    return data


def method_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("method")
        .agg(
            cases=("method", "size"),
            ceres_convergence_rate=("ceres_converged", "mean"),
            solution_success_rate=("solution_success", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            p90_rmse_gap_pct=("rmse_gap_pct", lambda x: x.quantile(0.9)),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_time_ratio_best=("time_ratio_best", "median"),
            median_iterations=("iterations", "median"),
            iteration_win_rate=("iteration_winner", "mean"),
            rmse_win_rate=("rmse_winner", "mean"),
            time_win_rate=("time_winner", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    for column in (
        "ceres_convergence_rate",
        "solution_success_rate",
        "iteration_win_rate",
        "rmse_win_rate",
        "time_win_rate",
    ):
        summary[column] *= 100.0
    return summary


def family_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["family", "method"])
        .agg(
            cases=("method", "size"),
            ceres_convergence_rate=("ceres_converged", "mean"),
            solution_success_rate=("solution_success", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_time_ratio_best=("time_ratio_best", "median"),
            median_iterations=("iterations", "median"),
        )
        .reset_index()
    )
    summary["ceres_convergence_rate"] *= 100.0
    summary["solution_success_rate"] *= 100.0
    return summary


def select_diagnostic_cases(data: pd.DataFrame, maximum: int = 12) -> pd.DataFrame:
    keys = [
        "base_dataset",
        "quality_dataset",
        "family",
        "trajectory",
        "target_parallax_deg",
        "parallax_median_deg",
        "depth_scale",
        "pose_level",
        "track_length",
        "coordinate_offset",
        "seed",
    ]
    spread = (
        data.groupby(keys, dropna=False)
        .agg(
            rmse_spread_pct=("rmse_gap_pct", "max"),
            iteration_spread=("iterations", lambda x: x.max() - x.min()),
            time_ratio_spread=("time_ratio_best", "max"),
        )
        .reset_index()
    )
    selected: list[pd.DataFrame] = [
        spread.sort_values(
            ["rmse_spread_pct", "iteration_spread"], ascending=False
        ).head(6)
    ]

    parallax = spread[spread["family"].eq("parallax_depth")]
    for trajectory in ("lateral", "forward"):
        subset = parallax[parallax["trajectory"].eq(trajectory)]
        for alpha in (subset["target_parallax_deg"].min(), subset["target_parallax_deg"].max()):
            candidates = subset[
                np.isclose(subset["target_parallax_deg"], alpha)
                & np.isclose(subset["depth_scale"], subset["depth_scale"].max())
            ]
            selected.append(candidates.head(1))

    track = spread[spread["family"].eq("track_stress")].sort_values(
        ["track_length", "rmse_spread_pct"], ascending=[True, False]
    )
    selected.append(track.head(2))
    coordinate = spread[spread["family"].eq("coordinate_stress")].sort_values(
        ["coordinate_offset", "rmse_spread_pct"], ascending=False
    )
    selected.append(coordinate.head(2))

    result = (
        pd.concat(selected, ignore_index=True)
        .drop_duplicates("quality_dataset")
        .sort_values(["family", "trajectory", "target_parallax_deg", "depth_scale"])
        .head(maximum)
        .reset_index(drop=True)
    )
    result.insert(0, "selection_id", np.arange(1, len(result) + 1))
    return result


def plot_overall(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 8.5), sharey=True)
    positions = np.arange(len(summary))
    labels = summary["method"]
    axes[0].barh(positions, summary["solution_success_rate"], color="#2a6fbb")
    axes[0].set_xlabel("Solution success rate (%)")
    axes[0].set_xlim(50, 101)
    axes[1].barh(positions, summary["median_time_ratio_best"], color="#d97706")
    axes[1].set_xlabel("Median solver-time ratio to fastest")
    axes[1].set_xscale("log")
    axes[2].barh(positions, summary["median_iterations"], color="#3a8f5b")
    axes[2].set_xlabel("Median LM iterations")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Synthetic benchmark: robustness and efficiency")
    figure.tight_layout()
    figure.savefig(output / "method_robustness_efficiency.png", dpi=220)
    plt.close(figure)


def load_scale_validation(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale_root = root.parent / f"{root.name}_scale"
    if not (scale_root / "clean_results" / "summary.csv").exists():
        return pd.DataFrame(), pd.DataFrame()
    scale_data = load_results(scale_root)
    return scale_data, method_summary(scale_data)


def plot_scale_validation(data: pd.DataFrame, output: Path) -> None:
    if data.empty:
        return
    labels = (
        data[["quality_dataset", "pose_level", "trajectory"]]
        .drop_duplicates()
        .assign(
            label=lambda frame: frame["pose_level"].astype(str)
            + "\n"
            + frame["trajectory"].astype(str)
        )
        .set_index("quality_dataset")["label"]
    )
    rmse = data.pivot(
        index="quality_dataset", columns="method", values="rmse_gap_pct"
    ).reindex(columns=METHOD_ORDER)
    time = data.pivot(
        index="quality_dataset", columns="method", values="time_ratio_best"
    ).reindex(columns=METHOD_ORDER)
    rmse.index = [labels[index] for index in rmse.index]
    time.index = rmse.index

    figure, axes = plt.subplots(2, 1, figsize=(16, 7.5), constrained_layout=True)
    image_rmse = axes[0].imshow(
        np.log10(1.0 + rmse.to_numpy(dtype=float)),
        aspect="auto",
        cmap="magma",
    )
    axes[0].set_title("log10(1 + final-RMSE gap to best, %)")
    image_time = axes[1].imshow(
        np.log10(time.to_numpy(dtype=float)),
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_title("log10(solver-time ratio to fastest)")
    for axis in axes:
        axis.set_xticks(range(len(METHOD_ORDER)))
        axis.set_xticklabels(METHOD_ORDER, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(range(len(rmse.index)))
        axis.set_yticklabels(rmse.index, fontsize=8)
    figure.colorbar(image_rmse, ax=axes[0], shrink=0.8)
    figure.colorbar(image_time, ax=axes[1], shrink=0.8)
    figure.suptitle("Scale-validation boundary cases: 12 cameras, 600 points")
    figure.savefig(output / "scale_validation_heatmaps.png", dpi=220)
    plt.close(figure)


def plot_parallax_heatmaps(data: pd.DataFrame, output: Path, trajectory: str) -> None:
    subset = data[
        data["family"].eq("parallax_depth")
        & data["trajectory"].eq(trajectory)
        & data["method"].isin(FOCUS_METHODS)
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    image = None
    for axis, method in zip(axes.flat, FOCUS_METHODS):
        method_data = subset[subset["method"].eq(method)]
        pivot = method_data.pivot_table(
            index="target_parallax_deg",
            columns="depth_scale",
            values="solution_success",
            aggfunc="mean",
        ).sort_index()
        values = 100.0 * pivot.to_numpy(dtype=float)
        image = axis.imshow(values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        axis.set_title(method)
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels([f"{x:g}" for x in pivot.columns])
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"{x:g}" for x in pivot.index])
        axis.set_xlabel("Initial range scale")
        axis.set_ylabel("Target parallax (deg)")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="Solution success (%)")
    figure.suptitle(f"Parallax-depth robustness: {trajectory} trajectory")
    figure.savefig(
        output / f"parallax_depth_success_{trajectory}.png",
        dpi=220,
    )
    plt.close(figure)


def plot_track_stress(data: pd.DataFrame, output: Path) -> None:
    subset = data[
        data["family"].eq("track_stress") & data["method"].isin(FOCUS_METHODS)
    ]
    grouped = (
        subset.groupby(["method", "track_length"])
        .agg(success=("solution_success", "mean"), iterations=("iterations", "median"))
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in FOCUS_METHODS:
        method_data = grouped[grouped["method"].eq(method)]
        axes[0].plot(
            method_data["track_length"],
            100.0 * method_data["success"],
            marker="o",
            label=method,
        )
        axes[1].plot(
            method_data["track_length"],
            method_data["iterations"],
            marker="o",
            label=method,
        )
    axes[0].set_ylabel("Solution success rate (%)")
    axes[1].set_ylabel("Median LM iterations")
    for axis in axes:
        axis.set_xlabel("Track length")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle("Track-length stress test")
    figure.tight_layout()
    figure.savefig(output / "track_length_stress.png", dpi=220)
    plt.close(figure)


def plot_coordinate_stress(data: pd.DataFrame, output: Path) -> None:
    subset = data[
        data["family"].eq("coordinate_stress") & data["method"].isin(FOCUS_METHODS)
    ]
    grouped = (
        subset.groupby(["method", "coordinate_offset"])
        .agg(
            rmse_gap=("rmse_gap_pct", "median"),
            time_ratio=("time_ratio_best", "median"),
        )
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in FOCUS_METHODS:
        method_data = grouped[grouped["method"].eq(method)]
        x = method_data["coordinate_offset"].replace(0.0, 1.0)
        axes[0].plot(x, method_data["rmse_gap"] + 1e-9, marker="o", label=method)
        axes[1].plot(x, method_data["time_ratio"], marker="o", label=method)
    axes[0].set_ylabel("Median final-RMSE gap to best (%)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Median solver-time ratio to fastest")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Global coordinate offset (0 shown at 1)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle("Global-coordinate numerical stress")
    figure.tight_layout()
    figure.savefig(output / "coordinate_offset_stress.png", dpi=220)
    plt.close(figure)


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        vertex_count = None
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count is missing: {path}")
        rows = []
        for _ in range(vertex_count):
            values = stream.readline().split()
            rows.append([float(values[0]), float(values[1]), float(values[2])])
    return np.asarray(rows, dtype=np.float64)


def similarity_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.dot(singular, sign) / max(variance, 1e-30))
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale


def diagnostic_accuracy(root: Path, selection: pd.DataFrame) -> pd.DataFrame:
    diagnostic_root = root / "diagnostic_results"
    if not (diagnostic_root / "summary.csv").exists():
        return pd.DataFrame()
    rows = []
    for record in selection.itertuples(index=False):
        original = root / "datasets" / record.base_dataset / "original"
        true_points = np.loadtxt(original / "XYZ.txt")
        camera_file = next(original.glob("Cam*.txt"))
        true_cameras = np.loadtxt(camera_file)[:, 3:6]
        for method in METHOD_ORDER:
            run_root = (
                diagnostic_root
                / record.base_dataset
                / record.quality_dataset
                / method
            )
            pose_path = run_root / "FinalPose.txt"
            point_path = run_root / "Final3D.ply"
            if not pose_path.exists() or not point_path.exists():
                continue
            estimated_cameras = np.loadtxt(pose_path)[:, 3:6]
            estimated_points = read_ply_vertices(point_path)
            source = np.vstack([estimated_cameras, estimated_points])
            target = np.vstack([true_cameras, true_points])
            aligned, scale = similarity_align(source, target)
            camera_aligned = aligned[: len(true_cameras)]
            point_aligned = aligned[len(true_cameras) :]
            camera_error = np.linalg.norm(camera_aligned - true_cameras, axis=1)
            point_error = np.linalg.norm(point_aligned - true_points, axis=1)
            rows.append(
                {
                    "quality_dataset": record.quality_dataset,
                    "method": method,
                    "similarity_scale": scale,
                    "camera_center_rmse": float(np.sqrt(np.mean(camera_error**2))),
                    "point_rmse": float(np.sqrt(np.mean(point_error**2))),
                    "point_median_error": float(np.median(point_error)),
                    "point_p95_error": float(np.percentile(point_error, 95)),
                }
            )
    return pd.DataFrame(rows)


def write_conclusions(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    family: pd.DataFrame,
    accuracy: pd.DataFrame,
    output: Path,
) -> None:
    indexed = summary.set_index("method")
    a0 = indexed.loc["A0-XYZ-W"]
    a2mc = indexed.loc["A2-Parallax-Mc"]
    a2mw = indexed.loc["A2-Parallax-Mw"]
    idp = indexed.loc["A1-XYInvZ-Ac"]
    invz = indexed.loc["A0-XYInvZ-W"]
    success_ranking = summary.sort_values(
        ["solution_success_rate", "ceres_convergence_rate"],
        ascending=False,
    )
    best_success = success_ranking.iloc[0]
    a2_frame_efficiency = (
        "Mw"
        if a2mw.median_time_ratio_best < a2mc.median_time_ratio_best
        else "Mc"
    )

    parallax_family = family[family["family"].eq("parallax_depth")].set_index("method")
    track_family = family[family["family"].eq("track_stress")].set_index("method")
    coordinate_1e9 = data[
        data["family"].eq("coordinate_stress")
        & np.isclose(data["coordinate_offset"], 1e9)
    ]
    coordinate_success = (
        coordinate_1e9.groupby("method")["solution_success"].mean() * 100.0
    )

    lines = [
        "# Controlled Synthetic BA Benchmark",
        "",
        "## Scope",
        "",
        f"- {data['quality_dataset'].nunique()} controlled scenes and {len(data)} method runs.",
        "- 8 cameras, 200 points, 0.5 px image noise, and paired physical initialization.",
        "- Stress factors: parallax, range error, camera-pose error, track length, and global coordinate offset.",
        "- Solution success means a usable finite solution within 1% of the best final RMSE for the same scene.",
        "",
        "## Main Findings",
        "",
        (
            f"- A2-Parallax-Mc achieved {a2mc.solution_success_rate:.1f}% solution success "
            f"and {a2mc.ceres_convergence_rate:.1f}% formal Ceres convergence. "
            f"A0-XYZ-W achieved {a0.solution_success_rate:.1f}% and "
            f"{a0.ceres_convergence_rate:.1f}%, respectively."
        ),
        (
            f"- In the parallax-depth family, A2-Parallax-Mc reached "
            f"{parallax_family.loc['A2-Parallax-Mc', 'solution_success_rate']:.1f}% "
            f"solution success, compared with "
            f"{parallax_family.loc['A0-XYZ-W', 'solution_success_rate']:.1f}% for A0-XYZ-W."
        ),
        (
            f"- A2-Parallax-Mc required a median solver-time ratio of "
            f"{a2mc.median_time_ratio_best:.2f}x relative to the fastest method in each scene. "
            f"A2-Parallax-Mw required {a2mw.median_time_ratio_best:.2f}x."
        ),
        (
            f"- The highest broad solution-success rate was obtained by "
            f"{best_success.method} ({best_success.solution_success_rate:.1f}%). "
            f"A0-XYInvZ-W and A1-XYInvZ-Ac reached "
            f"{invz.solution_success_rate:.1f}% and {idp.solution_success_rate:.1f}%, respectively."
        ),
        (
            f"- Under short-track stress, A2-Parallax-Mc reached "
            f"{track_family.loc['A2-Parallax-Mc', 'solution_success_rate']:.1f}% success, "
            f"while A1-XYInvZ-Ac reached "
            f"{track_family.loc['A1-XYInvZ-Ac', 'solution_success_rate']:.1f}%."
        ),
        (
            f"- At a 1e9-unit global offset, A0-XYZ-W success was "
            f"{coordinate_success.get('A0-XYZ-W', math.nan):.1f}%, and "
            f"A2-Parallax-Mc success was "
            f"{coordinate_success.get('A2-Parallax-Mc', math.nan):.1f}%."
        ),
        "",
        "## Interpretation",
        "",
        "- The synthetic evidence does not support universal A2-Parallax superiority.",
        "- A2-Parallax-Mc expands the reliable solution region for some depth-uncertain cases, especially compared with XYZ, but inverse-depth representations are at least as robust overall in this suite.",
        (
            f"- With all three object-point variables registered as one Schur block, "
            f"the {a2_frame_efficiency} frame had the lower overall median time ratio "
            f"({min(a2mc.median_time_ratio_best, a2mw.median_time_ratio_best):.2f}x versus "
            f"{max(a2mc.median_time_ratio_best, a2mw.median_time_ratio_best):.2f}x). "
            "The frame effect is therefore empirical and scenario-dependent, not an inherent split-block penalty."
        ),
        "- Small parallax alone is not sufficient to guarantee an A2 advantage. Motion topology, track length, anchor geometry, and the type of initialization error matter jointly.",
        "- Formal Ceres convergence and solution quality must be reported separately: several methods reached comparable final RMSE at the iteration limit.",
        "",
        "## Files",
        "",
        "- `method_summary.csv`: overall method statistics.",
        "- `family_summary.csv`: statistics by controlled stress family.",
        "- `diagnostic_selection.csv`: representative boundary cases.",
        "- PNG files: robustness, track-length, coordinate-offset, and parallax-depth plots.",
    ]
    if not accuracy.empty:
        best_accuracy = (
            accuracy.groupby("method")["point_rmse"].median().sort_values().head(5)
        )
        lines.extend(
            [
                "",
                "## Diagnostic Truth Alignment",
                "",
                "Median aligned point RMSE on selected boundary cases:",
                "",
            ]
        )
        for method, value in best_accuracy.items():
            lines.append(f"- {method}: {value:.6g} scene units")
    (output / "conclusions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chinese_conclusions(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    family: pd.DataFrame,
    accuracy: pd.DataFrame,
    scale_data: pd.DataFrame,
    scale_summary: pd.DataFrame,
    output: Path,
) -> None:
    indexed = summary.set_index("method")
    a0 = indexed.loc["A0-XYZ-W"]
    invz = indexed.loc["A0-XYInvZ-W"]
    idp = indexed.loc["A1-XYInvZ-Ac"]
    a2mc = indexed.loc["A2-Parallax-Mc"]
    a2mw = indexed.loc["A2-Parallax-Mw"]
    success_ranking = summary.sort_values(
        ["solution_success_rate", "ceres_convergence_rate"],
        ascending=False,
    )
    best_success = success_ranking.iloc[0]
    faster_a2 = "A2-Parallax-Mw" if (
        a2mw.median_time_ratio_best < a2mc.median_time_ratio_best
    ) else "A2-Parallax-Mc"
    parallax_family = family[family["family"].eq("parallax_depth")].set_index("method")
    track_family = family[family["family"].eq("track_stress")].set_index("method")
    coordinate = data[
        data["family"].eq("coordinate_stress")
        & np.isclose(data["coordinate_offset"], 1e9)
    ]
    coordinate_success = coordinate.groupby("method")["solution_success"].mean() * 100

    lines = [
        "# 可控合成BA实验结论",
        "",
        "## 实验范围",
        "",
        f"- 边界扫描：{data['quality_dataset'].nunique()}个场景、{len(data)}次方法运行。",
        "- 基础规模：8个相机、200个物点、0.5 px像点噪声。",
        "- 控制变量：交会角、初始距离倍率、相机位姿误差、轨迹长度和全局坐标偏移。",
        "- 所有方法使用完全相同的物理初值，再转换为各自的物点参数。",
        "- `solution success`定义为最终RMSE不超过同一场景最优方法的1%。",
        (
            f"- 当前分析包含{len(accuracy)}条边界案例真值对齐记录。"
            if not accuracy.empty
            else "- 本轮只运行clean核心指标；一阶、二阶和Sim(3)真值对齐留待代表性案例诊断阶段。"
        ),
        "",
        "## 核心结果",
        "",
        (
            f"1. **A2方法在总体鲁棒性上进入第一梯队，但优势并非覆盖所有场景。** "
            f"总体成功率最高的是{best_success.method}（"
            f"{best_success.solution_success_rate:.1f}%）；A2-Parallax-Mc为"
            f"{a2mc.solution_success_rate:.1f}%，A0-XYZ-W为"
            f"{a0.solution_success_rate:.1f}%。"
        ),
        (
            f"2. **Parallax在距离初始化误差实验中有局部优势。** "
            f"在`parallax_depth`场景中，A2-Parallax-Mc成功率为"
            f"{parallax_family.loc['A2-Parallax-Mc', 'solution_success_rate']:.1f}%，"
            f"A0-XYZ-W为{parallax_family.loc['A0-XYZ-W', 'solution_success_rate']:.1f}%。"
            "这说明Parallax确实能扩大部分深度不确定场景的有效解区域。"
        ),
        (
            f"3. **A2具有明显的迭代优势，但不总是单场景最快。** "
            f"A2-Parallax-Mc与Mw的总体中位迭代数分别为"
            f"{a2mc.median_iterations:.1f}和{a2mw.median_iterations:.1f}；"
            f"中位耗时比分别为{a2mc.median_time_ratio_best:.2f}倍和"
            f"{a2mw.median_time_ratio_best:.2f}倍。"
        ),
        (
            f"4. **短轨迹下双锚点优势受可用观测数限制。** 在track stress中，"
            f"A1-XYInvZ-Ac成功率为"
            f"{track_family.loc['A1-XYInvZ-Ac', 'solution_success_rate']:.1f}%，"
            f"A2-Parallax-Mc为"
            f"{track_family.loc['A2-Parallax-Mc', 'solution_success_rate']:.1f}%。"
            "因此应将轨迹长度作为适用区域变量，而不是只给出总体排名。"
        ),
        (
            f"5. **统一三维物点参数块后，Mc/Mw的效率关系发生了根本变化。** "
            f"本轮总体上{faster_a2}的中位耗时比更低；"
            "这说明此前Mw的巨大耗时主要来自错误的参数块拆分，而不是参数化本身。"
        ),
        (
            f"6. **极大坐标偏移暴露了Parallax实现的数值敏感性。** "
            f"在1e9量级坐标偏移下，A0-XYZ-W的成功率为"
            f"{coordinate_success.get('A0-XYZ-W', math.nan):.1f}%，"
            f"A2-Parallax-Mc为"
            f"{coordinate_success.get('A2-Parallax-Mc', math.nan):.1f}%。"
            "这一结果需要结合PVL真实大坐标数据进一步核验，但已表明应检查坐标中心化和锚点基线计算。"
        ),
        (
            f"7. **Ceres终止状态不能等同于解质量。** A2-Parallax-Mc正式"
            f"CONVERGENCE比例为{a2mc.ceres_convergence_rate:.1f}%，"
            "但部分达到100次迭代上限的结果仍与最优RMSE接近，因此论文应同时报告"
            "termination type、最终RMSE和达到阈值所需迭代数。"
        ),
        "",
        "## 放大验证",
        "",
    ]
    if not scale_data.empty:
        scale_indexed = scale_summary.set_index("method")
        scale_a0 = scale_indexed.loc["A0-XYZ-W"]
        scale_a2mc = scale_indexed.loc["A2-Parallax-Mc"]
        scale_a2mw = scale_indexed.loc["A2-Parallax-Mw"]
        lines.extend(
            [
                f"- 6个代表场景扩大到12相机、600物点，共{len(scale_data)}次运行。",
                (
                    f"- A2-Parallax-Mc在这组刻意挑选的困难案例中成功率为"
                    f"{scale_a2mc.solution_success_rate:.1f}%，中位时间比为"
                    f"{scale_a2mc.median_time_ratio_best:.2f}倍；A0-XYZ-W成功率为"
                    f"{scale_a0.solution_success_rate:.1f}%。"
                ),
                (
                    f"- 放大场景中A2-Parallax-Mc与Mw的中位时间比分别为"
                    f"{scale_a2mc.median_time_ratio_best:.2f}倍和"
                    f"{scale_a2mw.median_time_ratio_best:.2f}倍。"
                ),
                "- 在极小视差和大距离误差案例中，A2-Mc能够达到接近最优的最终RMSE；"
                "但在短轨迹、大坐标和严重位姿误差案例中，它没有保持这一优势。",
            ]
        )
    lines.extend(
        [
            "",
            "## 真值对齐结果",
            "",
            "- 弱几何场景中，不同方法即使得到接近的重投影RMSE，Sim(3)对齐后的三维误差仍可能明显不同。",
            "- 因此论文不能只用最终像点RMSE判断参数化优劣，应同时报告相机中心误差和三维点误差。",
        ]
    )
    if not accuracy.empty:
        medians = accuracy.groupby("method")["point_rmse"].median().sort_values()
        lines.append("")
        lines.append("所选边界案例中位三维点RMSE最低的五种方法：")
        lines.append("")
        for method, value in medians.head(5).items():
            lines.append(f"- {method}: {value:.6g} 场景单位")
        lines.append("")
        lines.append(
            "注意：这些案例是按差异最大的边界场景选择的，不应作为总体精度排名。"
        )
    lines.extend(
        [
            "",
            "## 建议写入论文的结论",
            "",
            "> 两锚点Parallax参数化在受控实验中表现出较高的总体成功率和显著的"
            "迭代数优势，尤其适用于距离初值不确定和弱深度约束场景；但其收益仍依赖"
            "运动拓扑、轨迹长度、锚点几何和坐标数值尺度。Mc与Mw必须在相同的三维"
            "Schur物点块实现下比较，不能把参数块拆分造成的性能损失归因于参数化本身。",
            "",
            "## 输出文件",
            "",
            "- `method_summary.csv`：总体统计。",
            "- `family_summary.csv`：按实验族统计。",
            (
                "- `diagnostic_solution_accuracy.csv`：边界案例真值对齐误差。"
                if not accuracy.empty
                else "- `diagnostic_selection.csv`：后续机制诊断的代表性场景。"
            ),
            "- `scale_validation_heatmaps.png`：放大场景的解质量与耗时热图。",
            "- 其余PNG：鲁棒性、交会角—距离误差、短轨迹和大坐标分析。",
        ]
    )
    (output / "conclusions_zh.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\zuo\projects\CEP\synthetic_benchmark"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    data = load_results(root)
    summary = method_summary(data)
    family = family_summary(data)
    selection = select_diagnostic_cases(data)
    accuracy = diagnostic_accuracy(root, selection)
    scale_data, scale_summary = load_scale_validation(root)

    data.to_csv(output / "paired_results.csv", index=False)
    summary.to_csv(output / "method_summary.csv", index=False)
    family.to_csv(output / "family_summary.csv", index=False)
    selection.to_csv(output / "diagnostic_selection.csv", index=False)
    if not accuracy.empty:
        accuracy.to_csv(output / "diagnostic_solution_accuracy.csv", index=False)
    if not scale_data.empty:
        scale_data.to_csv(output / "scale_paired_results.csv", index=False)
        scale_summary.to_csv(output / "scale_method_summary.csv", index=False)

    plot_overall(summary, output)
    plot_parallax_heatmaps(data, output, "lateral")
    plot_parallax_heatmaps(data, output, "forward")
    plot_track_stress(data, output)
    plot_coordinate_stress(data, output)
    plot_scale_validation(scale_data, output)
    write_conclusions(data, summary, family, accuracy, output)
    write_chinese_conclusions(
        data,
        summary,
        family,
        accuracy,
        scale_data,
        scale_summary,
        output,
    )

    print(f"Cases: {data['quality_dataset'].nunique()}")
    print(f"Runs: {len(data)}")
    print(f"Diagnostic selections: {len(selection)}")
    print(f"Output: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
