#!/usr/bin/env python3
"""Generate paper-ready benchmark tables and figures from CEP outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

LEGACY_METHOD_ALIASES = {
    "A1-IDP-Civera-Ac": "A1-XYInvZ-Ac",
}


def canonical_method_name(value: object) -> str:
    name = str(value)
    return LEGACY_METHOD_ALIASES.get(name, name)


def canonicalize_method_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "method" not in frame.columns:
        return frame
    result = frame.copy()
    result["method"] = result["method"].map(canonical_method_name)
    return result

QUALITY_PATTERN = re.compile(
    r"-(?P<mode>init|joint)-rmse(?P<integer>\d+)p(?P<fraction>\d+)px$"
)


def parse_quality_dataset(name: str) -> dict[str, object]:
    match = QUALITY_PATTERN.search(str(name))
    if not match:
        return {
            "quality_mode": "unknown",
            "quality_rmse_px": np.nan,
            "quality_label": str(name),
        }
    fraction = match.group("fraction")
    rmse = float(f"{int(match.group('integer'))}.{fraction}")
    mode = match.group("mode")
    mode_label = "pose" if mode == "init" else "joint"
    return {
        "quality_mode": mode,
        "quality_rmse_px": rmse,
        "quality_label": f"{rmse:g} px ({mode_label})",
    }


def add_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    parsed = pd.DataFrame(
        [
            parse_quality_dataset(name)
            for name in frame["quality_dataset"].astype(str)
        ],
        index=frame.index,
    )
    for column in parsed:
        frame[column] = parsed[column]
    return frame


def quality_labels(frame: pd.DataFrame) -> list[str]:
    quality = (
        frame[["quality_label", "quality_rmse_px"]]
        .drop_duplicates()
        .sort_values(["quality_rmse_px", "quality_label"])
    )
    return quality["quality_label"].astype(str).tolist()


def method_colors() -> dict[str, object]:
    palette = plt.get_cmap("tab20")
    return {
        method: palette(index % 20)
        for index, method in enumerate(METHOD_ORDER)
    }


def save_heatmap(
    pivot: pd.DataFrame,
    output: Path,
    colorbar_label: str,
    value_format: str = ".2f",
) -> None:
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    width = max(7.5, 1.15 * len(pivot.columns))
    height = max(5.0, 0.42 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.tick_params(axis="x", rotation=35)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    if values.size <= 120:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    continue
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "black" if luminance > 0.55 else "white"
                ax.text(
                    column,
                    row,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_quality_lineplot(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    output: Path,
    log_scale: bool = False,
) -> None:
    if frame.empty or column not in frame:
        return
    colors = method_colors()
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for method in ordered_methods(frame):
        group = (
            frame.loc[frame["method"] == method]
            .sort_values("quality_rmse_px")
            .dropna(subset=["quality_rmse_px", column])
        )
        if group.empty:
            continue
        ax.plot(
            group["quality_rmse_px"],
            group[column],
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=colors.get(method),
            label=method,
        )
    ax.set_xscale("log")
    if log_scale:
        positive = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not positive.empty and (positive > 0).all():
            ax.set_yscale("log")
    ax.set_xlabel("Target initial reprojection RMSE (px)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        help="Optional root containing per-run convergence.txt files.",
    )
    parser.add_argument("--max-curves", type=int, default=12)
    parser.add_argument(
        "--solution-rmse-relative-tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum final-RMSE excess relative to the best matched run for "
            "classifying an equivalent solution."
        ),
    )
    return parser.parse_args()


def ordered_methods(frame: pd.DataFrame) -> list[str]:
    present = set(frame["method"].dropna().astype(str))
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def save_boxplot(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    output: Path,
    log_scale: bool = False,
) -> None:
    methods = ordered_methods(frame)
    values = [
        frame.loc[(frame["method"] == method) & frame[column].notna(), column].to_numpy()
        for method in methods
    ]
    if not any(len(value) for value in values):
        return

    fig, ax = plt.subplots(figsize=(max(7.0, len(methods) * 0.8), 4.8))
    ax.boxplot(values, tick_labels=methods, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    if log_scale and all(np.all(value > 0.0) for value in values if len(value)):
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def generate_aggregate_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    solution_rmse_relative_tolerance: float,
) -> None:
    frame = add_quality_columns(frame)
    frame["execution_success"] = frame["status"].eq("ok")
    numeric_columns = [
        "solver_time_sec",
        "linear_solver_time_sec",
        "iterations",
        "final_rmse_px",
        "final_gradient_max_norm",
        "accepted_steps",
        "rejected_steps",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["termination_type"] = pd.to_numeric(
        frame["termination_type"], errors="coerce"
    )
    frame["solver_converged"] = (
        frame["execution_success"] & frame["termination_type"].eq(0)
    )
    frame["best_final_rmse_px"] = frame.groupby("quality_dataset")[
        "final_rmse_px"
    ].transform("min")
    frame["final_rmse_ratio_to_best"] = (
        frame["final_rmse_px"] / frame["best_final_rmse_px"]
    )
    frame["solution_success"] = (
        frame["solver_converged"]
        & frame["final_rmse_px"].notna()
        & (
            frame["final_rmse_ratio_to_best"]
            <= 1.0 + max(0.0, solution_rmse_relative_tolerance)
        )
    )
    frame.loc[~frame["solution_success"]].to_csv(
        output_dir / "solution_failures.csv", index=False
    )

    rows = []
    for method, group in frame.groupby("method", sort=False):
        successful = group[group["solution_success"]]
        rows.append(
            {
                "method": method,
                "runs": len(group),
                "execution_success_rate": group["execution_success"].mean(),
                "solver_convergence_rate": group["solver_converged"].mean(),
                "solution_success_rate": group["solution_success"].mean(),
                "solution_failure_rate": 1.0 - group["solution_success"].mean(),
                "median_solver_time_sec": successful["solver_time_sec"].median(),
                "median_linear_solver_time_sec": successful[
                    "linear_solver_time_sec"
                ].median(),
                "median_iterations": successful["iterations"].median(),
                "median_final_rmse_px": successful["final_rmse_px"].median(),
                "median_final_gradient_max_norm": successful[
                    "final_gradient_max_norm"
                ].median(),
            }
        )

    summary = pd.DataFrame(rows)
    method_order = ordered_methods(frame)
    summary["method"] = pd.Categorical(
        summary["method"], categories=method_order, ordered=True
    )
    summary = summary.sort_values("method")
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    summary.to_latex(
        output_dir / "method_summary.tex",
        index=False,
        float_format=lambda value: f"{value:.4g}",
        caption="Aggregate object-point parameterization benchmark results.",
        label="tab:parameterization_benchmark_summary",
    )

    by_quality = (
        frame.groupby(
            [
                "method",
                "quality_dataset",
                "quality_label",
                "quality_rmse_px",
                "quality_mode",
            ],
            as_index=False,
        )
        .agg(
            execution_success_rate=("execution_success", "mean"),
            solver_convergence_rate=("solver_converged", "mean"),
            solution_success_rate=("solution_success", "mean"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_linear_solver_time_sec=("linear_solver_time_sec", "median"),
            median_iterations=("iterations", "median"),
            median_final_rmse_px=("final_rmse_px", "median"),
            median_final_gradient_max_norm=("final_gradient_max_norm", "median"),
        )
        .sort_values(["quality_rmse_px", "method"])
    )
    by_quality.to_csv(output_dir / "method_summary_by_quality.csv", index=False)
    save_quality_lineplot(
        by_quality,
        "median_solver_time_sec",
        "Median Ceres solver time (s)",
        output_dir / "solver_time_by_quality.png",
        log_scale=True,
    )
    save_quality_lineplot(
        by_quality,
        "median_iterations",
        "Median LM iterations",
        output_dir / "iterations_by_quality.png",
    )
    save_quality_lineplot(
        by_quality,
        "median_final_rmse_px",
        "Median final reprojection RMSE (px)",
        output_dir / "final_rmse_by_quality.png",
        log_scale=True,
    )

    fig, ax = plt.subplots(figsize=(max(7.0, len(summary) * 0.8), 4.5))
    ax.bar(
        summary["method"].astype(str),
        summary["solution_failure_rate"],
        color="#C44E52",
    )
    ax.set_ylabel("Solution failure rate")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "failure_rate.png", dpi=300)
    plt.close(fig)

    solver_failure = (
        frame.groupby("method", as_index=False)["solver_converged"]
        .mean()
        .assign(solver_failure_rate=lambda data: 1.0 - data["solver_converged"])
    )
    solver_failure["method"] = pd.Categorical(
        solver_failure["method"], categories=method_order, ordered=True
    )
    solver_failure = solver_failure.sort_values("method")
    fig, ax = plt.subplots(figsize=(max(7.0, len(solver_failure) * 0.8), 4.5))
    ax.bar(
        solver_failure["method"].astype(str),
        solver_failure["solver_failure_rate"],
        color="#8172B2",
    )
    ax.set_ylabel("Ceres non-convergence rate")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "solver_failure_rate.png", dpi=300)
    plt.close(fig)

    successful = frame[frame["solution_success"]].copy()
    save_boxplot(
        successful,
        "solver_time_sec",
        "Ceres solver time (s)",
        output_dir / "solver_time_boxplot.png",
        log_scale=True,
    )
    save_boxplot(
        successful,
        "linear_solver_time_sec",
        "Linear solver time (s)",
        output_dir / "linear_solver_time_boxplot.png",
        log_scale=True,
    )
    save_boxplot(
        successful,
        "iterations",
        "LM iterations",
        output_dir / "iterations_boxplot.png",
    )
    save_boxplot(
        successful,
        "final_rmse_px",
        "Final reprojection RMSE (px)",
        output_dir / "final_rmse_boxplot.png",
        log_scale=True,
    )

    if successful.empty:
        return
    successful["runtime_rank"] = successful.groupby("quality_dataset")[
        "solver_time_sec"
    ].rank(method="average")
    ranks = (
        successful.groupby("method", as_index=False)["runtime_rank"]
        .mean()
        .rename(columns={"runtime_rank": "mean_runtime_rank"})
    )
    ranks["method"] = pd.Categorical(
        ranks["method"], categories=method_order, ordered=True
    )
    ranks = ranks.sort_values("method")
    ranks.to_csv(output_dir / "method_mean_runtime_rank.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(7.0, len(ranks) * 0.8), 4.5))
    ax.bar(ranks["method"].astype(str), ranks["mean_runtime_rank"], color="#4C72B0")
    ax.set_ylabel("Mean runtime rank (lower is better)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_runtime_rank.png", dpi=300)
    plt.close(fig)


def generate_convergence_figure(
    diagnostic_root: Path, output_dir: Path, max_curves: int
) -> None:
    files = sorted(diagnostic_root.rglob("convergence.txt"))[:max_curves]
    if not files:
        return

    curves = []
    for path in files:
        frame = pd.read_csv(path)
        curves.append(
            {
                "frame": frame,
                "method": canonical_method_name(path.parent.name),
                "quality_dataset": path.parent.parent.name,
            }
        )

    quality_frame = add_quality_columns(
        pd.DataFrame(
            {
                "quality_dataset": [curve["quality_dataset"] for curve in curves]
            }
        )
    )
    quality_order = quality_labels(quality_frame)
    representative_quality = list(
        dict.fromkeys(
            [
                quality_order[0],
                quality_order[len(quality_order) // 2],
                quality_order[-1],
            ]
        )
    )
    quality_lookup = {
        row["quality_dataset"]: row["quality_label"]
        for _, row in quality_frame.drop_duplicates("quality_dataset").iterrows()
    }
    colors = method_colors()

    gain_ratio_rows = []
    for curve in curves:
        trial = curve["frame"].loc[
            curve["frame"]["iteration"] > 0,
            ["gain_ratio", "step_valid", "step_successful"],
        ].copy()
        gain_ratio = pd.to_numeric(trial["gain_ratio"], errors="coerce").dropna()
        gain_ratio_rows.append(
            {
                "quality_dataset": curve["quality_dataset"],
                "method": curve["method"],
                "trial_steps": len(trial),
                "negative_gain_ratio_steps": int((gain_ratio < 0.0).sum()),
                "minimum_gain_ratio": gain_ratio.min() if not gain_ratio.empty else np.nan,
                "maximum_gain_ratio": gain_ratio.max() if not gain_ratio.empty else np.nan,
                "invalid_steps": int(
                    (
                        pd.to_numeric(trial["step_valid"], errors="coerce")
                        .fillna(0)
                        .eq(0)
                    ).sum()
                ),
                "unsuccessful_steps": int(
                    (
                        pd.to_numeric(trial["step_successful"], errors="coerce")
                        .fillna(0)
                        .eq(0)
                    ).sum()
                ),
            }
        )
    gain_ratio_summary = add_quality_columns(pd.DataFrame(gain_ratio_rows)).sort_values(
        ["quality_rmse_px", "method"]
    )
    gain_ratio_summary.to_csv(
        output_dir / "gain_ratio_summary.csv", index=False
    )
    quality_selection = (
        gain_ratio_summary.groupby(
            ["quality_label", "quality_rmse_px", "quality_mode"],
            as_index=False,
        )
        .agg(
            runs=("method", "count"),
            negative_gain_ratio_steps=("negative_gain_ratio_steps", "sum"),
            minimum_gain_ratio=("minimum_gain_ratio", "min"),
        )
        .sort_values("quality_rmse_px")
    )
    quality_selection["selected_for_convergence_figures"] = (
        quality_selection["negative_gain_ratio_steps"] > 0
    )
    quality_selection.to_csv(
        output_dir / "diagnostic_quality_selection.csv", index=False
    )
    discriminative_quality = quality_selection.loc[
        quality_selection["selected_for_convergence_figures"],
        "quality_label",
    ].astype(str).tolist()
    if not discriminative_quality:
        discriminative_quality = representative_quality

    def plot_convergence_panels(
        selected_quality: list[str],
        output: Path,
        selected_methods: list[str] | None = None,
    ) -> None:
        method_filter = set(selected_methods) if selected_methods else None
        fig, axes = plt.subplots(
            len(selected_quality),
            3,
            figsize=(14.5, 3.7 * len(selected_quality)),
            squeeze=False,
        )
        for row, quality in enumerate(selected_quality):
            for curve in curves:
                if quality_lookup.get(curve["quality_dataset"]) != quality:
                    continue
                if method_filter is not None and curve["method"] not in method_filter:
                    continue
                frame = curve["frame"]
                method = curve["method"]
                style = {
                    "linewidth": 1.2,
                    "marker": "o",
                    "markersize": 2.5,
                    "markevery": max(1, len(frame) // 20),
                    "color": colors.get(method),
                    "label": method,
                }
                axes[row, 0].plot(frame["iteration"], frame["rmse_px"], **style)
                axes[row, 1].plot(
                    frame["iteration"], frame["gradient_max_norm"], **style
                )
                trial = frame[frame["iteration"] > 0]
                axes[row, 2].plot(
                    trial["iteration"], trial["gain_ratio"], **style
                )
            axes[row, 0].set_ylabel(f"{quality}\nRMSE (px)")
            axes[row, 0].set_yscale("log")
            axes[row, 1].set_ylabel("Max gradient component")
            axes[row, 1].set_yscale("log")
            axes[row, 2].set_ylabel("LM gain ratio")
            axes[row, 2].set_ylim(-1.0, 1.05)
            axes[row, 2].axhline(0.0, color="#666666", linewidth=0.8)
            axes[row, 2].axhline(
                1.0, color="#666666", linewidth=0.8, linestyle="--"
            )
            for ax in axes[row]:
                ax.set_xlabel("Iteration")
                ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                fontsize=7,
                ncol=1,
                loc="center left",
                bbox_to_anchor=(0.99, 0.5),
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
        else:
            fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    method_groups = {
        "A0": [method for method in METHOD_ORDER if method.startswith("A0-")],
        "A1-Ac": [
            method
            for method in METHOD_ORDER
            if method.startswith("A1-") and method.endswith("-Ac")
        ],
        "A1-Aw": [
            method
            for method in METHOD_ORDER
            if method.startswith("A1-") and method.endswith("-Aw")
        ],
        "A2": [method for method in METHOD_ORDER if method.startswith("A2-")],
    }
    aggregate_summary = canonicalize_method_column(
        pd.read_csv(output_dir / "method_summary.csv")
    )
    method_to_group = {
        method: group
        for group, methods in method_groups.items()
        for method in methods
    }
    aggregate_summary["comparison_group"] = aggregate_summary["method"].map(
        method_to_group
    )
    winner_rows = []
    for comparison_group in ["A0", "A1-Ac", "A1-Aw", "A2"]:
        candidates = aggregate_summary.loc[
            aggregate_summary["comparison_group"] == comparison_group
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            [
                "solution_success_rate",
                "solver_convergence_rate",
                "median_iterations",
                "median_solver_time_sec",
            ],
            ascending=[False, False, True, True],
            na_position="last",
        )
        winner = candidates.iloc[0].copy()
        winner["selection_priority"] = (
            "solution success, solver convergence, iterations, solver time"
        )
        winner_rows.append(winner)
    group_winners = pd.DataFrame(winner_rows)
    group_winners.to_csv(output_dir / "group_winners.csv", index=False)
    winner_methods = group_winners["method"].astype(str).tolist()

    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence.png",
        winner_methods,
    )
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_group_winners.png",
        winner_methods,
    )
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_A0.png",
        method_groups["A0"],
    )
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_A1_Ac.png",
        method_groups["A1-Ac"],
    )
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_A1_Aw.png",
        method_groups["A1-Aw"],
    )
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_A2.png",
        method_groups["A2"],
    )
    frame_ablation_methods = [
        "A1-XYZ-Ac",
        "A1-XYZ-Aw",
        "A1-XYInvZ-Ac",
        "A1-XYInvZ-Aw",
        "A1-SphRange-Ac",
        "A1-SphRange-Aw",
        "A1-SphInvRange-Ac",
        "A1-SphInvRange-Aw",
        "A2-Parallax-Mc",
        "A2-Parallax-Mw",
    ]
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_frame_ablation.png",
        frame_ablation_methods,
    )
    pd.DataFrame(
        [
            {
                "parameterization_family": "XYZ",
                "camera_frame_method": "A1-XYZ-Ac",
                "world_parallel_method": "A1-XYZ-Aw",
            },
            {
                "parameterization_family": "XYInvZ",
                "camera_frame_method": "A1-XYInvZ-Ac",
                "world_parallel_method": "A1-XYInvZ-Aw",
            },
            {
                "parameterization_family": "SphRange",
                "camera_frame_method": "A1-SphRange-Ac",
                "world_parallel_method": "A1-SphRange-Aw",
            },
            {
                "parameterization_family": "SphInvRange",
                "camera_frame_method": "A1-SphInvRange-Ac",
                "world_parallel_method": "A1-SphInvRange-Aw",
            },
            {
                "parameterization_family": "Parallax",
                "camera_frame_method": "A2-Parallax-Mc",
                "world_parallel_method": "A2-Parallax-Mw",
            },
        ]
    ).to_csv(output_dir / "frame_ablation_pairs.csv", index=False)
    plot_convergence_panels(
        discriminative_quality,
        output_dir / "diagnostic_convergence_all_methods.png",
    )


def generate_point_conditioning_outputs(
    diagnostic_root: Path, output_dir: Path
) -> None:
    frames = []
    for path in sorted(diagnostic_root.rglob("point_block_conditioning.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["method"] = canonical_method_name(path.parent.name)
        frame["quality_dataset"] = path.parent.parent.name
        frames.append(frame)
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = add_quality_columns(combined)
    combined["condition_number"] = pd.to_numeric(
        combined["condition_number"], errors="coerce"
    )
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined.to_csv(output_dir / "point_block_conditioning_samples.csv", index=False)

    summary = (
        combined.groupby("method", as_index=False)
        .agg(
            samples=("point_index", "count"),
            full_rank_fraction=("numerical_rank", lambda values: (values == 3).mean()),
            median_condition_number=("condition_number", "median"),
            q90_condition_number=("condition_number", lambda values: values.quantile(0.9)),
            median_eigen_min=("eigen_min", "median"),
            median_eigen_max=("eigen_max", "median"),
        )
    )
    summary.to_csv(output_dir / "point_block_conditioning_summary.csv", index=False)
    by_quality = (
        combined.groupby(
            ["method", "quality_label", "quality_rmse_px"], as_index=False
        )
        .agg(
            samples=("point_index", "count"),
            full_rank_fraction=("numerical_rank", lambda values: (values == 3).mean()),
            median_condition_number=("condition_number", "median"),
            q90_condition_number=("condition_number", lambda values: values.quantile(0.9)),
        )
        .sort_values(["quality_rmse_px", "method"])
    )
    by_quality.to_csv(
        output_dir / "point_block_conditioning_by_quality.csv", index=False
    )
    heatmap = by_quality.pivot(
        index="method",
        columns="quality_label",
        values="median_condition_number",
    )
    heatmap = heatmap.reindex(
        index=ordered_methods(combined),
        columns=quality_labels(combined),
    )
    save_heatmap(
        np.log10(heatmap),
        output_dir / "point_block_conditioning_by_quality.png",
        "log10 median point-block condition number",
    )
    save_boxplot(
        combined.dropna(subset=["condition_number"]),
        "condition_number",
        "Point-block condition number",
        output_dir / "point_block_conditioning_boxplot.png",
        log_scale=True,
    )


def generate_schur_outputs(diagnostic_root: Path, output_dir: Path) -> None:
    summaries = []
    spectra = []
    for path in sorted(diagnostic_root.rglob("schur_summary.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["method"] = canonical_method_name(path.parent.name)
        frame["quality_dataset"] = path.parent.parent.name
        summaries.append(frame)
    for path in sorted(diagnostic_root.rglob("schur_spectrum.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["method"] = canonical_method_name(path.parent.name)
        frame["quality_dataset"] = path.parent.parent.name
        spectra.append(frame)
    if not summaries:
        return

    summary = add_quality_columns(pd.concat(summaries, ignore_index=True))
    summary["condition_number"] = pd.to_numeric(
        summary["condition_number"], errors="coerce"
    )
    summary = summary.replace([np.inf, -np.inf], np.nan)
    summary.to_csv(output_dir / "schur_summary_samples.csv", index=False)

    aggregate = (
        summary.groupby("method", as_index=False)["condition_number"]
        .median()
        .sort_values("condition_number")
    )
    fig, ax = plt.subplots(figsize=(max(7.0, len(aggregate) * 0.8), 4.5))
    ax.bar(aggregate["method"], aggregate["condition_number"], color="#55A868")
    ax.set_yscale("log")
    ax.set_ylabel("Median sampled Schur condition number")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "schur_condition_number.png", dpi=300)
    plt.close(fig)

    heatmap = summary.pivot_table(
        index="method",
        columns="quality_label",
        values="condition_number",
        aggfunc="median",
    )
    heatmap = heatmap.reindex(
        index=ordered_methods(summary),
        columns=quality_labels(summary),
    )
    save_heatmap(
        np.log10(heatmap),
        output_dir / "schur_condition_number_by_quality.png",
        "log10 sampled Schur condition number",
    )

    if not spectra:
        return
    spectrum_frames = []
    for frame in spectra:
        spectrum_frames.append(add_quality_columns(frame))

    def plot_spectrum_panels(
        selected_quality: list[str], output: Path, columns: int
    ) -> None:
        rows = int(np.ceil(len(selected_quality) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(7.0 * columns, 4.3 * rows),
            squeeze=False,
        )
        colors = method_colors()
        for index, quality in enumerate(selected_quality):
            ax = axes.flat[index]
            for frame in spectrum_frames:
                if frame["quality_label"].iloc[0] != quality:
                    continue
                method = frame["method"].iloc[0]
                positive = frame[
                    pd.to_numeric(
                        frame["normalized_singular_value"], errors="coerce"
                    )
                    > 0.0
                ]
                ax.plot(
                    positive["index"],
                    positive["normalized_singular_value"],
                    linewidth=1.1,
                    color=colors.get(method),
                    label=method,
                )
            ax.set_yscale("log")
            ax.set_title(quality)
            ax.set_xlabel("Singular-value index")
            ax.set_ylabel("Normalized singular value")
            ax.grid(alpha=0.25)
        for index in range(len(selected_quality), rows * columns):
            axes.flat[index].axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                fontsize=7,
                ncol=1,
                loc="center left",
                bbox_to_anchor=(0.99, 0.5),
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
        else:
            fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    all_quality = quality_labels(summary)
    representative_quality = list(
        dict.fromkeys(
            [
                all_quality[0],
                all_quality[len(all_quality) // 2],
                all_quality[-1],
            ]
        )
    )
    plot_spectrum_panels(
        representative_quality,
        output_dir / "schur_spectrum.png",
        columns=len(representative_quality),
    )
    plot_spectrum_panels(
        all_quality,
        output_dir / "schur_spectrum_all_quality.png",
        columns=2,
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = canonicalize_method_column(pd.read_csv(args.summary))
    required = {"quality_dataset", "method", "status", "solver_time_sec"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")

    generate_aggregate_outputs(
        frame,
        args.out,
        args.solution_rmse_relative_tolerance,
    )
    if args.diagnostic_root:
        generate_convergence_figure(
            args.diagnostic_root, args.out, max(1, args.max_curves)
        )
        generate_point_conditioning_outputs(args.diagnostic_root, args.out)
        generate_schur_outputs(args.diagnostic_root, args.out)


if __name__ == "__main__":
    main()
