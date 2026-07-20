#!/usr/bin/env python3
"""Analyze CEP public-release <100-image clean benchmark outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

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

NUMERIC_COLUMNS = [
    "wall_time_sec",
    "solver_time_sec",
    "linear_solver_time_sec",
    "cameras",
    "points",
    "observations",
    "initial_cost",
    "final_cost",
    "initial_rmse_px",
    "final_rmse_px",
    "iterations",
    "accepted_steps",
    "rejected_steps",
    "linear_solver_iterations",
    "final_gradient_max_norm",
    "final_gradient_norm",
    "termination_type",
]

QUALITY_RE = re.compile(r"-(?P<mode>init|joint)-rmse(?P<int>\d+)p(?P<frac>\d+)px$")
BASE_RE = re.compile(
    r"^BA-problem-(?P<id>\d+)-i(?P<images>\d+)-p(?P<points>\d+)"
    r"-o(?P<observations>\d+)-g(?P<gcps>\d+)-c(?P<checkpoints>\d+)$"
)


def method_families(method: str) -> tuple[str, str]:
    if method.startswith("A0-"):
        frame = "A0 global"
    elif method.endswith("-Ac") or method.endswith("-Mc"):
        frame = "camera-frame anchored"
    elif method.endswith("-Aw") or method.endswith("-Mw"):
        frame = "world-parallel anchored"
    else:
        frame = "other"

    if "Parallax" in method:
        variable = "Parallax"
    elif "XYZ" in method and "XYInvZ" not in method:
        variable = "XYZ"
    elif "XYInvZ" in method:
        variable = "XYInvZ"
    elif "SphRange" in method and "SphInvRange" not in method:
        variable = "SphRange"
    elif "SphInvRange" in method:
        variable = "SphInvRange"
    else:
        variable = "other"
    return frame, variable


def parse_target_rmse(name: str) -> tuple[str, float]:
    match = QUALITY_RE.search(name)
    if not match:
        return "unknown", np.nan
    mode = match.group("mode")
    value = float(f"{int(match.group('int'))}.{match.group('frac')}")
    return mode, value


def parse_base_name(name: str) -> dict[str, float | int | str]:
    match = BASE_RE.match(name)
    if not match:
        return {
            "base_id": name,
            "base_images": np.nan,
            "base_points": np.nan,
            "base_observations": np.nan,
            "gcps": np.nan,
            "checkpoints": np.nan,
        }
    groups = match.groupdict()
    return {
        "base_id": groups["id"],
        "base_images": int(groups["images"]),
        "base_points": int(groups["points"]),
        "base_observations": int(groups["observations"]),
        "gcps": int(groups["gcps"]),
        "checkpoints": int(groups["checkpoints"]),
    }


def image_bucket(images: float) -> str:
    if pd.isna(images):
        return "unknown"
    images = int(images)
    if images < 20:
        return "010-019"
    if images < 50:
        return "020-049"
    if images < 75:
        return "050-074"
    return "075-099"


def markdown_table(frame: pd.DataFrame, columns: list[str], fmt: dict[str, str] | None = None) -> str:
    table = frame[columns].copy()
    if fmt:
        for column, pattern in fmt.items():
            if column in table.columns:
                table[column] = table[column].map(
                    lambda value: pattern.format(value) if pd.notna(value) else ""
                )

    def escape(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "| " + " | ".join(escape(str(column)) for column in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                escape(str(row[column])) if pd.notna(row[column]) else ""
                for column in table.columns
            )
            + " |"
        )
    return "\n".join(lines)


def load_manifest(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    manifest = pd.read_csv(path)
    manifest = manifest.rename(
        columns={
            "base": "base_dataset",
            "quality": "quality_dataset",
            "images": "manifest_images",
            "points": "manifest_points",
            "observations": "manifest_observations",
        }
    )
    for column in ["manifest_images", "manifest_points", "manifest_observations"]:
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce")
    return manifest


def enrich_summary(summary: Path, manifest_path: Path | None) -> pd.DataFrame:
    data = pd.read_csv(summary)
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["method_str"] = data["method"].astype(str)
    data["method"] = pd.Categorical(data["method_str"], METHOD_ORDER, ordered=True)
    data["dataset_key"] = data["base_dataset"].astype(str) + " / " + data["quality_dataset"].astype(str)
    data["eligible"] = (
        data["status"].astype(str).eq("ok")
        & data["final_rmse_px"].notna()
        & np.isfinite(data["final_rmse_px"])
        & (data["final_rmse_px"] > 0)
    )

    parsed_quality = data["quality_dataset"].map(parse_target_rmse)
    data["quality_mode"] = parsed_quality.map(lambda value: value[0])
    data["target_rmse_px"] = parsed_quality.map(lambda value: value[1])
    data["quality_level"] = data.apply(
        lambda row: (
            f"{row.quality_mode}-{row.target_rmse_px:g}px"
            if row.quality_mode != "unknown" and pd.notna(row.target_rmse_px)
            else "unknown"
        ),
        axis=1,
    )

    base_meta = pd.DataFrame(data["base_dataset"].drop_duplicates().map(parse_base_name).tolist())
    base_meta["base_dataset"] = data["base_dataset"].drop_duplicates().to_numpy()
    data = data.merge(base_meta, on="base_dataset", how="left")

    manifest = load_manifest(manifest_path)
    if not manifest.empty:
        data = data.merge(
            manifest[
                [
                    "base_dataset",
                    "quality_dataset",
                    "manifest_images",
                    "manifest_points",
                    "manifest_observations",
                ]
            ],
            on=["base_dataset", "quality_dataset"],
            how="left",
        )
    else:
        data["manifest_images"] = np.nan
        data["manifest_points"] = np.nan
        data["manifest_observations"] = np.nan

    data["images"] = data["manifest_images"].combine_first(data["cameras"]).combine_first(
        data["base_images"]
    )
    data["image_bucket"] = data["images"].map(image_bucket)

    best = (
        data[data["eligible"]]
        .groupby("dataset_key", observed=True)["final_rmse_px"]
        .min()
        .rename("best_final_rmse_px")
    )
    data = data.join(best, on="dataset_key")
    data["rmse_ratio_to_best"] = data["final_rmse_px"] / data["best_final_rmse_px"]
    data["rmse_gap_pct"] = (data["rmse_ratio_to_best"] - 1.0) * 100.0
    data["within_0p1pct"] = data["eligible"] & (data["rmse_ratio_to_best"] <= 1.001)
    data["within_1pct"] = data["eligible"] & (data["rmse_ratio_to_best"] <= 1.01)
    data["within_5pct"] = data["eligible"] & (data["rmse_ratio_to_best"] <= 1.05)
    data["within_10pct"] = data["eligible"] & (data["rmse_ratio_to_best"] <= 1.10)
    data["max_iter"] = data["iterations"] >= 100
    data["ceres_converged"] = data["termination_type"] == 0
    data["solution_success"] = data["ceres_converged"] & data["within_1pct"]

    data["rmse_rank"] = data.groupby("dataset_key", observed=True)["final_rmse_px"].rank(
        method="min", ascending=True
    )
    data["solver_time_rank"] = data.groupby("dataset_key", observed=True)[
        "solver_time_sec"
    ].rank(method="min", ascending=True)
    data["iteration_rank"] = data.groupby("dataset_key", observed=True)["iterations"].rank(
        method="min", ascending=True
    )
    data["solver_time_good"] = data["solver_time_sec"].where(data["within_1pct"])
    data["good_time_rank"] = data.groupby("dataset_key", observed=True)[
        "solver_time_good"
    ].rank(method="min", ascending=True)

    frame_variable = data["method_str"].map(method_families)
    data["frame_family"] = frame_variable.map(lambda value: value[0])
    data["variable_family"] = frame_variable.map(lambda value: value[1])
    return data


def aggregate_by(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        data.groupby(group_columns, observed=True)
        .agg(
            n=("dataset_key", "count"),
            datasets=("dataset_key", "nunique"),
            status_ok_rate=("status", lambda values: values.astype(str).eq("ok").mean()),
            ceres_convergence_rate=("ceres_converged", "mean"),
            solution_success_rate=("solution_success", "mean"),
            within_1pct_rate=("within_1pct", "mean"),
            within_5pct_rate=("within_5pct", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            q75_rmse_gap_pct=("rmse_gap_pct", lambda values: values.quantile(0.75)),
            q90_rmse_gap_pct=("rmse_gap_pct", lambda values: values.quantile(0.90)),
            mean_rmse_rank=("rmse_rank", "mean"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_linear_solver_time_sec=("linear_solver_time_sec", "median"),
            median_iterations=("iterations", "median"),
            max_iter_rate=("max_iter", "mean"),
            mean_good_time_rank=("good_time_rank", "mean"),
        )
        .reset_index()
    )


def build_outputs(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    best_rows = (
        data[data["eligible"]]
        .sort_values(["dataset_key", "final_rmse_px", "solver_time_sec"])
        .groupby("dataset_key", observed=True)
        .head(1)
    )
    fastest_good_rows = (
        data[data["within_1pct"]]
        .sort_values(["dataset_key", "solver_time_sec", "final_rmse_px"])
        .groupby("dataset_key", observed=True)
        .head(1)
    )

    method = aggregate_by(data, ["method_str"]).rename(columns={"method_str": "method"})
    linear_fraction = (
        data.assign(linear_fraction=data["linear_solver_time_sec"] / data["solver_time_sec"])
        .groupby("method_str", observed=True)["linear_fraction"]
        .median()
        .rename("median_linear_solver_fraction")
    )
    method = method.merge(linear_fraction, left_on="method", right_index=True, how="left")

    best_counts = (
        best_rows["method_str"].value_counts().rename_axis("method").reset_index(name="best_rmse_count")
    )
    fastest_counts = (
        fastest_good_rows["method_str"]
        .value_counts()
        .rename_axis("method")
        .reset_index(name="fastest_within_1pct_count")
    )
    method = method.merge(best_counts, on="method", how="left").merge(
        fastest_counts, on="method", how="left"
    )
    method[["best_rmse_count", "fastest_within_1pct_count"]] = method[
        ["best_rmse_count", "fastest_within_1pct_count"]
    ].fillna(0).astype(int)
    method["method"] = pd.Categorical(method["method"], METHOD_ORDER, ordered=True)
    method = method.sort_values(
        ["solution_success_rate", "within_1pct_rate", "median_rmse_gap_pct", "median_solver_time_sec"],
        ascending=[False, False, True, True],
    )

    frame_family = aggregate_by(data, ["frame_family"]).sort_values(
        ["solution_success_rate", "median_rmse_gap_pct"], ascending=[False, True]
    )
    variable_family = aggregate_by(data, ["variable_family"]).sort_values(
        ["solution_success_rate", "median_rmse_gap_pct"], ascending=[False, True]
    )

    bucket_method = aggregate_by(data, ["image_bucket", "method_str"]).rename(
        columns={"method_str": "method"}
    )
    bucket_method["method"] = pd.Categorical(bucket_method["method"], METHOD_ORDER, ordered=True)
    bucket_method = bucket_method.sort_values(
        ["image_bucket", "solution_success_rate", "median_rmse_gap_pct", "median_solver_time_sec"],
        ascending=[True, False, True, True],
    )

    quality_method = aggregate_by(data, ["quality_level", "quality_mode", "target_rmse_px", "method_str"]).rename(
        columns={"method_str": "method"}
    )
    quality_method["method"] = pd.Categorical(quality_method["method"], METHOD_ORDER, ordered=True)
    quality_method = quality_method.sort_values(
        [
            "quality_mode",
            "target_rmse_px",
            "solution_success_rate",
            "median_rmse_gap_pct",
            "median_solver_time_sec",
        ],
        ascending=[True, True, False, True, True],
    )

    pairs = [
        ("A2-Parallax-Mc", "A2-Parallax-Mw"),
        ("A1-XYZ-Ac", "A1-XYZ-Aw"),
        ("A1-XYInvZ-Ac", "A1-XYInvZ-Aw"),
        ("A1-SphRange-Ac", "A1-SphRange-Aw"),
        ("A1-SphInvRange-Ac", "A1-SphInvRange-Aw"),
    ]
    wide = data.pivot(
        index="dataset_key",
        columns="method_str",
        values=["final_rmse_px", "solver_time_sec", "within_1pct", "solution_success"],
    )
    pair_rows = []
    for left, right in pairs:
        if ("final_rmse_px", left) not in wide or ("final_rmse_px", right) not in wide:
            continue
        rmse_left = wide[("final_rmse_px", left)]
        rmse_right = wide[("final_rmse_px", right)]
        time_left = wide[("solver_time_sec", left)]
        time_right = wide[("solver_time_sec", right)]
        pair_rows.append(
            {
                "method_a": left,
                "method_b": right,
                "a_better_rmse_count": int((rmse_left < rmse_right).sum()),
                "b_better_rmse_count": int((rmse_right < rmse_left).sum()),
                "a_faster_count": int((time_left < time_right).sum()),
                "b_faster_count": int((time_right < time_left).sum()),
                "median_solver_time_ratio_a_over_b": float((time_left / time_right).median()),
                "median_rmse_ratio_a_over_b": float((rmse_left / rmse_right).median()),
                "a_within_1pct_rate": float(wide[("within_1pct", left)].mean()),
                "b_within_1pct_rate": float(wide[("within_1pct", right)].mean()),
                "a_solution_success_rate": float(wide[("solution_success", left)].mean()),
                "b_solution_success_rate": float(wide[("solution_success", right)].mean()),
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    dataset_base = (
        data.groupby("dataset_key", observed=True)
        .agg(
            base_dataset=("base_dataset", "first"),
            quality_dataset=("quality_dataset", "first"),
            images=("images", "first"),
            points=("points", "first"),
            observations=("observations", "first"),
            image_bucket=("image_bucket", "first"),
            quality_mode=("quality_mode", "first"),
            target_rmse_px=("target_rmse_px", "first"),
            initial_rmse_px=("initial_rmse_px", "median"),
            methods_run=("method_str", "nunique"),
            ok_runs=("status", lambda values: values.astype(str).eq("ok").sum()),
            within_1pct_methods=("within_1pct", "sum"),
            solution_success_methods=("solution_success", "sum"),
        )
        .reset_index()
    )
    best_dataset = best_rows[
        ["dataset_key", "method_str", "final_rmse_px", "solver_time_sec", "iterations"]
    ].rename(
        columns={
            "method_str": "best_rmse_method",
            "final_rmse_px": "best_rmse_px",
            "solver_time_sec": "best_method_solver_time_sec",
            "iterations": "best_method_iterations",
        }
    )
    fastest_dataset = fastest_good_rows[
        ["dataset_key", "method_str", "final_rmse_px", "rmse_gap_pct", "solver_time_sec", "iterations"]
    ].rename(
        columns={
            "method_str": "fastest_within_1pct_method",
            "final_rmse_px": "fastest_within_1pct_rmse_px",
            "rmse_gap_pct": "fastest_within_1pct_rmse_gap_pct",
            "solver_time_sec": "fastest_within_1pct_solver_time_sec",
            "iterations": "fastest_within_1pct_iterations",
        }
    )
    dataset_winners = dataset_base.merge(best_dataset, on="dataset_key", how="left").merge(
        fastest_dataset, on="dataset_key", how="left"
    )

    winner_counts = (
        dataset_winners["best_rmse_method"]
        .value_counts()
        .rename_axis("method")
        .reset_index(name="best_rmse_datasets")
        .merge(
            dataset_winners["fastest_within_1pct_method"]
            .value_counts()
            .rename_axis("method")
            .reset_index(name="fastest_good_datasets"),
            on="method",
            how="outer",
        )
        .fillna(0)
    )
    winner_counts[["best_rmse_datasets", "fastest_good_datasets"]] = winner_counts[
        ["best_rmse_datasets", "fastest_good_datasets"]
    ].astype(int)
    winner_counts["method"] = pd.Categorical(winner_counts["method"], METHOD_ORDER, ordered=True)
    winner_counts = winner_counts.sort_values(
        ["best_rmse_datasets", "fastest_good_datasets"], ascending=False
    )

    hard_cases = dataset_winners.sort_values(
        ["solution_success_methods", "within_1pct_methods", "initial_rmse_px", "images"],
        ascending=[True, True, False, True],
    )

    parallax = data[data["variable_family"] == "Parallax"].copy()
    parallax_exceptions = parallax[
        (~parallax["within_1pct"]) | (~parallax["ceres_converged"])
    ].sort_values(["dataset_key", "method_str"])

    return {
        "method_summary": method,
        "frame_family_summary": frame_family,
        "variable_family_summary": variable_family,
        "bucket_method_summary": bucket_method,
        "quality_method_summary": quality_method,
        "pairwise_frame_comparisons": pairwise,
        "dataset_winners": dataset_winners,
        "winner_counts": winner_counts,
        "hard_cases": hard_cases,
        "parallax_exceptions": parallax_exceptions,
    }


def write_report(data: pd.DataFrame, outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    method = outputs["method_summary"]
    frame = outputs["frame_family_summary"]
    variable = outputs["variable_family_summary"]
    pairwise = outputs["pairwise_frame_comparisons"]
    bucket = outputs["bucket_method_summary"]
    quality = outputs["quality_method_summary"]
    hard = outputs["hard_cases"]
    dataset_count = data["dataset_key"].nunique()
    method_count = data["method_str"].nunique()
    complete_groups = (
        data.groupby("dataset_key", observed=True)["method_str"].nunique().eq(method_count).sum()
    )
    leader = method.iloc[0]
    fastest = method.sort_values(
        ["fastest_within_1pct_count", "solution_success_rate", "median_solver_time_sec"],
        ascending=[False, False, True],
    ).iloc[0]
    variable_leader = variable.iloc[0]
    frame_leader = frame.iloc[0]
    parallax_pair = pairwise[
        (pairwise["method_a"] == "A2-Parallax-Mc")
        & (pairwise["method_b"] == "A2-Parallax-Mw")
    ]
    if not parallax_pair.empty:
        parallax_pair_row = parallax_pair.iloc[0]
        parallax_guidance = (
            f"`A2-Parallax-Mc` has solution-success "
            f"{parallax_pair_row.a_solution_success_rate:.1%} versus "
            f"{parallax_pair_row.b_solution_success_rate:.1%} for `A2-Parallax-Mw`, "
            f"while Mc/Mw median solver-time ratio is "
            f"{parallax_pair_row.median_solver_time_ratio_a_over_b:.2f}x."
        )
    else:
        parallax_guidance = "Parallax Mc/Mw pair was not fully available."

    hard_quality = (
        quality.groupby(["quality_level", "quality_mode", "target_rmse_px"], observed=True)
        .agg(
            median_solution_success_rate=("solution_success_rate", "median"),
            best_solution_success_rate=("solution_success_rate", "max"),
        )
        .reset_index()
        .sort_values(["best_solution_success_rate", "median_solution_success_rate"])
        .head(3)
    )
    hard_quality_text = "; ".join(
        f"{row.quality_level} best={row.best_solution_success_rate:.1%}, "
        f"median={row.median_solution_success_rate:.1%}"
        for _, row in hard_quality.iterrows()
    )
    bucket_winners = (
        bucket.sort_values(
            ["image_bucket", "solution_success_rate", "median_rmse_gap_pct", "median_solver_time_sec"],
            ascending=[True, False, True, True],
        )
        .groupby("image_bucket", observed=True)
        .head(1)
    )
    bucket_text = "; ".join(
        f"{row.image_bucket}: `{row.method}` ({row.solution_success_rate:.1%})"
        for _, row in bucket_winners.iterrows()
    )
    hard_case_count = int((hard["solution_success_methods"] <= 2).sum())
    hard_case_rate = hard_case_count / max(len(hard), 1)

    report = [
        "# Public-release <100-image clean benchmark analysis\n",
        f"- Runs: {len(data)} ({dataset_count} quality datasets x {method_count} methods)",
        f"- Complete matched quality sets: {complete_groups}/{dataset_count}",
        f"- Status: {data.status.astype(str).eq('ok').sum()} ok, {(~data.status.astype(str).eq('ok')).sum()} failed",
        f"- Ceres convergence rate: {(data.ceres_converged.mean() * 100):.1f}%",
        f"- Solution-success rate (Ceres convergence and <=1% from best RMSE): {(data.solution_success.mean() * 100):.1f}%",
        f"- Max-iteration rate: {(data.max_iter.mean() * 100):.1f}%",
        f"- Total wall time in CSV: {data.wall_time_sec.sum() / 3600:.2f} h; total solver time: {data.solver_time_sec.sum() / 3600:.2f} h",
        f"- Image range: {int(data.images.min())}-{int(data.images.max())}; median observations per run: {data.observations.median():.0f}",
        "\n## Guidance conclusions\n",
        f"- Default robust choice: `{leader.method}` has the highest combined solution-success ranking "
        f"({leader.solution_success_rate:.1%} solution-success, "
        f"{leader.within_1pct_rate:.1%} within 1%, median RMSE gap "
        f"{leader.median_rmse_gap_pct:.3g}%).",
        f"- Fast practical choice: `{fastest.method}` is the fastest acceptable winner on "
        f"{int(fastest.fastest_within_1pct_count)} matched quality sets, with median solver time "
        f"{fastest.median_solver_time_sec:.2f}s.",
        f"- Variable-family signal: `{variable_leader.variable_family}` is the strongest family "
        f"({variable_leader.solution_success_rate:.1%} solution-success, median RMSE gap "
        f"{variable_leader.median_rmse_gap_pct:.3g}%).",
        f"- Frame-family signal: `{frame_leader.frame_family}` is strongest overall "
        f"({frame_leader.solution_success_rate:.1%} solution-success).",
        f"- Parallax frame trade-off: {parallax_guidance}",
        f"- Image-count guidance by bucket: {bucket_text}.",
        f"- Most difficult perturbation levels by aggregate success: {hard_quality_text}.",
        f"- Hard-case concentration: {hard_case_count}/{len(hard)} quality sets "
        f"({hard_case_rate:.1%}) have two or fewer solution-success methods.",
        "\n## Method ranking by robust solution quality\n",
        markdown_table(
            method,
            [
                "method",
                "solution_success_rate",
                "within_1pct_rate",
                "within_5pct_rate",
                "median_rmse_gap_pct",
                "q90_rmse_gap_pct",
                "best_rmse_count",
                "fastest_within_1pct_count",
                "median_solver_time_sec",
                "median_iterations",
                "max_iter_rate",
            ],
            {
                "solution_success_rate": "{:.3f}",
                "within_1pct_rate": "{:.3f}",
                "within_5pct_rate": "{:.3f}",
                "median_rmse_gap_pct": "{:.2f}",
                "q90_rmse_gap_pct": "{:.2f}",
                "median_solver_time_sec": "{:.2f}",
                "median_iterations": "{:.1f}",
                "max_iter_rate": "{:.3f}",
            },
        ),
        "\n## Pairwise frame comparisons\n",
        markdown_table(
            outputs["pairwise_frame_comparisons"],
            outputs["pairwise_frame_comparisons"].columns.tolist(),
            {
                "median_solver_time_ratio_a_over_b": "{:.3f}",
                "median_rmse_ratio_a_over_b": "{:.6f}",
                "a_within_1pct_rate": "{:.3f}",
                "b_within_1pct_rate": "{:.3f}",
                "a_solution_success_rate": "{:.3f}",
                "b_solution_success_rate": "{:.3f}",
            },
        ),
        "\n## Frame family summary\n",
        markdown_table(
            outputs["frame_family_summary"],
            [
                "frame_family",
                "n",
                "solution_success_rate",
                "within_1pct_rate",
                "within_5pct_rate",
                "median_rmse_gap_pct",
                "median_solver_time_sec",
                "median_iterations",
                "max_iter_rate",
            ],
            {
                "solution_success_rate": "{:.3f}",
                "within_1pct_rate": "{:.3f}",
                "within_5pct_rate": "{:.3f}",
                "median_rmse_gap_pct": "{:.2f}",
                "median_solver_time_sec": "{:.2f}",
                "median_iterations": "{:.1f}",
                "max_iter_rate": "{:.3f}",
            },
        ),
        "\n## Variable family summary\n",
        markdown_table(
            outputs["variable_family_summary"],
            [
                "variable_family",
                "n",
                "solution_success_rate",
                "within_1pct_rate",
                "within_5pct_rate",
                "median_rmse_gap_pct",
                "median_solver_time_sec",
                "median_iterations",
                "max_iter_rate",
            ],
            {
                "solution_success_rate": "{:.3f}",
                "within_1pct_rate": "{:.3f}",
                "within_5pct_rate": "{:.3f}",
                "median_rmse_gap_pct": "{:.2f}",
                "median_solver_time_sec": "{:.2f}",
                "median_iterations": "{:.1f}",
                "max_iter_rate": "{:.3f}",
            },
        ),
        "\n## Image-bucket highlights: top 5 per bucket\n",
    ]
    bucket = outputs["bucket_method_summary"]
    for name in ["010-019", "020-049", "050-074", "075-099"]:
        subset = bucket[bucket["image_bucket"] == name].head(5)
        if subset.empty:
            continue
        report.append(f"\n### {name} images\n")
        report.append(
            markdown_table(
                subset,
                [
                    "method",
                    "solution_success_rate",
                    "within_1pct_rate",
                    "median_rmse_gap_pct",
                    "median_solver_time_sec",
                    "median_iterations",
                    "max_iter_rate",
                ],
                {
                    "solution_success_rate": "{:.3f}",
                    "within_1pct_rate": "{:.3f}",
                    "median_rmse_gap_pct": "{:.2f}",
                    "median_solver_time_sec": "{:.2f}",
                    "median_iterations": "{:.1f}",
                    "max_iter_rate": "{:.3f}",
                },
            )
        )

    report.append("\n## Quality-level highlights: top 3 per perturbation level\n")
    quality = outputs["quality_method_summary"]
    levels = (
        quality[["quality_level", "quality_mode", "target_rmse_px"]]
        .drop_duplicates()
        .sort_values(["quality_mode", "target_rmse_px"])
    )
    for _, level in levels.iterrows():
        subset = quality[quality["quality_level"] == level["quality_level"]].head(3)
        if subset.empty:
            continue
        report.append(f"\n### {level['quality_level']}\n")
        report.append(
            markdown_table(
                subset,
                [
                    "method",
                    "solution_success_rate",
                    "within_1pct_rate",
                    "median_rmse_gap_pct",
                    "median_solver_time_sec",
                    "median_iterations",
                    "max_iter_rate",
                ],
                {
                    "solution_success_rate": "{:.3f}",
                    "within_1pct_rate": "{:.3f}",
                    "median_rmse_gap_pct": "{:.2f}",
                    "median_solver_time_sec": "{:.2f}",
                    "median_iterations": "{:.1f}",
                    "max_iter_rate": "{:.3f}",
                },
            )
        )

    report.append("\n## Dataset winner counts\n")
    report.append(
        markdown_table(
            outputs["winner_counts"],
            ["method", "best_rmse_datasets", "fastest_good_datasets"],
        )
    )

    report.append("\n## Hardest matched quality sets\n")
    report.append(
        markdown_table(
            outputs["hard_cases"].head(20),
            [
                "base_dataset",
                "quality_dataset",
                "images",
                "initial_rmse_px",
                "within_1pct_methods",
                "solution_success_methods",
                "best_rmse_method",
                "fastest_within_1pct_method",
            ],
            {
                "initial_rmse_px": "{:.2f}",
            },
        )
    )

    (output_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")


def safe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path, index=False)
    except PermissionError as error:
        fallback = path.with_name(path.stem + "_new" + path.suffix)
        frame.to_csv(fallback, index=False)
        print(f"WARNING could not overwrite {path}: {error}")
        print(f"Wrote fallback {fallback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = enrich_summary(args.summary, args.manifest)
    outputs = build_outputs(data)
    safe_to_csv(data, args.out / "summary_enriched.csv")
    for name, frame in outputs.items():
        safe_to_csv(frame, args.out / f"{name}.csv")
    write_report(data, outputs, args.out)

    print(f"ANALYSIS_DIR {args.out}")
    print("\nTOP METHOD SUMMARY")
    print(
        outputs["method_summary"][
            [
                "method",
                "solution_success_rate",
                "within_1pct_rate",
                "within_5pct_rate",
                "median_rmse_gap_pct",
                "q90_rmse_gap_pct",
                "best_rmse_count",
                "fastest_within_1pct_count",
                "median_solver_time_sec",
                "median_iterations",
                "max_iter_rate",
            ]
        ].to_string(index=False)
    )
    print("\nPAIRWISE")
    print(outputs["pairwise_frame_comparisons"].to_string(index=False))
    print("\nFRAME")
    print(outputs["frame_family_summary"].to_string(index=False))
    print("\nVARIABLE")
    print(outputs["variable_family_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
