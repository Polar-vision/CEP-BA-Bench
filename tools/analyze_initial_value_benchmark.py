#!/usr/bin/env python3
"""Analyze CEP Initial Value benchmark summary outputs."""

from __future__ import annotations

import argparse
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


def markdown_table(frame: pd.DataFrame, columns: list[str], fmt: dict[str, str] | None = None) -> str:
    table = frame[columns].copy()
    if fmt:
        for column, pattern in fmt.items():
            if column in table.columns:
                table[column] = table[column].map(
                    lambda value: pattern.format(value) if pd.notna(value) else ""
                )
    headers = [str(column) for column in table.columns]
    rows = []
    for _, row in table.iterrows():
        rows.append([str(row[column]) if pd.notna(row[column]) else "" for column in table.columns])

    def escape(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "| " + " | ".join(escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def enrich_summary(summary: Path) -> pd.DataFrame:
    data = pd.read_csv(summary)
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["category"] = data["base_dataset"].str.split("__").str[0]
    data["dataset_name"] = data["base_dataset"].str.split("__").str[1]
    data["method"] = pd.Categorical(data["method"], METHOD_ORDER, ordered=True)
    data["method_str"] = data["method"].astype(str)

    best = (
        data.groupby("base_dataset", observed=True)["final_rmse_px"]
        .min()
        .rename("best_final_rmse_px")
    )
    data = data.join(best, on="base_dataset")
    data["rmse_ratio_to_best"] = data["final_rmse_px"] / data["best_final_rmse_px"]
    data["rmse_gap_pct"] = (data["rmse_ratio_to_best"] - 1.0) * 100.0

    data["within_0p1pct"] = data["rmse_ratio_to_best"] <= 1.001
    data["within_1pct"] = data["rmse_ratio_to_best"] <= 1.01
    data["within_5pct"] = data["rmse_ratio_to_best"] <= 1.05
    data["within_10pct"] = data["rmse_ratio_to_best"] <= 1.10
    data["max_iter"] = data["iterations"] >= 100

    data["rmse_rank"] = data.groupby("base_dataset", observed=True)["final_rmse_px"].rank(
        method="min", ascending=True
    )
    data["solver_time_rank"] = data.groupby("base_dataset", observed=True)[
        "solver_time_sec"
    ].rank(method="min", ascending=True)
    data["iteration_rank"] = data.groupby("base_dataset", observed=True)["iterations"].rank(
        method="min", ascending=True
    )
    data["solver_time_good"] = data["solver_time_sec"].where(data["within_1pct"])
    data["good_time_rank"] = data.groupby("base_dataset", observed=True)[
        "solver_time_good"
    ].rank(method="min", ascending=True)

    frame_variable = data["method_str"].map(method_families)
    data["frame_family"] = frame_variable.map(lambda value: value[0])
    data["variable_family"] = frame_variable.map(lambda value: value[1])
    return data


def build_outputs(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    best_rows = (
        data.sort_values(["base_dataset", "final_rmse_px", "solver_time_sec"])
        .groupby("base_dataset", observed=True)
        .head(1)
    )
    fastest_good_rows = (
        data[data["within_1pct"]]
        .sort_values(["base_dataset", "solver_time_sec"])
        .groupby("base_dataset", observed=True)
        .head(1)
    )

    method = (
        data.groupby("method_str", observed=True)
        .agg(
            n=("base_dataset", "count"),
            status_ok_rate=("status", lambda values: (values.astype(str) == "ok").mean()),
            ceres_convergence_rate=("termination_type", lambda values: (values == 0).mean()),
            max_iter_rate=("max_iter", "mean"),
            within_0p1pct_rate=("within_0p1pct", "mean"),
            within_1pct_rate=("within_1pct", "mean"),
            within_5pct_rate=("within_5pct", "mean"),
            within_10pct_rate=("within_10pct", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            q75_rmse_gap_pct=("rmse_gap_pct", lambda values: values.quantile(0.75)),
            q90_rmse_gap_pct=("rmse_gap_pct", lambda values: values.quantile(0.90)),
            mean_rmse_rank=("rmse_rank", "mean"),
            median_iterations=("iterations", "median"),
            mean_iterations=("iterations", "mean"),
            median_solver_time_sec=("solver_time_sec", "median"),
            total_solver_time_sec=("solver_time_sec", "sum"),
            median_wall_time_sec=("wall_time_sec", "median"),
            total_wall_time_sec=("wall_time_sec", "sum"),
            median_linear_solver_time_sec=("linear_solver_time_sec", "median"),
            mean_time_rank=("solver_time_rank", "mean"),
            mean_good_time_rank=("good_time_rank", "mean"),
        )
        .reset_index()
        .rename(columns={"method_str": "method"})
    )
    linear_fraction = (
        data.assign(linear_fraction=data["linear_solver_time_sec"] / data["solver_time_sec"])
        .groupby("method_str", observed=True)["linear_fraction"]
        .median()
        .rename("median_linear_solver_fraction")
    )
    method = method.merge(linear_fraction, left_on="method", right_index=True, how="left")

    best_counts = (
        best_rows["method_str"]
        .value_counts()
        .rename_axis("method")
        .reset_index(name="best_rmse_count")
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
        ["within_1pct_rate", "median_rmse_gap_pct", "median_solver_time_sec"],
        ascending=[False, True, True],
    )

    category_method = (
        data.groupby(["category", "method_str"], observed=True)
        .agg(
            n=("base_dataset", "count"),
            within_1pct_rate=("within_1pct", "mean"),
            within_5pct_rate=("within_5pct", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            q90_rmse_gap_pct=("rmse_gap_pct", lambda values: values.quantile(0.90)),
            max_iter_rate=("max_iter", "mean"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_iterations=("iterations", "median"),
            mean_rmse_rank=("rmse_rank", "mean"),
            mean_good_time_rank=("good_time_rank", "mean"),
        )
        .reset_index()
        .rename(columns={"method_str": "method"})
    )
    category_method["method"] = pd.Categorical(
        category_method["method"], METHOD_ORDER, ordered=True
    )
    category_method = category_method.sort_values(
        ["category", "within_1pct_rate", "median_rmse_gap_pct", "median_solver_time_sec"],
        ascending=[True, False, True, True],
    )

    frame_family = (
        data.groupby("frame_family")
        .agg(
            n=("base_dataset", "count"),
            within_1pct_rate=("within_1pct", "mean"),
            within_5pct_rate=("within_5pct", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_iterations=("iterations", "median"),
            max_iter_rate=("max_iter", "mean"),
        )
        .reset_index()
        .sort_values("within_1pct_rate", ascending=False)
    )
    variable_family = (
        data.groupby("variable_family")
        .agg(
            n=("base_dataset", "count"),
            within_1pct_rate=("within_1pct", "mean"),
            within_5pct_rate=("within_5pct", "mean"),
            median_rmse_gap_pct=("rmse_gap_pct", "median"),
            median_solver_time_sec=("solver_time_sec", "median"),
            median_iterations=("iterations", "median"),
            max_iter_rate=("max_iter", "mean"),
        )
        .reset_index()
        .sort_values("within_1pct_rate", ascending=False)
    )

    pairs = [
        ("A2-Parallax-Mc", "A2-Parallax-Mw"),
        ("A1-XYZ-Ac", "A1-XYZ-Aw"),
        ("A1-XYInvZ-Ac", "A1-XYInvZ-Aw"),
        ("A1-SphRange-Ac", "A1-SphRange-Aw"),
        ("A1-SphInvRange-Ac", "A1-SphInvRange-Aw"),
    ]
    wide = data.pivot(
        index="base_dataset",
        columns="method_str",
        values=["final_rmse_px", "solver_time_sec", "within_1pct"],
    )
    pair_rows = []
    for left, right in pairs:
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
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    dataset_base = (
        data.groupby(["category", "base_dataset"])
        .agg(
            cameras=("cameras", "first"),
            points=("points", "first"),
            observations=("observations", "first"),
            initial_rmse_px=("initial_rmse_px", "median"),
        )
        .reset_index()
    )
    best_dataset = best_rows[
        ["base_dataset", "method_str", "final_rmse_px", "solver_time_sec", "iterations"]
    ].rename(
        columns={
            "method_str": "best_rmse_method",
            "final_rmse_px": "best_rmse_px",
            "solver_time_sec": "best_method_solver_time_sec",
            "iterations": "best_method_iterations",
        }
    )
    fastest_dataset = fastest_good_rows[
        [
            "base_dataset",
            "method_str",
            "final_rmse_px",
            "rmse_gap_pct",
            "solver_time_sec",
            "iterations",
        ]
    ].rename(
        columns={
            "method_str": "fastest_within_1pct_method",
            "final_rmse_px": "fastest_within_1pct_rmse_px",
            "rmse_gap_pct": "fastest_within_1pct_rmse_gap_pct",
            "solver_time_sec": "fastest_within_1pct_solver_time_sec",
            "iterations": "fastest_within_1pct_iterations",
        }
    )
    dataset_winners = dataset_base.merge(best_dataset, on="base_dataset", how="left").merge(
        fastest_dataset, on="base_dataset", how="left"
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
    winner_counts = winner_counts.sort_values(
        ["best_rmse_datasets", "fastest_good_datasets"], ascending=False
    )

    return {
        "method_summary": method,
        "category_method_summary": category_method,
        "frame_family_summary": frame_family,
        "variable_family_summary": variable_family,
        "pairwise_frame_comparisons": pairwise,
        "dataset_winners": dataset_winners,
        "winner_counts": winner_counts,
    }


def write_report(data: pd.DataFrame, outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    method = outputs["method_summary"]
    report = [
        "# Initial Value benchmark analysis\n",
        f"- Runs: {len(data)} ({data.base_dataset.nunique()} datasets x {data.method_str.nunique()} methods)",
        f"- Status: {data.status.eq('ok').sum()} ok, {(~data.status.eq('ok')).sum()} failed",
        f"- Ceres convergence rate: {(data.termination_type.eq(0).mean() * 100):.1f}%",
        f"- Max-iteration rate: {(data.max_iter.mean() * 100):.1f}%",
        f"- Total wall time in CSV: {data.wall_time_sec.sum() / 3600:.2f} h; total solver time: {data.solver_time_sec.sum() / 3600:.2f} h",
        "\n## Method ranking by robust solution quality\n",
        markdown_table(
            method,
            [
                "method",
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
            },
        ),
        "\n## Frame family summary\n",
        markdown_table(
            outputs["frame_family_summary"],
            outputs["frame_family_summary"].columns.tolist(),
            {
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
            outputs["variable_family_summary"].columns.tolist(),
            {
                "within_1pct_rate": "{:.3f}",
                "within_5pct_rate": "{:.3f}",
                "median_rmse_gap_pct": "{:.2f}",
                "median_solver_time_sec": "{:.2f}",
                "median_iterations": "{:.1f}",
                "max_iter_rate": "{:.3f}",
            },
        ),
    ]
    report.append("\n## Category highlights: top 5 per category by 1% solution rate\n")
    category = outputs["category_method_summary"]
    for name in ["Close-Range", "Oblique-5", "UAV", "Vehicle"]:
        report.append(f"\n### {name}\n")
        report.append(
            markdown_table(
                category[category.category == name].head(5),
                [
                    "method",
                    "within_1pct_rate",
                    "median_rmse_gap_pct",
                    "median_solver_time_sec",
                    "median_iterations",
                    "max_iter_rate",
                ],
                {
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
            outputs["winner_counts"].columns.tolist(),
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
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = enrich_summary(args.summary)
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
