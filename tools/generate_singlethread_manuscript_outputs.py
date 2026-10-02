#!/usr/bin/env python3
"""Generate single-thread manuscript tables and profile figures."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from analyze_initial_value_benchmark import METHOD_ORDER, build_outputs, enrich_summary


CATEGORIES = ["Close-Range", "Oblique-5", "UAV", "Vehicle"]
TOLERANCES = [0.01, 0.05]

METHOD_COLORS = {
    "A2-Parallax-Mw": "#0072B2",
    "A2-Parallax-Mc": "#CC79A7",
    "A0-SphInvRange-W": "#009E73",
    "A1-XYInvZ-Ac": "#E69F00",
    "A1-SphInvRange-Aw": "#56B4E9",
    "A1-SphInvRange-Ac": "#009E73",
    "A0-XYInvZ-W": "#F0E442",
    "A1-XYZ-Aw": "#8C6BB1",
    "A0-XYZ-W": "#7F7F7F",
    "A1-XYZ-Ac": "#D55E00",
    "A1-SphRange-Ac": "#E15759",
    "A1-XYInvZ-Aw": "#B07AA1",
    "A0-SphRange-W": "#CC6677",
    "A1-SphRange-Aw": "#999999",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--previous-clean-root", type=Path)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmark_initial_value_problems",
    )
    return parser.parse_args()


def pct(value):
    return 100.0 * value


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value != 0.0 and (abs(value) < 0.01 or abs(value) >= 1000.0):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    return f"{pct(value):.1f}"


def tex_escape(value: str) -> str:
    return str(value).replace("_", r"\_")


def maybe_bold(text: str, is_best: bool) -> str:
    return rf"\textbf{{{text}}}" if is_best else text


def mark_best(values: pd.Series, higher: bool) -> pd.Series:
    finite = pd.to_numeric(values, errors="coerce")
    target = finite.max() if higher else finite.min()
    return pd.Series(np.isclose(finite, target, rtol=1e-12, atol=1e-12), index=values.index)


def ensure_dirs(root: Path) -> tuple[Path, Path]:
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def write_execution_summary(clean: pd.DataFrame, diagnostic: pd.DataFrame, path: Path) -> None:
    category_counts = (
        clean[["base_dataset", "category"]]
        .drop_duplicates()
        .groupby("category")
        .size()
        .reindex(CATEGORIES)
    )
    category_text = ", ".join(f"{name}: {int(category_counts[name])}" for name in CATEGORIES)
    clean_ok = int(clean["status"].astype(str).eq("ok").sum())
    diag_ok = int(diagnostic["status"].astype(str).eq("ok").sum())
    conv_rate = pct(clean["termination_type"].eq(0).mean())
    max_iter_rate = pct(clean["iterations"].ge(100).mean())
    total_solver_h = clean["solver_time_sec"].sum() / 3600.0
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Execution summary for the single-thread Initial Value benchmark tier.}",
        r"\label{tab:results_execution_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        rf"Base BA problems & {clean.base_dataset.nunique()} \\",
        rf"Parameterizations & {clean.method_str.nunique()} \\",
        rf"Clean-logged runs & {len(clean)} ({clean_ok} usable) \\",
        rf"Strict diagnostic runs & {len(diagnostic)} ({diag_ok} usable) \\",
        rf"Dataset categories & {category_text} \\",
        rf"Clean Ceres convergence rate & {conv_rate:.1f}\% \\",
        rf"Clean maximum-iteration rate & {max_iter_rate:.1f}\% \\",
        rf"Clean total solver time & {total_solver_h:.2f} h \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    write_text(path, "\n".join(lines))


def write_method_ranking(method: pd.DataFrame, path: Path) -> None:
    frame = method.copy()
    best_cols = {
        "within_1pct_rate": True,
        "within_5pct_rate": True,
        "median_rmse_gap_pct": False,
        "q90_rmse_gap_pct": False,
        "best_rmse_count": True,
        "fastest_within_1pct_count": True,
        "median_solver_time_sec": False,
        "median_iterations": False,
        "max_iter_rate": False,
    }
    best = {col: mark_best(frame[col], higher) for col, higher in best_cols.items()}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Method ranking on matched single-thread Initial Value runs. The RMSE gap is measured relative to the best final reprojection RMSE within each dataset.}",
        r"\label{tab:results_method_ranking}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Method & \shortstack{Within\\1\%} & \shortstack{Within\\5\%} & \shortstack{Med. gap\\(\%)} & \shortstack{Q90 gap\\(\%)} & Best & \shortstack{Fastest\\good} & \shortstack{Med. time\\(s)} & \shortstack{Med.\\it.} & \shortstack{Max-it.\\(\%)} \\",
        r"\midrule",
    ]
    for i, row in frame.iterrows():
        values = {
            "within_1pct_rate": fmt_pct(row["within_1pct_rate"]),
            "within_5pct_rate": fmt_pct(row["within_5pct_rate"]),
            "median_rmse_gap_pct": fmt_num(row["median_rmse_gap_pct"]),
            "q90_rmse_gap_pct": fmt_num(row["q90_rmse_gap_pct"]),
            "best_rmse_count": f"{int(row['best_rmse_count'])}",
            "fastest_within_1pct_count": f"{int(row['fastest_within_1pct_count'])}",
            "median_solver_time_sec": f"{row['median_solver_time_sec']:.2f}",
            "median_iterations": f"{row['median_iterations']:.1f}",
            "max_iter_rate": fmt_pct(row["max_iter_rate"]),
        }
        cells = [
            row["method"],
            *[
                maybe_bold(values[col], bool(best[col].loc[i]))
                for col in [
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
            ],
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize Bold values mark the best method in each metric column. Higher values are better for Within 1\%, Within 5\%, Best, and Fastest good; lower values are better for RMSE gaps, median time, median iterations, and Max-it.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def write_factor_summary(outputs: dict[str, pd.DataFrame], path: Path) -> None:
    variable = outputs["variable_family_summary"].set_index("variable_family").loc[
        ["Parallax", "SphInvRange", "XYInvZ", "XYZ", "SphRange"]
    ].reset_index()
    frame = outputs["frame_family_summary"].copy()
    blocks = [("Variable family", variable, "variable_family"), ("Frame family", frame, "frame_family")]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Factor-wise aggregation by object-point variable and reference-frame family in the single-thread benchmark.}",
        r"\label{tab:results_factor_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Family & Runs & \shortstack{Within\\1\%} & \shortstack{Within\\5\%} & \shortstack{Med. gap\\(\%)} & \shortstack{Med.\\it.} & \shortstack{Max-it.\\(\%)} \\",
        r"\midrule",
    ]
    for block_name, block, name_col in blocks:
        best = {
            "within_1pct_rate": mark_best(block["within_1pct_rate"], True),
            "within_5pct_rate": mark_best(block["within_5pct_rate"], True),
            "median_rmse_gap_pct": mark_best(block["median_rmse_gap_pct"], False),
            "median_iterations": mark_best(block["median_iterations"], False),
            "max_iter_rate": mark_best(block["max_iter_rate"], False),
        }
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{block_name}}}}} \\")
        for i, row in block.iterrows():
            cells = [
                row[name_col],
                f"{int(row['n'])}",
                maybe_bold(fmt_pct(row["within_1pct_rate"]), bool(best["within_1pct_rate"].loc[i])),
                maybe_bold(fmt_pct(row["within_5pct_rate"]), bool(best["within_5pct_rate"].loc[i])),
                maybe_bold(fmt_num(row["median_rmse_gap_pct"]), bool(best["median_rmse_gap_pct"].loc[i])),
                maybe_bold(f"{row['median_iterations']:.1f}", bool(best["median_iterations"].loc[i])),
                maybe_bold(fmt_pct(row["max_iter_rate"]), bool(best["max_iter_rate"].loc[i])),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if block_name == "Variable family":
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\textwidth}",
            r"\footnotesize Bold values mark the best entry within each family block. Higher values are better for Within 1\% and Within 5\%; lower values are better for median RMSE gap, median iterations, and Max-it.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def write_category_summary(category: pd.DataFrame, method_order: list[str], path: Path) -> None:
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{longtable}{@{}llrrrrrr@{}}",
        r"\caption{Category-wise robustness and convergence summary for every method in the single-thread benchmark.}",
        r"\label{tab:appendix_category_method_summary}\\",
        r"\toprule",
        r"Category & Method & Runs & \shortstack{Within\\1\%} & \shortstack{Within\\5\%} & \shortstack{Med. gap\\(\%)} & \shortstack{Med.\\it.} & \shortstack{Max-it.\\rate (\%)} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Category & Method & Runs & \shortstack{Within\\1\%} & \shortstack{Within\\5\%} & \shortstack{Med. gap\\(\%)} & \shortstack{Med.\\it.} & \shortstack{Max-it.\\rate (\%)} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{@{}r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for cat in CATEGORIES:
        block = category[category["category"].eq(cat)].set_index("method").reindex(method_order).reset_index()
        best = {
            "within_1pct_rate": mark_best(block["within_1pct_rate"], True),
            "within_5pct_rate": mark_best(block["within_5pct_rate"], True),
            "median_rmse_gap_pct": mark_best(block["median_rmse_gap_pct"], False),
            "median_iterations": mark_best(block["median_iterations"], False),
            "max_iter_rate": mark_best(block["max_iter_rate"], False),
        }
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{cat}}}}} \\")
        for i, row in block.iterrows():
            cells = [
                "",
                row["method"],
                f"{int(row['n'])}",
                maybe_bold(fmt_pct(row["within_1pct_rate"]), bool(best["within_1pct_rate"].loc[i])),
                maybe_bold(fmt_pct(row["within_5pct_rate"]), bool(best["within_5pct_rate"].loc[i])),
                maybe_bold(fmt_num(row["median_rmse_gap_pct"]), bool(best["median_rmse_gap_pct"].loc[i])),
                maybe_bold(f"{row['median_iterations']:.1f}", bool(best["median_iterations"].loc[i])),
                maybe_bold(fmt_pct(row["max_iter_rate"]), bool(best["max_iter_rate"].loc[i])),
            ]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\end{longtable}",
            r"\par\smallskip",
            r"\footnotesize Max-it. rate is the percentage of runs that reached the 100-iteration budget; 0.0 denotes no maximum-iteration run in that category--method block. Bold values mark the best entry within each category block. Higher values are better for Within 1\% and Within 5\%; lower values are better for median RMSE gap, median iterations, and Max-it. rate.",
            r"\endgroup",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def compute_endpoint_profiles(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    details = []
    fastest = clean.groupby("base_dataset")["solver_time_sec"].min().rename("fastest_solver_time_sec")
    frame = clean.join(fastest, on="base_dataset")
    for tol in TOLERANCES:
        label = f"{int(tol * 100)}%"
        for method in METHOD_ORDER:
            subset = frame[frame["method_str"].eq(method)].copy()
            success = subset["final_rmse_px"] <= (1.0 + tol) * subset["best_final_rmse_px"]
            ratio = subset["solver_time_sec"] / subset["fastest_solver_time_sec"]
            ratio = ratio.where(success, np.inf)
            for dataset, value, ok in zip(subset["base_dataset"], ratio, success):
                details.append(
                    {
                        "accuracy_tolerance": label,
                        "base_dataset": dataset,
                        "method": method,
                        "endpoint_ratio": float(value),
                        "endpoint_success": bool(ok),
                    }
                )
            rows.append(
                {
                    "accuracy_tolerance": label,
                    "method": method,
                    "accuracy_success_rate": float(success.mean()),
                    "profile_rate_at_alpha_2": float((ratio <= 2.0).mean()),
                    "profile_rate_at_alpha_5": float((ratio <= 5.0).mean()),
                }
            )
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(details)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary["tol_sort"] = summary["accuracy_tolerance"].map({"1%": 0, "5%": 1})
    summary = summary.sort_values(["tol_sort", "accuracy_success_rate", "profile_rate_at_alpha_5"], ascending=[True, False, False]).drop(columns="tol_sort")
    return summary, detail


def read_convergence(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda col: col in set(usecols))


def compute_first_passage(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    for _, row in clean.iterrows():
        conv_path = Path(row["report"]).parent / "convergence.csv"
        conv = read_convergence(
            conv_path,
            ["iteration", "rmse_px", "cumulative_time_sec", "step_successful"],
        )
        accepted = conv[conv["step_successful"].eq(1)].copy()
        for tol in TOLERANCES:
            threshold = (1.0 + tol) * row["best_final_rmse_px"]
            hit = accepted[accepted["rmse_px"] <= threshold].head(1)
            if hit.empty:
                first_time = math.inf
                first_iteration = math.nan
            else:
                first_time = float(hit["cumulative_time_sec"].iloc[0])
                first_iteration = int(hit["iteration"].iloc[0])
            raw_rows.append(
                {
                    "accuracy_tolerance": f"{int(tol * 100)}%",
                    "base_dataset": row["base_dataset"],
                    "method": row["method_str"],
                    "threshold_rmse_px": float(threshold),
                    "first_passage_sec": first_time,
                    "first_passage_iteration": first_iteration,
                    "final_success": bool(row["final_rmse_px"] <= threshold),
                }
            )
    detail = pd.DataFrame(raw_rows)
    ratios = []
    for (tol_label, dataset), group in detail.groupby(["accuracy_tolerance", "base_dataset"]):
        finite = group["first_passage_sec"].replace([np.inf, -np.inf], np.nan)
        best_time = finite.min()
        for idx, value in group["first_passage_sec"].items():
            ratios.append((idx, float(value / best_time) if math.isfinite(value) and best_time > 0 else math.inf))
    ratio_series = pd.Series({idx: value for idx, value in ratios}, name="first_passage_ratio")
    detail = detail.join(ratio_series)
    rows = []
    for (tol_label, method), group in detail.groupby(["accuracy_tolerance", "method"]):
        finite_times = group["first_passage_sec"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "accuracy_tolerance": tol_label,
                "method": method,
                "first_passage_success_rate": float(np.isfinite(group["first_passage_sec"]).mean()),
                "profile_rate_at_alpha_2": float((group["first_passage_ratio"] <= 2.0).mean()),
                "profile_rate_at_alpha_5": float((group["first_passage_ratio"] <= 5.0).mean()),
                "median_first_passage_sec": float(finite_times.median()) if len(finite_times) else math.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary["tol_sort"] = summary["accuracy_tolerance"].map({"1%": 0, "5%": 1})
    summary = summary.sort_values(["tol_sort", "first_passage_success_rate", "profile_rate_at_alpha_5"], ascending=[True, False, False]).drop(columns="tol_sort")
    return summary, detail


def plot_profiles(detail: pd.DataFrame, ratio_col: str, out_path: Path) -> None:
    finite = detail[ratio_col].replace([np.inf, -np.inf], np.nan).dropna()
    xmax = max(10.0, min(100.0, float(finite.max()) * 1.05 if len(finite) else 10.0))
    x_values = np.unique(np.r_[1.0, np.geomspace(1.001, xmax, 420)])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65), sharey=True)
    for ax, tol_label in zip(axes, ["1%", "5%"]):
        subset = detail[detail["accuracy_tolerance"].eq(tol_label)]
        for method in METHOD_ORDER:
            values = subset[subset["method"].eq(method)][ratio_col].to_numpy(dtype=float)
            y_values = np.asarray([(values <= x).mean() for x in x_values], dtype=float)
            ax.step(
                x_values,
                y_values,
                where="post",
                color=METHOD_COLORS[method],
                linewidth=2.0 if method.startswith("A2-") else 1.0,
                alpha=0.98 if method.startswith("A2-") else 0.78,
                label=method,
            )
        label_specs = [
            ("A2-Parallax-Mw", min(8.0, 0.70 * xmax), (4, 1), "bottom"),
            ("A2-Parallax-Mc", min(6.0, 0.60 * xmax), (4, -2), "top"),
        ]
        for method, target_x, text_offset, va in label_specs:
            values = subset[subset["method"].eq(method)][ratio_col].to_numpy(dtype=float)
            target_y = float((values <= target_x).mean())
            ax.annotate(
                method,
                xy=(target_x, target_y),
                xycoords="data",
                xytext=text_offset,
                textcoords="offset points",
                color=METHOD_COLORS[method],
                fontsize=7.6,
                ha="left",
                va=va,
            )
        ax.set_xscale("log")
        ax.set_xlim(1.0, xmax)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(r"Runtime ratio $\alpha$")
        ax.set_title(f"Within {tol_label} of best RMSE", loc="left")
        ax.grid(True, which="both", alpha=0.22, linewidth=0.55)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{100 * value:.0f}" for value in np.linspace(0, 1, 6)])
        ax.tick_params(axis="both", labelsize=8.2)

    axes[0].set_ylabel("Problems solved (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if not label.startswith("A2-")
    ]
    fig.legend(
        [item[0] for item in legend_items],
        [item[1] for item in legend_items],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=4,
        fontsize=7.6,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.255, top=0.86, wspace=0.15)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_robustness_pareto(method: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    frame = method.copy()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.2, 4.45),
        gridspec_kw={"width_ratios": [1.14, 1.0], "wspace": 0.30},
    )
    bar = frame.sort_values("within_1pct_rate", ascending=True)
    axes[0].barh(
        bar["method"],
        pct(bar["within_1pct_rate"]),
        color=[METHOD_COLORS.get(m, "0.6") for m in bar["method"]],
    )
    axes[0].set_xlabel("Within 1% of best RMSE (%)")
    axes[0].set_xlim(0, 100)
    axes[0].grid(True, axis="x", color="0.88", linewidth=0.5)
    axes[0].set_title("(a) Robustness")
    scatter = axes[1].scatter(
        frame["median_solver_time_sec"],
        pct(frame["within_1pct_rate"]),
        s=45 + 280 * frame["max_iter_rate"],
        c=[METHOD_COLORS.get(m, "0.6") for m in frame["method"]],
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    # Place the key labels in empty regions and connect them with fine leader lines.
    label_positions = {
        "A2-Parallax-Mw": ((44.0, 97.0), "left"),
        "A2-Parallax-Mc": ((44.0, 84.5), "left"),
        "A0-SphInvRange-W": ((84.0, 35.5), "right"),
        "A1-XYInvZ-Ac": ((72.0, 28.5), "right"),
    }
    for _, row in frame.iterrows():
        method_name = str(row["method"])
        if method_name not in label_positions:
            continue
        (label_x, label_y), ha = label_positions[method_name]
        axes[1].annotate(
            method_name,
            (row["median_solver_time_sec"], pct(row["within_1pct_rate"])),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=7,
            ha=ha,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.35",
                "linewidth": 0.55,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            annotation_clip=False,
        )
    axes[1].set_xlabel("Median solver time (s)")
    axes[1].set_ylabel("Within 1% of best RMSE (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_xlim(30, 85)
    axes[1].set_xticks([30, 40, 50, 60, 70, 80])
    axes[1].grid(True, which="both", color="0.88", linewidth=0.5)
    axes[1].set_title("(b) Absolute time")
    axes[1].legend(
        *scatter.legend_elements(prop="sizes", num=4, func=lambda s: (s - 45) / 280 * 100),
        title="Max-it. (%)",
        loc="center",
        bbox_to_anchor=(0.64, 0.61),
        frameon=True,
        framealpha=0.88,
        fontsize=7,
        title_fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_category_heatmap(category: pd.DataFrame, method_order: list[str], out_path: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    matrix = (
        category.pivot(index="method", columns="category", values="within_1pct_rate")
        .reindex(method_order)
        .reindex(columns=CATEGORIES)
    )
    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    im = ax.imshow(pct(matrix.to_numpy()), cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = pct(matrix.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 65 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Within 1% of best RMSE (%)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    ax.set_title("Category-wise robustness", fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compare_clean_and_diagnostic(clean: pd.DataFrame, diagnostic: pd.DataFrame) -> dict[str, float]:
    merged = clean.merge(
        diagnostic,
        on=["base_dataset", "quality_dataset", "method_str"],
        suffixes=("_clean", "_diagnostic"),
    )
    result: dict[str, float] = {
        "matched_rows": float(len(merged)),
        "status_differences": float((merged["status_clean"].astype(str) != merged["status_diagnostic"].astype(str)).sum()),
    }
    for col in [
        "final_rmse_px",
        "final_cost",
        "iterations",
        "accepted_steps",
        "rejected_steps",
        "linear_solver_iterations",
        "termination_type",
    ]:
        a = pd.to_numeric(merged[f"{col}_clean"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_diagnostic"], errors="coerce")
        result[f"{col}_max_abs_diff"] = float(np.nanmax(np.abs(a - b)))
        result[f"{col}_changed"] = float((np.abs(a - b) > 1e-12).sum())

    path_checked = 0
    path_row_mismatch = 0
    path_value_mismatch = 0
    max_path_rmse = 0.0
    max_path_cost = 0.0
    common_cols = ["iteration", "cost", "rmse_px", "step_successful", "linear_solver_iterations"]
    for _, row in merged.iterrows():
        clean_path = Path(row["report_clean"]).parent / "convergence.csv"
        diagnostic_path = Path(row["report_diagnostic"]).parent / "convergence_strict.csv"
        c = read_convergence(clean_path, common_cols)
        d = read_convergence(diagnostic_path, common_cols)
        path_checked += 1
        if len(c) != len(d):
            path_row_mismatch += 1
            continue
        local_mismatch = False
        for col in common_cols:
            diff = np.abs(pd.to_numeric(c[col], errors="coerce") - pd.to_numeric(d[col], errors="coerce"))
            if col == "rmse_px":
                max_path_rmse = max(max_path_rmse, float(diff.max()))
            elif col == "cost":
                max_path_cost = max(max_path_cost, float(diff.max()))
            if (diff > 1e-12).any():
                local_mismatch = True
        if local_mismatch:
            path_value_mismatch += 1
    result.update(
        {
            "path_checked": float(path_checked),
            "path_row_mismatch": float(path_row_mismatch),
            "path_value_mismatch": float(path_value_mismatch),
            "path_rmse_max_abs_diff": max_path_rmse,
            "path_cost_max_abs_diff": max_path_cost,
        }
    )
    return result


def write_threading_tables(
    clean: pd.DataFrame,
    diagnostic: pd.DataFrame,
    previous_clean: pd.DataFrame | None,
    problems_root: Path,
    tables: Path,
) -> None:
    consistency = compare_clean_and_diagnostic(clean, diagnostic)
    summary_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Numerical reproducibility, thread-count sensitivity, and initialization severity. Primary accuracy, robustness, and accuracy--runtime profiles use the single-thread execution; the archived multi-thread execution quantifies numerical sensitivity.}",
        r"\label{tab:threading_sensitivity_summary}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}p{0.47\textwidth}p{0.47\textwidth}@{}}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Single-thread path reproducibility}} \\",
        rf"Single-thread clean-logged vs strict diagnostic matched rows & {int(consistency['matched_rows'])} \\",
        rf"Final RMSE / cost max absolute difference & {consistency['final_rmse_px_max_abs_diff']:.1e} px / {consistency['final_cost_max_abs_diff']:.1e} \\",
        rf"Changed iterations / accepted steps / rejected steps & {int(consistency['iterations_changed'])} / {int(consistency['accepted_steps_changed'])} / {int(consistency['rejected_steps_changed'])} \\",
        rf"Changed termination types & {int(consistency['termination_type_changed'])} \\",
        rf"Convergence paths compared / value mismatches & {int(consistency['path_checked'])} / {int(consistency['path_value_mismatch'])} \\",
        rf"Path RMSE / cost max absolute difference & {consistency['path_rmse_max_abs_diff']:.1e} px / {consistency['path_cost_max_abs_diff']:.1e} \\",
    ]
    if previous_clean is not None:
        prev_method = build_outputs(previous_clean)["method_summary"].set_index("method")
        new_method = build_outputs(clean)["method_summary"].set_index("method")
        joined = prev_method.join(new_method, lsuffix="_previous", rsuffix="_single")
        max_w1 = pct((joined["within_1pct_rate_single"] - joined["within_1pct_rate_previous"]).abs().max())
        max_w5 = pct((joined["within_5pct_rate_single"] - joined["within_5pct_rate_previous"]).abs().max())
        time_ratio = joined["median_solver_time_sec_single"] / joined["median_solver_time_sec_previous"]

        keys = ["base_dataset", "method_str"]
        previous_pairs = previous_clean.set_index(keys)
        single_pairs = clean.set_index(keys)
        common = previous_pairs.index.intersection(single_pairs.index)
        pairwise = previous_pairs.loc[common].join(
            single_pairs.loc[common],
            lsuffix="_multi",
            rsuffix="_single",
        )
        rmse_min = pairwise[["final_rmse_px_multi", "final_rmse_px_single"]].min(axis=1)
        rmse_max = pairwise[["final_rmse_px_multi", "final_rmse_px_single"]].max(axis=1)
        pairwise["final_rmse_factor"] = rmse_max / rmse_min
        pairwise["rmse_change_gt_1pct"] = pairwise["final_rmse_factor"].gt(1.01)
        pairwise["rmse_change_gt_5pct"] = pairwise["final_rmse_factor"].gt(1.05)
        pairwise["rmse_change_gt_2x"] = pairwise["final_rmse_factor"].gt(2.0)
        previous_best = pairwise.groupby(level="base_dataset")["final_rmse_px_multi"].transform("min")
        single_best = pairwise.groupby(level="base_dataset")["final_rmse_px_single"].transform("min")
        pairwise["within_1pct_multi"] = pairwise["final_rmse_px_multi"].le(1.01 * previous_best)
        pairwise["within_1pct_single"] = pairwise["final_rmse_px_single"].le(1.01 * single_best)
        pairwise["within_5pct_multi"] = pairwise["final_rmse_px_multi"].le(1.05 * previous_best)
        pairwise["within_5pct_single"] = pairwise["final_rmse_px_single"].le(1.05 * single_best)

        sensitive_1 = pairwise["rmse_change_gt_1pct"]
        sensitive_5 = pairwise["rmse_change_gt_5pct"]
        count_1 = int(sensitive_1.sum())
        count_5 = int(sensitive_5.sum())
        count_2x = int(pairwise["rmse_change_gt_2x"].sum())
        classification_changes_1 = int(
            pairwise["within_1pct_multi"].ne(pairwise["within_1pct_single"]).sum()
        )
        classification_changes_5 = int(
            pairwise["within_5pct_multi"].ne(pairwise["within_5pct_single"]).sum()
        )
        max_iter_either = int(
            (
                pairwise.loc[sensitive_5, "iterations_multi"].ge(100)
                | pairwise.loc[sensitive_5, "iterations_single"].ge(100)
            ).sum()
        )
        max_iter_both = int(
            (
                pairwise.loc[sensitive_5, "iterations_multi"].ge(100)
                & pairwise.loc[sensitive_5, "iterations_single"].ge(100)
            ).sum()
        )
        accepted_changed = int(
            pairwise.loc[sensitive_5, "accepted_steps_multi"]
            .ne(pairwise.loc[sensitive_5, "accepted_steps_single"])
            .sum()
        )
        rejected_changed = int(
            pairwise.loc[sensitive_5, "rejected_steps_multi"]
            .ne(pairwise.loc[sensitive_5, "rejected_steps_single"])
            .sum()
        )

        method_sensitivity = (
            pairwise.groupby(level="method_str")["rmse_change_gt_1pct"]
            .sum()
            .sort_values(ascending=False)
            .rename("pairs_gt_1pct")
        )
        dataset_sensitivity = (
            pairwise.groupby(level="base_dataset")["rmse_change_gt_1pct"]
            .sum()
            .sort_values(ascending=False)
            .rename("methods_gt_1pct")
        )
        high_sensitivity_datasets = dataset_sensitivity[dataset_sensitivity.ge(5)].index
        dataset_difficulty = clean.groupby("base_dataset").agg(
            initial_rmse_px=("initial_rmse_px", "first"),
            max_iteration_methods=("iterations", lambda values: int(values.ge(100).sum())),
            converged_methods=("termination_type", lambda values: int(values.eq(0).sum())),
            median_iterations=("iterations", "median"),
            median_rejected_steps=("rejected_steps", "median"),
        )
        high_difficulty = dataset_difficulty.loc[high_sensitivity_datasets].median()
        other_difficulty = dataset_difficulty.loc[
            ~dataset_difficulty.index.isin(high_sensitivity_datasets)
        ].median()

        dataset_initial = clean[["base_dataset", "initial_rmse_px"]].drop_duplicates("base_dataset").copy()
        focal_lengths = []
        for dataset_name in dataset_initial["base_dataset"].astype(str):
            calibration = np.loadtxt(
                problems_root / dataset_name / "quality" / "Initial Value" / "cal.txt"
            )
            focal_lengths.append(float(np.mean([calibration[0, 0], calibration[1, 1]])))
        dataset_initial["mean_focal_px"] = focal_lengths
        dataset_initial["normalized_initial_rmse"] = (
            dataset_initial["initial_rmse_px"] / dataset_initial["mean_focal_px"]
        )
        mild_datasets = dataset_initial.loc[
            dataset_initial["normalized_initial_rmse"].le(0.1), "base_dataset"
        ]
        mild = clean[clean["base_dataset"].isin(mild_datasets)].copy()
        mild["best_final_rmse_px"] = mild.groupby("base_dataset")["final_rmse_px"].transform("min")
        mild["within_1pct"] = mild["final_rmse_px"].le(1.01 * mild["best_final_rmse_px"])
        mild_rates = mild.groupby("method_str")["within_1pct"].mean().sort_values(ascending=False)
        best_non_parallax_rate = mild_rates[~mild_rates.index.str.startswith("A2-")].max()

        pairwise.reset_index().to_csv(tables / "threading_pair_comparison.csv", index=False)
        method_sensitivity.reset_index().to_csv(
            tables / "threading_pair_sensitivity_by_method.csv", index=False
        )
        dataset_sensitivity.reset_index().to_csv(
            tables / "threading_pair_sensitivity_by_dataset.csv", index=False
        )
        dataset_initial.to_csv(tables / "initialization_severity_by_dataset.csv", index=False)
        mild_rates.rename("within_1pct_rate").reset_index().to_csv(
            tables / "mild_initialization_method_robustness.csv", index=False
        )

        summary_lines.extend(
            [
                r"\addlinespace",
                r"\multicolumn{2}{@{}l}{\textit{Archived multi-thread vs single-thread sensitivity}} \\",
                rf"Matched method--dataset pairs & {len(pairwise)} \\",
                rf"Final-RMSE factor $>1.01$ / $>1.05$ / $>2$ & {count_1} ({pct(count_1 / len(pairwise)):.1f}\%) / {count_5} ({pct(count_5 / len(pairwise)):.1f}\%) / {count_2x} ({pct(count_2x / len(pairwise)):.1f}\%) \\",
                rf"Changed within-1\% / within-5\% classifications & {classification_changes_1} / {classification_changes_5} \\",
                rf"Among the {count_5} pairs above 1.05: at least one / both runs reached 100 iterations & {max_iter_either} / {max_iter_both} \\",
                rf"Among the {count_5} pairs above 1.05: changed accepted / rejected step counts & {accepted_changed} / {rejected_changed} \\",
                rf"Largest contributors above 1.01 & A0-SphRange-W: {int(method_sensitivity['A0-SphRange-W'])}; A0-SphInvRange-W: {int(method_sensitivity['A0-SphInvRange-W'])} \\",
                rf"Multi-thread vs single-thread max method-level change & {max_w1:.1f} pp within 1\%; {max_w5:.1f} pp within 5\% \\",
                rf"Median solver-time ratio range, single-thread / multi-thread & {time_ratio.min():.2f}--{time_ratio.max():.2f} \\",
                r"\addlinespace",
                r"\multicolumn{2}{@{}l}{\textit{Initialization severity}} \\",
                rf"Five datasets with $\geq5$ methods above 1.01 vs remaining 100 datasets & Median initial RMSE: {high_difficulty['initial_rmse_px']:.0f} vs {other_difficulty['initial_rmse_px']:.0f} px; max-iteration methods: {high_difficulty['max_iteration_methods']:.0f} vs {other_difficulty['max_iteration_methods']:.0f}; rejected steps: {high_difficulty['median_rejected_steps']:.1f} vs {other_difficulty['median_rejected_steps']:.1f} \\",
                rf"Initial RMSE $\leq0.1$ mean focal length ({len(mild_datasets)} datasets): within-1\% rate & A2-Parallax-Mw: {fmt_pct(mild_rates['A2-Parallax-Mw'])}\%; A2-Parallax-Mc: {fmt_pct(mild_rates['A2-Parallax-Mc'])}\%; best non-parallax rate: {fmt_pct(best_non_parallax_rate)}\% \\",
            ]
        )
        comparison = pd.DataFrame(
            {
                "method": METHOD_ORDER,
                "previous_within_1pct": prev_method.reindex(METHOD_ORDER)["within_1pct_rate"].to_numpy(),
                "single_within_1pct": new_method.reindex(METHOD_ORDER)["within_1pct_rate"].to_numpy(),
                "previous_within_5pct": prev_method.reindex(METHOD_ORDER)["within_5pct_rate"].to_numpy(),
                "single_within_5pct": new_method.reindex(METHOD_ORDER)["within_5pct_rate"].to_numpy(),
                "median_time_ratio": time_ratio.reindex(METHOD_ORDER).to_numpy(),
            }
        )
        comparison.to_csv(tables / "threading_method_comparison.csv", index=False)
    summary_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    write_text(tables / "threading_sensitivity_summary.tex", "\n".join(summary_lines))


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, value) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def rank_biserial_for_lower_better(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    nonzero = diff != 0
    if not np.any(nonzero):
        return 0.0
    diff = diff[nonzero]
    ranks = stats.rankdata(np.abs(diff))
    w_a_better = ranks[diff < 0].sum()
    w_b_better = ranks[diff > 0].sum()
    total = ranks.sum()
    return float((w_a_better - w_b_better) / total) if total else 0.0


def fmt_p(value: float) -> str:
    if value < 0.001 or value >= 1000:
        return f"{value:.2e}"
    return f"{value:.2f}"


def write_pairwise_statistics(clean: pd.DataFrame, tables: Path) -> None:
    wide = clean.pivot(index="base_dataset", columns="method_str", values="final_rmse_px").reindex(columns=METHOD_ORDER)
    friedman = stats.friedmanchisquare(*[wide[method].to_numpy() for method in METHOD_ORDER])
    n_blocks = wide.shape[0]
    n_methods = wide.shape[1]
    kendall_w = float(friedman.statistic / (n_blocks * (n_methods - 1)))
    comparisons = [
        ("A2-Parallax-Mw", "A2-Parallax-Mc"),
        ("A2-Parallax-Mw", "A0-SphInvRange-W"),
        ("A2-Parallax-Mw", "A1-XYInvZ-Ac"),
        ("A2-Parallax-Mw", "A1-SphInvRange-Aw"),
        ("A2-Parallax-Mc", "A0-SphInvRange-W"),
        ("A2-Parallax-Mc", "A1-XYInvZ-Ac"),
        ("A2-Parallax-Mc", "A1-SphInvRange-Aw"),
        ("A0-SphInvRange-W", "A0-XYInvZ-W"),
        ("A1-XYInvZ-Ac", "A1-XYInvZ-Aw"),
        ("A1-SphInvRange-Ac", "A1-SphInvRange-Aw"),
    ]
    rows = []
    p_values = []
    for left, right in comparisons:
        a = wide[left].to_numpy(dtype=float)
        b = wide[right].to_numpy(dtype=float)
        test = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p_values.append(float(test.pvalue))
        rows.append(
            {
                "method_a": left,
                "method_b": right,
                "a_better": int((a < b).sum()),
                "b_better": int((b < a).sum()),
                "median_ratio_a_over_b": float(np.median(a / b)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial_a_better": rank_biserial_for_lower_better(a, b),
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted):
        row["holm_p_value"] = value
    table = pd.DataFrame(rows)
    table.to_csv(tables / "results_pairwise_statistics.csv", index=False)
    write_text(
        tables / "results_friedman_summary.txt",
        "\n".join(
            [
                f"friedman_statistic: {float(friedman.statistic)}",
                f"friedman_p_value: {float(friedman.pvalue)}",
                f"kendall_w: {kendall_w}",
                f"n_blocks: {n_blocks}",
                f"n_methods: {n_methods}",
                "",
            ]
        ),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Matched pairwise Wilcoxon signed-rank comparisons on single-thread final reprojection RMSE. The overall Friedman test gives $\chi^2={float(friedman.statistic):.1f}$, $p={float(friedman.pvalue):.2e}$, Kendall's $W={kendall_w:.3f}$ for {n_blocks} matched datasets and {n_methods} methods.}}",
        r"\label{tab:results_pairwise_statistics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method A & Method B & A/B wins & Median RMSE ratio & Rank-biserial & Holm $p$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            " & ".join(
                [
                    row["method_a"],
                    row["method_b"],
                    f"{int(row['a_better'])}/{int(row['b_better'])}",
                    f"{row['median_ratio_a_over_b']:.3f}",
                    f"{row['rank_biserial_a_better']:.3f}",
                    fmt_p(row["holm_p_value"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"\end{table*}", ""])
    write_text(tables / "results_pairwise_statistics.tex", "\n".join(lines))


def fmt_metric(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value != 0.0 and (abs(value) >= 1.0e4 or abs(value) < 1.0e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def best_metric(value: float, best: float, digits: int = 3) -> str:
    text = fmt_metric(value, digits)
    if pd.notna(value) and np.isclose(float(value), float(best), rtol=1e-12, atol=1e-15):
        return rf"\textbf{{{text}}}"
    return text


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        vertex_count = None
        for line in stream:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Cannot find vertex count in {path}")
        data = np.loadtxt(stream, max_rows=vertex_count, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :3]


def find_camera_file(folder: Path) -> Path:
    direct = folder / "Cam.txt"
    if direct.exists():
        return direct
    candidates = sorted(folder.glob("Cam*.txt"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Cannot identify camera file in {folder}")


def similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    scale = float(np.dot(singular_values, sign) / max(variance, 1e-30))
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation


def euler_to_world_to_camera(euler: np.ndarray) -> np.ndarray:
    euler = np.atleast_2d(np.asarray(euler, dtype=np.float64))
    ey = euler[:, 0]
    ex = euler[:, 1]
    ez = euler[:, 2]
    c1 = np.cos(ey)
    c2 = np.cos(ex)
    c3 = np.cos(ez)
    s1 = np.sin(ey)
    s2 = np.sin(ex)
    s3 = np.sin(ez)
    rotation = np.empty((len(euler), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = c1 * c3 - s1 * s2 * s3
    rotation[:, 0, 1] = c2 * s3
    rotation[:, 0, 2] = s1 * c3 + c1 * s2 * s3
    rotation[:, 1, 0] = -c1 * s3 - s1 * s2 * c3
    rotation[:, 1, 1] = c2 * c3
    rotation[:, 1, 2] = -s1 * s3 + c1 * s2 * c3
    rotation[:, 2, 0] = -s1 * c2
    rotation[:, 2, 1] = -s2
    rotation[:, 2, 2] = c1 * c2
    return rotation


def camera_rotation_errors_deg(
    estimated_euler: np.ndarray, reference_euler: np.ndarray, alignment_rotation: np.ndarray
) -> np.ndarray:
    estimated = euler_to_world_to_camera(estimated_euler)
    reference = euler_to_world_to_camera(reference_euler)
    estimated_in_reference_frame = estimated @ alignment_rotation.T
    relative = estimated_in_reference_frame @ np.transpose(reference, (0, 2, 1))
    trace = np.trace(relative, axis1=1, axis2=2)
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_reference_recovery(
    diagnostic: pd.DataFrame,
    diagnostic_root: Path,
    problems_root: Path,
    tables: Path,
) -> pd.DataFrame:
    rows = []
    for base_dataset, group in diagnostic.groupby("base_dataset", sort=True):
        original = problems_root / str(base_dataset) / "original"
        quality = problems_root / str(base_dataset) / "quality" / "Initial Value"
        reference_points = np.loadtxt(original / "XYZ.txt", dtype=np.float64)
        reference_pose = np.loadtxt(find_camera_file(original), dtype=np.float64)
        reference_euler = reference_pose[:, :3]
        reference_cameras = reference_pose[:, 3:6]

        initial_points = np.loadtxt(quality / "XYZ.txt", dtype=np.float64)
        initial_pose = np.loadtxt(find_camera_file(quality), dtype=np.float64)
        initial_euler = initial_pose[:, :3]
        initial_cameras = initial_pose[:, 3:6]
        initial_source = np.vstack([initial_cameras, initial_points])
        initial_target = np.vstack([reference_cameras, reference_points])
        initial_aligned, initial_scale, initial_alignment_rotation = similarity_transform(
            initial_source, initial_target
        )
        initial_camera_errors = np.linalg.norm(
            initial_aligned[: len(reference_cameras)] - reference_cameras,
            axis=1,
        )
        initial_point_errors = np.linalg.norm(
            initial_aligned[len(reference_cameras) :] - reference_points,
            axis=1,
        )
        initial_point_rmse = float(np.sqrt(np.mean(initial_point_errors**2)))
        initial_camera_rmse = float(np.sqrt(np.mean(initial_camera_errors**2)))
        initial_rotation_errors = camera_rotation_errors_deg(
            initial_euler, reference_euler, initial_alignment_rotation
        )
        initial_camera_rotation_median = float(np.median(initial_rotation_errors))

        for record in group.itertuples(index=False):
            method = str(getattr(record, "method_str", getattr(record, "method", "")))
            quality_dataset = str(getattr(record, "quality_dataset", "Initial Value"))
            run_root = diagnostic_root / str(base_dataset) / quality_dataset / method
            pose_path = run_root / "FinalPose.txt"
            point_path = run_root / "Final3D.ply"
            if not pose_path.exists() or not point_path.exists():
                continue
            estimated_pose = np.loadtxt(pose_path, dtype=np.float64)
            estimated_euler = estimated_pose[:, :3]
            estimated_cameras = estimated_pose[:, 3:6]
            estimated_points = read_ply_vertices(point_path)
            if len(estimated_points) != len(reference_points) or len(estimated_cameras) != len(reference_cameras):
                raise ValueError(f"Size mismatch for {base_dataset} / {method}")
            source = np.vstack([estimated_cameras, estimated_points])
            target = np.vstack([reference_cameras, reference_points])
            aligned, scale, alignment_rotation = similarity_transform(source, target)
            camera_aligned = aligned[: len(reference_cameras)]
            point_aligned = aligned[len(reference_cameras) :]
            camera_error = np.linalg.norm(camera_aligned - reference_cameras, axis=1)
            point_error = np.linalg.norm(point_aligned - reference_points, axis=1)
            rotation_error = camera_rotation_errors_deg(estimated_euler, reference_euler, alignment_rotation)
            point_rmse = float(np.sqrt(np.mean(point_error**2)))
            camera_rmse = float(np.sqrt(np.mean(camera_error**2)))
            rows.append(
                {
                    "base_dataset": str(base_dataset),
                    "quality_dataset": quality_dataset,
                    "method": method,
                    "category": str(getattr(record, "category", str(base_dataset).split("__", 1)[0])),
                    "cameras": int(getattr(record, "cameras", len(reference_cameras))),
                    "points": int(getattr(record, "points", len(reference_points))),
                    "similarity_scale": scale,
                    "initial_similarity_scale": initial_scale,
                    "initial_point_rmse": initial_point_rmse,
                    "point_rmse": point_rmse,
                    "point_median_error": float(np.median(point_error)),
                    "point_p90_error": float(np.percentile(point_error, 90)),
                    "point_p95_error": float(np.percentile(point_error, 95)),
                    "initial_camera_center_rmse": initial_camera_rmse,
                    "camera_center_rmse": camera_rmse,
                    "camera_center_median_error": float(np.median(camera_error)),
                    "initial_camera_rotation_median_deg": initial_camera_rotation_median,
                    "camera_rotation_median_deg": float(np.median(rotation_error)),
                    "camera_rotation_p90_deg": float(np.percentile(rotation_error, 90)),
                    "point_rmse_recovery_ratio": point_rmse / max(initial_point_rmse, 1e-30),
                    "camera_center_rmse_recovery_ratio": camera_rmse / max(initial_camera_rmse, 1e-30),
                }
            )
        print(f"reference recovery: {base_dataset} ({len(rows)} rows total)")
    recovery = pd.DataFrame(rows)
    recovery.to_csv(tables / "results_reference_recovery.csv", index=False)
    return recovery


def write_reference_recovery_table(recovery: pd.DataFrame, tables: Path) -> pd.DataFrame:
    recovery = recovery.copy()
    for col in [
        "point_rmse",
        "point_median_error",
        "point_p90_error",
        "camera_center_rmse",
        "camera_rotation_median_deg",
        "point_rmse_recovery_ratio",
    ]:
        recovery[col] = pd.to_numeric(recovery[col], errors="coerce")
    summary = (
        recovery.groupby("method")
        .agg(
            n=("method", "size"),
            median_point_rmse=("point_rmse", "median"),
            q75_point_rmse=("point_rmse", lambda x: x.quantile(0.75)),
            median_point_ratio=("point_rmse_recovery_ratio", "median"),
            median_point_error=("point_median_error", "median"),
            median_camera_rmse=("camera_center_rmse", "median"),
            median_camera_rotation=("camera_rotation_median_deg", "median"),
        )
        .reset_index()
    )
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("median_point_rmse")
    summary.to_csv(tables / "results_reference_recovery_summary.csv", index=False)
    best_values = {
        col: float(summary[col].min())
        for col in [
            "median_point_rmse",
            "q75_point_rmse",
            "median_point_ratio",
            "median_point_error",
            "median_camera_rmse",
            "median_camera_rotation",
        ]
    }
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    tex_escape(str(row["method"])),
                    f"{int(row['n'])}",
                    best_metric(row["median_point_rmse"], best_values["median_point_rmse"]),
                    best_metric(row["q75_point_rmse"], best_values["q75_point_rmse"]),
                    best_metric(row["median_point_ratio"], best_values["median_point_ratio"]),
                    best_metric(row["median_point_error"], best_values["median_point_error"]),
                    best_metric(row["median_camera_rmse"], best_values["median_camera_rmse"]),
                    best_metric(row["median_camera_rotation"], best_values["median_camera_rotation"]),
                ]
            )
            + r" \\"
        )
    write_text(
        tables / "results_reference_recovery.tex",
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Single-thread original-reference reconstruction recovery after Sim(3) alignment to the BA Datasets reference reconstruction.}",
                r"\label{tab:results_reference_recovery}",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{3.2pt}",
                r"\renewcommand{\arraystretch}{1.12}",
                r"\begin{tabular}{@{}lrrrrrrr@{}}",
                r"\toprule",
                r"\multirow{2}{*}{Method} & \multirow{2}{*}{Runs} & Med. Pt. & Q75 Pt. & Rec. & Med. Pt. & Cam. Ctr. & Cam. Rot. \\",
                r" & & RMSE & RMSE & ratio & err. & RMSE & (deg) \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\par\smallskip",
                r"\begin{minipage}{0.98\textwidth}",
                r"\footnotesize Lower values are better for all reported metrics; bold values mark the best method in each metric column. Med. Pt. RMSE is the median dataset-level object-point RMSE; Q75 Pt. RMSE is its 75th percentile across datasets; Rec. ratio is final/initial aligned point RMSE; Med. Pt. err. is the median per-point Euclidean error; Cam. Ctr. is camera-center RMSE; Cam. Rot. is the median camera-rotation error after Sim(3) alignment.",
                r"\end{minipage}",
                r"\end{table*}",
                "",
            ]
        ),
    )
    return summary


def reference_recovery_tests(recovery: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    wide = recovery.pivot(index="base_dataset", columns="method", values="point_rmse").reindex(columns=METHOD_ORDER)
    friedman = stats.friedmanchisquare(*[wide[method].to_numpy() for method in METHOD_ORDER])
    n_blocks = wide.shape[0]
    n_methods = wide.shape[1]
    comparisons = [
        ("A2-Parallax-Mw", "A2-Parallax-Mc"),
        ("A2-Parallax-Mw", "A0-SphInvRange-W"),
        ("A2-Parallax-Mw", "A1-XYInvZ-Ac"),
        ("A2-Parallax-Mw", "A1-SphInvRange-Aw"),
        ("A2-Parallax-Mc", "A0-SphInvRange-W"),
        ("A2-Parallax-Mc", "A1-XYInvZ-Ac"),
        ("A2-Parallax-Mc", "A1-SphInvRange-Aw"),
        ("A0-SphInvRange-W", "A0-XYInvZ-W"),
        ("A1-XYInvZ-Ac", "A1-XYInvZ-Aw"),
        ("A1-SphInvRange-Ac", "A1-SphInvRange-Aw"),
    ]
    rows = []
    p_values = []
    for left, right in comparisons:
        a = wide[left].to_numpy(dtype=float)
        b = wide[right].to_numpy(dtype=float)
        log_ratio = np.log(a / b)
        test = stats.wilcoxon(log_ratio, zero_method="wilcox", alternative="two-sided")
        p_values.append(float(test.pvalue))
        direction = np.log(b / a)
        nonzero = np.abs(direction) > 1e-15
        ranks = stats.rankdata(np.abs(direction[nonzero]))
        rank_biserial = float(np.sum(np.sign(direction[nonzero]) * ranks) / np.sum(ranks))
        rows.append(
            {
                "method_a": left,
                "method_b": right,
                "a_better": int((a < b).sum()),
                "b_better": int((b < a).sum()),
                "median_ratio_a_over_b": float(np.median(a / b)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial_a_better": rank_biserial,
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted):
        row["holm_p_value"] = value
    pairwise = pd.DataFrame(rows)
    stats_info = {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "kendall_w": float(friedman.statistic / (n_blocks * (n_methods - 1))),
        "n_blocks": int(n_blocks),
        "n_methods": int(n_methods),
    }
    return pairwise, stats_info


def write_reference_statistics_table(
    pairwise: pd.DataFrame,
    stats_info: dict[str, float],
    tables: Path,
) -> None:
    pairwise.to_csv(tables / "results_reference_pairwise_statistics.csv", index=False)
    write_text(
        tables / "results_reference_friedman_summary.txt",
        "\n".join(f"{key}: {value}" for key, value in stats_info.items()) + "\n",
    )
    rows = []
    for _, row in pairwise.iterrows():
        rows.append(
            " & ".join(
                [
                    tex_escape(row["method_a"]),
                    tex_escape(row["method_b"]),
                    f"{int(row['a_better'])}/{int(row['b_better'])}",
                    fmt_metric(row["median_ratio_a_over_b"], 3),
                    fmt_metric(row["rank_biserial_a_better"], 3),
                    fmt_p(row["holm_p_value"]),
                ]
            )
            + r" \\"
        )
    write_text(
        tables / "results_reference_pairwise_statistics.tex",
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                rf"\caption{{Matched pairwise Wilcoxon signed-rank comparisons on single-thread Sim(3)-aligned reference point RMSE. The overall Friedman test gives $\chi^2={stats_info['friedman_statistic']:.1f}$, $p={stats_info['friedman_p_value']:.2e}$, Kendall's $W={stats_info['kendall_w']:.3f}$ for {stats_info['n_blocks']} matched datasets and {stats_info['n_methods']} methods.}}",
                r"\label{tab:results_reference_pairwise_statistics}",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{3.0pt}",
                r"\renewcommand{\arraystretch}{1.08}",
                r"\begin{tabular}{llrrrr}",
                r"\toprule",
                r"Method A & Method B & A/B wins & Median RMSE ratio & Rank-biserial & Holm $p$ \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}%",
                r"\end{table*}",
                "",
            ]
        ),
    )


def plot_reference_recovery(recovery: pd.DataFrame, figures: Path) -> None:
    recovery = recovery.copy()
    method_order = (
        recovery.groupby("method", observed=False)["point_rmse"]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=recovery,
        x="method",
        y="point_rmse",
        order=method_order,
        ax=ax,
        fliersize=0.8,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(method, "#777777") for method in method_order],
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Aligned point RMSE to reference")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "results_reference_point_rmse.pdf", bbox_inches="tight")
    plt.close(fig)

    category_order = [name for name in CATEGORIES if name in recovery["category"].unique()]
    category_labels = {
        "Close-Range": "Close",
        "Oblique-5": "Oblique",
        "UAV": "UAV",
        "Vehicle": "Vehicle",
    }
    ratio = (
        recovery.groupby(["method", "category"], observed=False)["point_rmse_recovery_ratio"]
        .median()
        .reset_index()
    )
    ratio_pivot = ratio.pivot(index="method", columns="category", values="point_rmse_recovery_ratio")
    ratio_pivot = ratio_pivot.loc[method_order, category_order]
    ratio_pivot = ratio_pivot.rename(columns=category_labels)

    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    sns.heatmap(
        ratio_pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        cbar_kws={"label": "Median final / initial aligned point RMSE"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(figures / "results_reference_recovery_ratio_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.65))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)
    ax_box = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    sns.boxplot(
        data=recovery,
        x="point_rmse",
        y="method",
        order=method_order,
        ax=ax_box,
        fliersize=0.7,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(method, "#777777") for method in method_order],
        orient="h",
    )
    ax_box.set_xscale("log")
    ax_box.set_title("(a) Reference point RMSE", loc="left", fontsize=10, fontweight="bold")
    ax_box.set_xlabel("Reference point RMSE", fontsize=9.5)
    ax_box.set_ylabel("")
    ax_box.tick_params(axis="x", labelsize=8.5)
    ax_box.tick_params(axis="y", labelsize=7.8, pad=1.5)
    ax_box.grid(axis="x", alpha=0.25)

    ratio_ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ratio_tick_labels = ["1e-4", "1e-3", "1e-2", "0.1", "1", "10"]
    heat = sns.heatmap(
        ratio_pivot,
        cmap="viridis_r",
        norm=matplotlib.colors.LogNorm(vmin=ratio_ticks[0], vmax=ratio_ticks[-1]),
        ax=ax_heat,
        yticklabels=False,
        cbar_kws={"ticks": ratio_ticks, "fraction": 0.08, "pad": 0.06},
    )
    ax_heat.set_title("(b) Recovery ratio", loc="left", fontsize=10, fontweight="bold")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", labelrotation=0, labelsize=8.5)
    colorbar = heat.collections[0].colorbar
    colorbar.ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ratio_ticks))
    colorbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(ratio_tick_labels))
    colorbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.set_title("Ratio", fontsize=8.5, pad=4)

    fig.subplots_adjust(left=0.26, right=0.95, bottom=0.14, top=0.92)
    fig.savefig(figures / "results_reference_recovery_combined.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def termination_counts(clean: pd.DataFrame) -> dict[str, int]:
    type0 = clean[clean["termination_type"].eq(0)]
    return {
        "type0": int(len(type0)),
        "type1": int(clean["termination_type"].eq(1).sum()),
        "type0_outside_1pct": int((~type0["within_1pct"]).sum()),
        "type0_outside_5pct": int((~type0["within_5pct"]).sum()),
    }


def write_key_numbers(
    clean: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    endpoint_summary: pd.DataFrame,
    first_summary: pd.DataFrame,
    tables: Path,
) -> None:
    method = outputs["method_summary"].set_index("method")
    key_methods = ["A2-Parallax-Mw", "A2-Parallax-Mc", "A0-SphInvRange-W"]
    rows = []
    for m in key_methods:
        row = method.loc[m]
        rows.append(
            {
                "method": m,
                "within_1pct": row["within_1pct_rate"],
                "within_5pct": row["within_5pct_rate"],
                "best_rmse_count": int(row["best_rmse_count"]),
                "fastest_within_1pct_count": int(row["fastest_within_1pct_count"]),
                "median_time_sec": row["median_solver_time_sec"],
                "median_iterations": row["median_iterations"],
                "max_iter_rate": row["max_iter_rate"],
            }
        )
    pd.DataFrame(rows).to_csv(tables / "singlethread_key_method_numbers.csv", index=False)
    term = termination_counts(clean)
    profile_rows = []
    for profile_name, frame, metric in [
        ("endpoint", endpoint_summary, "accuracy_success_rate"),
        ("first_passage", first_summary, "first_passage_success_rate"),
    ]:
        for tol in ["1%", "5%"]:
            for method_name in ["A2-Parallax-Mw", "A2-Parallax-Mc", "A0-SphInvRange-W"]:
                row = frame[frame["accuracy_tolerance"].eq(tol) & frame["method"].astype(str).eq(method_name)].iloc[0]
                profile_rows.append(
                    {
                        "profile": profile_name,
                        "accuracy_tolerance": tol,
                        "method": method_name,
                        "success_rate": row[metric],
                        "rate_at_alpha_2": row["profile_rate_at_alpha_2"],
                        "rate_at_alpha_5": row["profile_rate_at_alpha_5"],
                    }
                )
    pd.DataFrame(profile_rows).to_csv(tables / "singlethread_profile_key_numbers.csv", index=False)
    pd.DataFrame([term]).to_csv(tables / "singlethread_termination_counts.csv", index=False)


def main() -> None:
    args = parse_args()
    tables, figures = ensure_dirs(args.manuscript_root)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    project_root = Path(__file__).resolve().parents[2]
    legacy_assets = load_module(
        "legacy_results_assets",
        project_root / "documents" / "ISPRS" / "New" / "generate_results_assets.py",
    )
    legacy_profiles = load_module(
        "legacy_performance_profiles",
        project_root / "CEP-BA-Bench" / "tools" / "generate_performance_profiles.py",
    )
    legacy_assets.CLEAN = args.clean_root
    legacy_assets.DIAG = args.diagnostic_root
    legacy_assets.FIG = figures
    legacy_assets.TAB = tables
    clean_summary = args.clean_root / "summary.csv"
    diagnostic_summary = args.diagnostic_root / "summary.csv"
    clean = enrich_summary(clean_summary)
    diagnostic = enrich_summary(diagnostic_summary)
    previous_clean = None
    if args.previous_clean_root is not None:
        previous_clean = enrich_summary(args.previous_clean_root / "summary.csv")

    outputs = build_outputs(clean)
    method = outputs["method_summary"]
    method_order = method["method"].astype(str).tolist()

    write_execution_summary(clean, diagnostic, tables / "results_execution_summary.tex")
    write_method_ranking(method, tables / "results_method_ranking.tex")
    write_factor_summary(outputs, tables / "results_factor_summary.tex")
    write_category_summary(outputs["category_method_summary"], method_order, tables / "appendix_category_method_summary.tex")

    endpoint_summary, endpoint_detail = compute_endpoint_profiles(clean)
    endpoint_summary.to_csv(tables / "performance_profile_summary.csv", index=False)
    endpoint_detail.to_csv(tables / "performance_profile_detail.csv", index=False)
    endpoint_plot_data = clean.copy()
    endpoint_plot_data["method"] = endpoint_plot_data["method_str"]
    legacy_profiles.plot_profiles(
        endpoint_plot_data,
        figures / "results_performance_profiles.pdf",
    )

    first_summary, first_detail = compute_first_passage(clean)
    first_summary.to_csv(tables / "first_passage_profile_summary.csv", index=False)
    first_detail.to_csv(tables / "first_passage_profile_detail.csv", index=False)
    plot_profiles(
        first_detail,
        "first_passage_ratio",
        figures / "results_first_passage_performance_profiles.pdf",
    )

    legacy_assets.plot_robustness_pareto_combined(
        method,
        output_path=figures / "results_robustness_pareto_combined.pdf",
        xlim=(30.0, 82.0),
        label_positions={
            "A2-Parallax-Mw": (42.0, 96.0, "left", 0.0),
            "A2-Parallax-Mc": (42.0, 90.0, "left", 0.0),
            "A0-SphInvRange-W": (50.0, 66.0, "left", 0.18),
            "A1-XYInvZ-Ac": (48.0, 48.0, "left", -0.18),
        },
    )
    legacy_assets.plot_category_heatmap(
        outputs["category_method_summary"],
        output_path=figures / "results_category_heatmap.pdf",
    )

    write_threading_tables(clean, diagnostic, previous_clean, args.problems_root, tables)
    write_key_numbers(clean, outputs, endpoint_summary, first_summary, tables)
    write_pairwise_statistics(clean, tables)
    recovery = compute_reference_recovery(diagnostic, args.diagnostic_root, args.problems_root, tables)
    write_reference_recovery_table(recovery, tables)
    reference_pairwise, reference_stats = reference_recovery_tests(recovery)
    write_reference_statistics_table(reference_pairwise, reference_stats, tables)
    legacy_assets.plot_reference_recovery(recovery)

    print(f"clean_rows={len(clean)} diagnostic_rows={len(diagnostic)}")
    print(method[["method", "within_1pct_rate", "within_5pct_rate", "best_rmse_count", "fastest_within_1pct_count", "median_solver_time_sec", "median_iterations", "max_iter_rate"]].to_string(index=False))
    print("\nEndpoint profile key rows")
    print(endpoint_summary[endpoint_summary["method"].astype(str).isin(["A2-Parallax-Mw", "A2-Parallax-Mc", "A0-SphInvRange-W"])].to_string(index=False))
    print("\nFirst-passage profile key rows")
    print(first_summary[first_summary["method"].astype(str).isin(["A2-Parallax-Mw", "A2-Parallax-Mc", "A0-SphInvRange-W"])].to_string(index=False))


if __name__ == "__main__":
    main()
