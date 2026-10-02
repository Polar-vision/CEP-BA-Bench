#!/usr/bin/env python3
"""Regenerate remaining manuscript experiment figures from single-thread outputs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_ORDER = [
    "A2-Parallax-Mw",
    "A2-Parallax-Mc",
    "A0-SphInvRange-W",
    "A1-XYInvZ-Ac",
    "A1-SphInvRange-Aw",
    "A1-SphInvRange-Ac",
    "A0-XYInvZ-W",
    "A1-XYZ-Aw",
    "A0-XYZ-W",
    "A1-XYZ-Ac",
    "A1-SphRange-Ac",
    "A1-XYInvZ-Aw",
    "A0-SphRange-W",
    "A1-SphRange-Aw",
]

METHOD_COLORS = {
    "A2-Parallax-Mw": "#0072B2",
    "A2-Parallax-Mc": "#CC79A7",
    "A0-SphInvRange-W": "#009E73",
    "A1-XYInvZ-Ac": "#E69F00",
    "A1-SphInvRange-Aw": "#66a61e",
    "A1-SphInvRange-Ac": "#1b9e77",
    "A0-XYInvZ-W": "#8c564b",
    "A1-XYZ-Aw": "#b0b0b0",
    "A0-XYZ-W": "#333333",
    "A1-XYZ-Ac": "#9aa0a6",
    "A1-SphRange-Ac": "#d95f02",
    "A1-XYInvZ-Aw": "#9467bd",
    "A0-SphRange-W": "#a6761d",
    "A1-SphRange-Aw": "#e6ab02",
}

CONVERGENCE_METHODS = [
    "A2-Parallax-Mw",
    "A2-Parallax-Mc",
    "A0-SphInvRange-W",
    "A1-XYInvZ-Ac",
    "A0-XYZ-W",
]

CONVERGENCE_LINESTYLES = {
    "A2-Parallax-Mw": "-",
    "A2-Parallax-Mc": "--",
    "A0-SphInvRange-W": "-",
    "A1-XYInvZ-Ac": "-.",
    "A0-XYZ-W": ":",
}

REPRESENTATIVE_CONVERGENCE_CASES = [
    ("Close-Range__CR12-problem-88-46484", "Close-Range / CR12"),
    ("Oblique-5__R3-problem-36-20326", "Oblique-5 / R3"),
    ("UAV__ODM13-problem-18-5509", "UAV / ODM13"),
    ("Vehicle__KD5-problem-150-70567", "Vehicle / KD5"),
]

BOUNDARY_CONVERGENCE_CASE = ("Vehicle__KD4-problem-161-61484", "Vehicle / KD4 boundary case")

D144_QUALITY_ORDER = [
    ("init-rmse002p00px", "Init 2"),
    ("init-rmse005p00px", "Init 5"),
    ("init-rmse010p00px", "Init 10"),
    ("init-rmse020p00px", "Init 20"),
    ("init-rmse050p00px", "Init 50"),
    ("init-rmse100p00px", "Init 100"),
    ("joint-rmse200p00px", "Joint 200"),
    ("joint-rmse500p00px", "Joint 500"),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=root / "benchmark_initial_value_diagnostic_strict_v2_20260920_singlethread_tasklog",
    )
    parser.add_argument(
        "--d144-accuracy-root",
        type=Path,
        default=root / "d144_diag_singlethread_confirm_20260926_230615" / "absolute_accuracy_singlethread",
    )
    parser.add_argument(
        "--d144-gcp-split",
        type=Path,
        default=root
        / "gcp_splits"
        / "BA-problem-000144-i2823-p188150-o1073331-g23-c0"
        / "gcp_13control_10checkpoint_uniform_xy_summary.csv",
    )
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        default=Path(r"E:\zuo\projects\documents\ISPRS\New_GSIS"),
    )
    return parser.parse_args()


def tex_escape(text: object) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value != 0.0 and (abs(value) >= 1.0e4 or abs(value) < 1.0e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def best_num(value: float, best: float, digits: int = 2) -> str:
    text = num(value, digits)
    if pd.notna(value) and np.isclose(float(value), float(best), rtol=1e-12, atol=1e-15):
        return rf"\textbf{{{text}}}"
    return text


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def accepted_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if "step_successful" in curve.columns:
        accepted = curve[curve["step_successful"].astype(int) == 1].copy()
    else:
        accepted = curve.copy()
    if accepted.empty:
        accepted = curve.head(1).copy()
    accepted = accepted.sort_values("iteration").reset_index(drop=True)
    accepted["accepted_step"] = np.arange(len(accepted))
    return accepted


def trial_count_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if "step_successful" not in curve.columns:
        return pd.DataFrame(columns=["accepted_step", "trial_count"])
    curve = curve.sort_values("iteration").reset_index(drop=True)
    records = []
    trial_count = 0
    accepted_step = 0
    for _, row in curve.iterrows():
        if int(row.get("iteration", 0)) == 0:
            continue
        trial_count += 1
        if int(row["step_successful"]) == 1:
            accepted_step += 1
            records.append({"accepted_step": accepted_step, "trial_count": trial_count})
            trial_count = 0
    return pd.DataFrame(records)


def retry_line_width(trial_count: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(trial_count, dtype=float)
    return 0.8 + 0.45 * np.minimum(values - 2.0, 4.0)


def compact_log_tick(value: float, _position: int) -> str:
    if value <= 0 or not np.isfinite(value):
        return ""
    if value >= 10:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
    if value >= 1:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
    return f"{value:.1g}"


def collect_diagnostic_summaries(diagnostic_root: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diag = pd.read_csv(diagnostic_root / "summary.csv")
    numeric = [
        "iterations",
        "rejected_steps",
        "final_lm_gain_ratio",
        "final_relative_step_size",
        "final_gradient_lipschitz_estimate",
        "final_direction_quality",
    ]
    for col in numeric:
        diag[col] = pd.to_numeric(diag[col], errors="coerce")
    diag_summary = (
        diag.groupby("method", observed=False)
        .agg(
            median_iterations=("iterations", "median"),
            median_rejected_steps=("rejected_steps", "median"),
            median_lm_gain_ratio=("final_lm_gain_ratio", "median"),
            median_relative_step_size=("final_relative_step_size", "median"),
            median_lipschitz=("final_gradient_lipschitz_estimate", "median"),
            median_direction_quality=("final_direction_quality", "median"),
        )
        .reset_index()
    )

    schur_rows = []
    for path in diagnostic_root.rglob("schur_summary.csv"):
        frame = pd.read_csv(path)
        frame["method"] = path.relative_to(diagnostic_root).parts[-2]
        schur_rows.append(frame)
    schur = pd.concat(schur_rows, ignore_index=True)
    for col in ["condition_number", "numerical_rank", "nullity"]:
        schur[col] = pd.to_numeric(schur[col], errors="coerce")
    schur_summary = (
        schur.groupby("method", observed=False)
        .agg(
            schur_samples=("method", "size"),
            median_schur_condition=("condition_number", "median"),
            q90_schur_condition=("condition_number", lambda values: values.quantile(0.9)),
            median_schur_nullity=("nullity", "median"),
        )
        .reset_index()
    )

    point_rows = []
    for path in diagnostic_root.rglob("point_block_conditioning.csv"):
        frame = pd.read_csv(path, usecols=["condition_number", "numerical_rank"])
        frame["method"] = path.relative_to(diagnostic_root).parts[-2]
        point_rows.append(frame)
    point = pd.concat(point_rows, ignore_index=True)
    point["condition_number"] = pd.to_numeric(point["condition_number"], errors="coerce")
    point["condition_number"] = point["condition_number"].replace([np.inf, -np.inf], np.nan)
    point["numerical_rank"] = pd.to_numeric(point["numerical_rank"], errors="coerce")
    point_summary = (
        point.groupby("method", observed=False)
        .agg(
            point_samples=("method", "size"),
            median_point_condition=("condition_number", "median"),
            q90_point_condition=("condition_number", lambda values: values.quantile(0.9)),
            point_rank_def_rate=("numerical_rank", lambda values: (values < 3).mean()),
        )
        .reset_index()
    )

    diagnostic = diag_summary.merge(point_summary, on="method").merge(schur_summary, on="method")
    diagnostic["method"] = pd.Categorical(diagnostic["method"], METHOD_ORDER, ordered=True)
    diagnostic = diagnostic.sort_values("method")
    diagnostic.to_csv(tables / "results_diagnostic_summary.csv", index=False)
    return diagnostic, point, schur


def write_diagnostic_table(diagnostic: pd.DataFrame, tables: Path) -> None:
    best_values = {
        "median_iterations": float(diagnostic["median_iterations"].min()),
        "median_rejected_steps": float(diagnostic["median_rejected_steps"].min()),
        "median_lm_gain_ratio": float(diagnostic["median_lm_gain_ratio"].max()),
    }
    rows = []
    for _, row in diagnostic.iterrows():
        rows.append(
            " & ".join(
                [
                    tex_escape(row["method"]),
                    best_num(row["median_iterations"], best_values["median_iterations"], 1),
                    best_num(row["median_rejected_steps"], best_values["median_rejected_steps"], 1),
                    best_num(row["median_lm_gain_ratio"], best_values["median_lm_gain_ratio"], 3),
                    num(row["median_point_condition"], 2),
                    num(row["q90_point_condition"], 2),
                    num(row["median_schur_condition"], 2),
                ]
            )
            + r" \\"
        )
    write(
        tables / "results_diagnostic_summary.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Single-thread strict diagnostic summary. Point-block and Schur-complement condition numbers are sampled at the final state and computed without LM damping.}}
\label{{tab:results_diagnostic_summary}}
\footnotesize
\setlength{{\tabcolsep}}{{2.6pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Method}} & Med. & Rejected & LM & Med. & Q90 & Med. \\
 & it. & steps & gain & $\kappa(C_k)$ & $\kappa(C_k)$ & $\kappa^+(S)$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Bold values mark the best method only for optimizer-effort and trust-region reliability columns: median iterations, rejected steps, and LM gain. Lower values are better for median iterations, rejected steps, and condition numbers; higher values are better for LM gain. Condition numbers are undamped structural diagnostics, not the damped LM systems used for individual trial steps, and are not bolded.
\end{{minipage}}
\end{{table*}}
""",
    )


def plot_conditioning(point: pd.DataFrame, schur: pd.DataFrame, figures: Path) -> None:
    point = point.copy()
    point["method"] = pd.Categorical(point["method"], METHOD_ORDER, ordered=True)
    schur = schur.copy()
    schur["method"] = pd.Categorical(schur["method"], METHOD_ORDER, ordered=True)

    method_order = (
        point.dropna(subset=["condition_number"])
        .groupby("method", observed=False)["condition_number"]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    fig = plt.figure(figsize=(7.2, 3.75))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.13)
    ax_point = fig.add_subplot(grid[0, 0])
    ax_schur = fig.add_subplot(grid[0, 1])

    sns.boxplot(
        data=point.dropna(subset=["condition_number"]),
        x="condition_number",
        y="method",
        order=method_order,
        ax=ax_point,
        fliersize=0.45,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(method, "#777777") for method in method_order],
        orient="h",
    )
    ax_point.set_xscale("log")
    ax_point.set_title(r"(a) Point block $\kappa(C_k)$", loc="left", fontsize=10, fontweight="bold")
    ax_point.set_xlabel(r"Undamped condition number $\kappa(C_k)$", fontsize=9.3)
    ax_point.set_ylabel("")
    ax_point.tick_params(axis="x", labelsize=8.3)
    ax_point.tick_params(axis="y", labelsize=7.8, pad=1.5)
    ax_point.grid(axis="x", alpha=0.25)

    sns.boxplot(
        data=schur.dropna(subset=["condition_number"]),
        x="condition_number",
        y="method",
        order=method_order,
        ax=ax_schur,
        fliersize=0.65,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(method, "#777777") for method in method_order],
        orient="h",
    )
    ax_schur.set_xscale("log")
    ax_schur.set_title(r"(b) Schur complement $\kappa^{+}(S)$", loc="left", fontsize=10, fontweight="bold")
    ax_schur.set_xlabel(r"Undamped condition number $\kappa^{+}(S)$", fontsize=9.3)
    ax_schur.set_ylabel("")
    ax_schur.set_yticklabels([])
    ax_schur.tick_params(axis="x", labelsize=8.3)
    ax_schur.tick_params(axis="y", length=0)
    ax_schur.grid(axis="x", alpha=0.25)

    fig.subplots_adjust(left=0.255, right=0.985, bottom=0.17, top=0.90)
    fig.savefig(figures / "results_second_order_combined.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_appendix_convergence_case_table(diagnostic_root: Path, tables: Path) -> None:
    diag = pd.read_csv(diagnostic_root / "summary.csv")
    cases = REPRESENTATIVE_CONVERGENCE_CASES + [BOUNDARY_CONVERGENCE_CASE]
    case_order = {dataset: i for i, (dataset, _) in enumerate(cases)}
    method_order = {method: i for i, method in enumerate(CONVERGENCE_METHODS)}
    diag = diag[diag["base_dataset"].isin(case_order) & diag["method"].isin(CONVERGENCE_METHODS)].copy()
    diag["case_order"] = diag["base_dataset"].map(case_order)
    diag["method_order"] = diag["method"].map(method_order)
    diag = diag.sort_values(["case_order", "method_order"])
    labels = dict(cases)
    rows = []
    for dataset, _title in cases:
        block = diag[diag["base_dataset"] == dataset].copy()
        if block.empty:
            continue
        best_final = float(block["final_rmse_px"].min())
        best_iterations = float(block["iterations"].min())
        best_rejected = float(block["rejected_steps"].min())

        def mark_int(value: float, best: float) -> str:
            text = f"{int(value)}"
            return rf"\textbf{{{text}}}" if np.isclose(float(value), best) else text

        def mark_display_num(value: float, best: float, digits: int = 3) -> str:
            text = num(value, digits)
            return rf"\textbf{{{text}}}" if text == num(best, digits) else text

        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{tex_escape(labels[dataset])}}}}} \\")
        for _, row in block.iterrows():
            rows.append(
                " & ".join(
                    [
                        "",
                        tex_escape(row["method"]),
                        num(row["initial_rmse_px"], 3),
                        mark_display_num(row["final_rmse_px"], best_final, 3),
                        mark_int(row["iterations"], best_iterations),
                        mark_int(row["rejected_steps"], best_rejected),
                        num(row["final_lm_gain_ratio"], 3),
                    ]
                )
                + r" \\"
            )
    write(
        tables / "appendix_convergence_case_summary.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Single-thread numerical summaries for the convergence-curve cases.}}
\label{{tab:appendix_convergence_case_summary}}
\footnotesize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Case}} & \multirow{{2}}{{*}}{{Method}} & Initial & Final & \multirow{{2}}{{*}}{{Iter.}} & \multirow{{2}}{{*}}{{Rejected}} & Final LM \\
 & & RMSE & RMSE & & & gain \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Bold values mark the best entry within each case block for final RMSE, iteration count, and rejected steps. Lower values are better for these three columns. Initial RMSE is common input information, and final LM gain is reported as a diagnostic value.
\end{{minipage}}
\end{{table*}}
""",
    )


def plot_convergence_cases(diagnostic_root: Path, figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.25), sharex=False, sharey=False)
    for index, (ax, (dataset, title)) in enumerate(zip(axes.ravel(), REPRESENTATIVE_CONVERGENCE_CASES)):
        for method in CONVERGENCE_METHODS:
            path = diagnostic_root / dataset / "Initial Value" / method / "convergence_strict.csv"
            if not path.exists():
                continue
            curve = accepted_curve(pd.read_csv(path))
            ax.plot(
                curve["accepted_step"],
                curve["rmse_px"],
                label=method,
                color=METHOD_COLORS.get(method, None),
                linestyle=CONVERGENCE_LINESTYLES.get(method, "-"),
                linewidth=1.75,
            )
        row, col = divmod(index, 2)
        ax.set_title(title, fontsize=10.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(compact_log_tick))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("Accepted LM step" if row == 1 else "", fontsize=10.0)
        ax.set_ylabel("RMSE (px)" if col == 0 else "", fontsize=10.0)
        ax.tick_params(axis="both", labelsize=9.0)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.6, frameon=False)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.19, top=0.93, wspace=0.25, hspace=0.38)
    fig.savefig(figures / "results_representative_convergence_grid.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    dataset, title = BOUNDARY_CONVERGENCE_CASE
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for method in CONVERGENCE_METHODS:
        path = diagnostic_root / dataset / "Initial Value" / method / "convergence_strict.csv"
        if not path.exists():
            continue
        curve = accepted_curve(pd.read_csv(path))
        ax.plot(
            curve["accepted_step"],
            curve["rmse_px"],
            label=method,
            color=METHOD_COLORS.get(method, None),
            linestyle=CONVERGENCE_LINESTYLES.get(method, "-"),
            linewidth=1.9,
        )
    ax.set_title(title, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Accepted LM step")
    ax.set_ylabel("Reprojection RMSE (px, log scale)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "appendix_convergence_kd4.pdf")
    plt.close(fig)


def plot_convergence_step_counts(diagnostic_root: Path, figures: Path) -> None:
    cases = REPRESENTATIVE_CONVERGENCE_CASES + [BOUNDARY_CONVERGENCE_CASE]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.9), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    method_positions = {method: i for i, method in enumerate(CONVERGENCE_METHODS)}
    method_labels = ["Parallax-Mw", "Parallax-Mc", "SphInvRange-W", "XYInvZ-Ac", "XYZ-W"]
    for index, (ax, (dataset, title)) in enumerate(zip(axes_flat, cases)):
        for method in CONVERGENCE_METHODS:
            path = diagnostic_root / dataset / "Initial Value" / method / "convergence_strict.csv"
            if not path.exists():
                continue
            curve = trial_count_curve(pd.read_csv(path))
            if curve.empty:
                continue
            method_y = method_positions[method]
            method_color = METHOD_COLORS.get(method, None)
            final_step = float(curve["accepted_step"].max())
            ax.scatter(
                [final_step],
                [method_y],
                s=52,
                marker=">",
                facecolors="none",
                edgecolors=method_color,
                linewidth=1.15,
                zorder=4,
            )
            retry = curve[curve["trial_count"] > 1].copy()
            if not retry.empty:
                ax.vlines(
                    retry["accepted_step"],
                    method_y - 0.32,
                    method_y + 0.32,
                    colors=method_color,
                    linewidths=retry_line_width(retry["trial_count"]),
                    alpha=0.95,
                    zorder=3,
                )
        row, col = divmod(index, 2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Accepted LM step" if row == 2 else "", fontsize=8.8)
        ax.set_yticks(range(len(CONVERGENCE_METHODS)))
        ax.set_yticklabels(method_labels if col == 0 else [], fontsize=7.7)
        ax.set_ylim(len(CONVERGENCE_METHODS) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=8.0)
        ax.grid(alpha=0.25)
        ax.grid(axis="y", alpha=0.16)
    for ax in axes_flat[len(cases) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.955, wspace=0.24, hspace=0.42)
    fig.savefig(figures / "appendix_convergence_step_counts.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_lm_damping_paths(diagnostic_root: Path, figures: Path) -> None:
    cases = REPRESENTATIVE_CONVERGENCE_CASES + [BOUNDARY_CONVERGENCE_CASE]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.9), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    legend_handles: dict[str, Line2D] = {}
    for index, (ax, (dataset, title)) in enumerate(zip(axes_flat, cases)):
        for method in CONVERGENCE_METHODS:
            path = diagnostic_root / dataset / "Initial Value" / method / "convergence_strict.csv"
            if not path.exists():
                continue
            curve = pd.read_csv(path)
            working = curve.copy()
            working["_iteration"] = pd.to_numeric(working["iteration"], errors="coerce")
            working["_lm_damping"] = pd.to_numeric(working["lm_damping"], errors="coerce")
            working = working[
                np.isfinite(working["_iteration"])
                & np.isfinite(working["_lm_damping"])
                & (working["_lm_damping"] > 0)
            ].sort_values("_iteration")
            if working.empty:
                continue
            (line,) = ax.plot(
                working["_iteration"],
                working["_lm_damping"],
                label=method,
                color=METHOD_COLORS.get(method, None),
                linestyle=CONVERGENCE_LINESTYLES.get(method, "-"),
                linewidth=1.45,
                alpha=0.95,
            )
            legend_handles.setdefault(method, line)
        row, col = divmod(index, 2)
        ax.set_title(title, fontsize=9.5)
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(1e-13, 1e5)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("LM trial iteration" if row == 2 else "", fontsize=8.8)
        ax.set_ylabel(r"LM damping $\lambda$" if col == 0 else "", fontsize=8.8)
        ax.tick_params(axis="both", labelsize=8.0)
        ax.grid(alpha=0.25)
    legend_axes = axes_flat[len(cases) :]
    if len(legend_axes) > 0:
        legend_axes[0].axis("off")
        handles = [legend_handles[method] for method in CONVERGENCE_METHODS if method in legend_handles]
        legend_axes[0].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc="center",
            title="Method",
            fontsize=8.3,
            title_fontsize=8.8,
            frameon=False,
            labelspacing=0.9,
        )
    for ax in legend_axes[1:]:
        ax.axis("off")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.955, wspace=0.24, hspace=0.42)
    fig.savefig(figures / "appendix_lm_damping_paths.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def d144_quality_label(quality_dataset: str) -> str:
    for token, label in D144_QUALITY_ORDER:
        if token in quality_dataset:
            return label
    return quality_dataset


def write_d144_checkpoint_table(d144_accuracy_root: Path, tables: Path) -> None:
    data = pd.read_csv(d144_accuracy_root / "checkpoint_absolute_accuracy_by_dataset_method.csv")
    chosen = [("init-rmse002p00px", "Initial RMSE: 2.04 px"), ("init-rmse050p00px", "Initial RMSE: 50.57 px")]
    blocks = []
    for token, _label in chosen:
        block = data[data["quality_dataset"].astype(str).str.contains(token)].copy()
        block = block.set_index("method").reindex(METHOD_ORDER)
        blocks.append(block)
    best = []
    for block in blocks:
        best.append(
            {
                "rmse_horizontal": float(block["rmse_horizontal"].min()),
                "rmse_z": float(block["rmse_z"].min()),
                "rmse_3d": float(block["rmse_3d"].min()),
            }
        )

    def cell(value: float, best_value: float) -> str:
        text = f"{float(value):.4f}"
        return rf"\textbf{{{text}}}" if np.isclose(float(value), best_value, rtol=1e-12, atol=1e-12) else text

    rows = []
    for method in METHOD_ORDER:
        left = blocks[0].loc[method]
        right = blocks[1].loc[method]
        rows.append(
            " & ".join(
                [
                    tex_escape(method),
                    cell(left["rmse_horizontal"], best[0]["rmse_horizontal"]),
                    cell(left["rmse_z"], best[0]["rmse_z"]),
                    cell(left["rmse_3d"], best[0]["rmse_3d"]),
                    cell(right["rmse_horizontal"], best[1]["rmse_horizontal"]),
                    cell(right["rmse_z"], best[1]["rmse_z"]),
                    cell(right["rmse_3d"], best[1]["rmse_3d"]),
                ]
            )
            + r" \\"
        )
    write(
        tables / "results_d144_checkpoint_accuracy.tex",
        rf"""\begin{{table}}[htbp]
\centering
\caption{{\textcolor{{revisionblue}}{{Single-thread independent check-point accuracy on two representative settings of the controlled D144 quality experiment. The full experiment contains eight quality sets.}} Thirteen independently measured ground control points define the metric datum and enter the adjustment; ten independently measured check points are withheld from optimization. Values are RMSE in m, obtained by forward intersection from final camera poses and held-out check-point observations.}}
\label{{tab:results_d144_checkpoint_accuracy}}
\footnotesize
\setlength{{\tabcolsep}}{{4.0pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
& \multicolumn{{3}}{{c}}{{Initial RMSE: 2.04 px}} & \multicolumn{{3}}{{c}}{{Initial RMSE: 50.57 px}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Method & $\mathrm{{RMSE}}_{{XY}}$ & $\mathrm{{RMSE}}_{{Z}}$ & $\mathrm{{RMSE}}_{{3D}}$ & $\mathrm{{RMSE}}_{{XY}}$ & $\mathrm{{RMSE}}_{{Z}}$ & $\mathrm{{RMSE}}_{{3D}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize \textcolor{{revisionblue}}{{The displayed 2.04 px and 50.57 px settings correspond to the D144 \texttt{{init-rmse002p00px}} and \texttt{{init-rmse050p00px}} quality sets, respectively; the sensitivity figure includes all eight quality sets.}} Each setting uses 124 held-out check-point observations in total. Bold values mark the smallest RMSE in each displayed column. This single-scene controlled experiment is reported separately from the 105-problem Initial Value ranking.
\end{{minipage}}
\end{{table}}
""",
    )


def plot_d144_checkpoint_heatmap(d144_accuracy_root: Path, figures: Path) -> None:
    accuracy = pd.read_csv(d144_accuracy_root / "checkpoint_absolute_accuracy_by_dataset_method.csv")
    accuracy["quality_label"] = accuracy["quality_dataset"].map(d144_quality_label)
    labels = [label for _, label in D144_QUALITY_ORDER]
    accuracy["quality_label"] = pd.Categorical(accuracy["quality_label"], labels, ordered=True)
    accuracy["rmse_3d"] = pd.to_numeric(accuracy["rmse_3d"], errors="coerce")
    accuracy["excess_mm"] = (
        accuracy["rmse_3d"] - accuracy.groupby("quality_label", observed=False)["rmse_3d"].transform("min")
    ) * 1000.0
    method_order = (
        accuracy.groupby("method", observed=False)["rmse_3d"].mean().sort_values().index.astype(str).tolist()
    )
    pivot = accuracy.pivot(index="method", columns="quality_label", values="excess_mm")
    heatmap_values = np.log10(1.0 + pivot.loc[method_order, labels])

    fig, ax_heat = plt.subplots(figsize=(7.2, 4.15))
    heat = sns.heatmap(
        heatmap_values,
        ax=ax_heat,
        cmap="YlOrRd",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"fraction": 0.065, "pad": 0.025},
    )
    ax_heat.set_xlabel("Initialization setting", fontsize=13.0)
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", labelrotation=28, labelsize=11.2)
    ax_heat.tick_params(axis="y", labelsize=10.8, pad=1.0)
    colorbar = heat.collections[0].colorbar
    color_ticks_mm = np.array([0, 1, 3, 10, 30, 60], dtype=float)
    colorbar.set_ticks(np.log10(1.0 + color_ticks_mm))
    colorbar.set_ticklabels(["0", "1", "3", "10", "30", "60"])
    colorbar.ax.tick_params(labelsize=10.2)
    colorbar.ax.set_title("mm", fontsize=10.8, pad=4)
    fig.subplots_adjust(left=0.255, right=0.955, bottom=0.22, top=0.985)
    fig.savefig(figures / "results_d144_checkpoint_rmse_excess_panel.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_d144_control_layout_panel(gcp_split: Path, figures: Path) -> None:
    gcp = pd.read_csv(gcp_split)
    control = gcp[gcp["role"].astype(str).str.lower() == "control"].copy()
    check = gcp[gcp["role"].astype(str).str.lower() == "checkpoint"].copy()
    fig, ax = plt.subplots(figsize=(4.7, 4.15))
    ax.scatter(
        control["x"],
        control["y"],
        s=72,
        marker="o",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.7,
        label="GCP",
        zorder=3,
    )
    ax.scatter(
        check["x"],
        check["y"],
        s=90,
        marker="^",
        color="#D55E00",
        edgecolor="white",
        linewidth=0.7,
        label="Check point",
        zorder=3,
    )
    label_offsets = {
        "0001": (6, -8),
        "0005": (5, 3),
        "0009": (-8, 7),
        "0010": (-12, 4),
        "0013": (6, 5),
        "0014": (6, 5),
        "0016": (6, 5),
        "0017": (6, 5),
        "0018": (6, -5),
        "0019": (7, 7),
        "0020": (6, 6),
        "0021": (-7, -6),
    }
    for _, row in gcp.iterrows():
        label = str(row["name"]).replace("GCP", "")
        offset = label_offsets.get(label, (3.0, 3.0))
        ax.annotate(
            label,
            (row["x"], row["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.4,
            color="0.25",
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
        )
    ax.set_xlabel("World X (km)", fontsize=13.3)
    ax.set_ylabel("World Y (km)", fontsize=13.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value / 1000:.1f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value / 1000:.1f}"))
    ax.set_aspect(1.10, adjustable="box")
    ax.tick_params(axis="both", labelsize=11.6)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=11.0)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.12, top=0.985)
    fig.savefig(figures / "results_d144_control_layout_panel.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables = args.manuscript_root / "tables"
    figures = args.manuscript_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    project_root = Path(__file__).resolve().parents[2]
    legacy_assets = load_module(
        "legacy_results_assets",
        project_root / "documents" / "ISPRS" / "New" / "generate_results_assets.py",
    )
    legacy_assets.DIAG = args.diagnostic_root
    legacy_assets.D144_ABSOLUTE = args.d144_accuracy_root
    legacy_assets.D144_GCP_SPLIT = args.d144_gcp_split
    legacy_assets.FIG = figures
    legacy_assets.TAB = tables

    diagnostic, point, schur = collect_diagnostic_summaries(args.diagnostic_root, tables)
    write_diagnostic_table(diagnostic, tables)
    legacy_assets.plot_conditioning(point, schur)
    write_appendix_convergence_case_table(args.diagnostic_root, tables)
    legacy_assets.plot_convergence_cases()
    legacy_assets.plot_lm_damping_paths()
    write_d144_checkpoint_table(args.d144_accuracy_root, tables)
    legacy_assets.plot_d144_control_layout_panel()
    legacy_assets.plot_d144_checkpoint_heatmap_panel()
    print("Regenerated single-thread experiment figures/tables:")
    print(f"  {tables}")
    print(f"  {figures}")


if __name__ == "__main__":
    main()
