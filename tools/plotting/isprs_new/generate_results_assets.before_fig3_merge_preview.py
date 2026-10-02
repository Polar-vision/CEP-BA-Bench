from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


BASE = Path(__file__).resolve().parent
CEP = Path(r"E:\zuo\projects\CEP")
CLEAN = CEP / "benchmark_initial_value_clean"
DIAG = CEP / "benchmark_initial_value_diagnostic_strict_v2"
FIG = BASE / "figures"
TAB = BASE / "tables"

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
    "A2-Parallax-Mw": "#1f77b4",
    "A2-Parallax-Mc": "#4c78a8",
    "A0-SphInvRange-W": "#59a14f",
    "A1-XYInvZ-Ac": "#f28e2b",
    "A1-SphInvRange-Aw": "#8cd17d",
    "A1-SphInvRange-Ac": "#86bc86",
    "A0-XYInvZ-W": "#ffbe7d",
    "A1-XYZ-Aw": "#b07aa1",
    "A0-XYZ-W": "#9c755f",
    "A1-XYZ-Ac": "#d4a6c8",
    "A1-SphRange-Ac": "#e15759",
    "A1-XYInvZ-Aw": "#edc948",
    "A0-SphRange-W": "#ff9da7",
    "A1-SphRange-Aw": "#bab0ac",
}

CONVERGENCE_METHODS = [
    "A2-Parallax-Mw",
    "A2-Parallax-Mc",
    "A0-SphInvRange-W",
    "A1-XYInvZ-Ac",
    "A0-XYZ-W",
]

CONVERGENCE_COLORS = {
    "A2-Parallax-Mw": "#0072B2",
    "A2-Parallax-Mc": "#CC79A7",
    "A0-SphInvRange-W": "#009E73",
    "A1-XYInvZ-Ac": "#E69F00",
    "A0-XYZ-W": "#333333",
}

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

D144_ABSOLUTE = CEP / "d144_diag" / "absolute_accuracy"
D144_GCP_SPLIT = (
    CEP
    / "gcp_splits"
    / "BA-problem-000144-i2823-p188150-o1073331-g23-c0"
    / "gcp_13control_10checkpoint_uniform_xy_summary.csv"
)
D144_INPUT_ROOT = (
    CEP
    / "PVL-BA-Bench"
    / "public_release"
    / "extracted"
    / "pvl-ba"
    / "BA-problem-000144-i2823-p188150-o1073331-g23-c0"
)

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


def accepted_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Return the solution trajectory after accepted LM steps only."""
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
    """Count LM trial attempts required before each accepted update."""
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
            records.append(
                {
                    "accepted_step": accepted_step,
                    "trial_count": trial_count,
                }
            )
            trial_count = 0
    return pd.DataFrame(records)


def retry_line_width(trial_count: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(trial_count, dtype=float)
    return 0.8 + 0.45 * np.minimum(values - 2.0, 4.0)


def compact_log_tick(value: float, _position: int) -> str:
    if value <= 0 or not np.isfinite(value):
        return ""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
    if value >= 1:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
    return f"{value:.1g}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}"


def num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value != 0 and (abs(value) >= 1e4 or abs(value) < 1e-2):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def best_num(value: float, best: float, digits: int = 2) -> str:
    text = num(value, digits)
    if pd.notna(value) and np.isclose(float(value), float(best), rtol=1e-12, atol=1e-15):
        return rf"\textbf{{{text}}}"
    return text


def best_pct(value: float, best: float, digits: int = 1) -> str:
    text = pct(value, digits)
    if pd.notna(value) and np.isclose(float(value), float(best), rtol=1e-12, atol=1e-15):
        return rf"\textbf{{{text}}}"
    return text


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


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
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.dot(singular, sign) / max(variance, 1e-30))
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation


def similarity_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    aligned, scale, _ = similarity_transform(source, target)
    return aligned, scale


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


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table_env(label: str, caption: str, colspec: str, header: str, rows: list[str]) -> str:
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\footnotesize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{{colspec}}}
\toprule
{header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
\end{{table*}}
"""


def longtable_env(label: str, caption: str, colspec: str, header: str, rows: list[str]) -> str:
    body = "\n".join(rows)
    return rf"""\begingroup
\footnotesize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{longtable}}{{{colspec}}}
\caption{{{caption}}}
\label{{{label}}}\\
\toprule
{header} \\
\midrule
\endfirsthead
\caption[]{{{caption} (continued)}}\\
\toprule
{header} \\
\midrule
\endhead
\multicolumn{{{len(colspec)}}}{{r}}{{Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
\endgroup
"""


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(CLEAN / "analysis" / "summary_enriched_new.csv")
    method = pd.read_csv(CLEAN / "analysis" / "method_summary_new.csv")
    diag = pd.read_csv(DIAG / "summary.csv")
    for frame in (clean, method, diag):
        if "method" in frame.columns:
            frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    return clean, method, diag


def write_execution_table(clean: pd.DataFrame, diag: pd.DataFrame) -> None:
    categories = clean["category"].value_counts().sort_index()
    category_text = ", ".join(f"{name}: {count // 14}" for name, count in categories.items())
    rows = [
        rf"Base BA problems & {clean['base_dataset'].nunique()} \\",
        rf"Parameterizations & {clean['method'].nunique()} \\",
        rf"Clean runs & {len(clean)} ({int((clean['status'] == 'ok').sum())} usable) \\",
        rf"Strict diagnostic runs & {len(diag)} ({int((diag['status'] == 'ok').sum())} usable) \\",
        rf"Dataset categories & {tex_escape(category_text)} \\",
        rf"Clean Ceres convergence rate & {pct((clean['termination_type'].astype(int) == 0).mean())}\% \\",
        rf"Clean maximum-iteration rate & {pct(clean['max_iter'].astype(bool).mean())}\% \\",
        rf"Clean total solver time & {clean['solver_time_sec'].sum() / 3600.0:.2f} h \\",
    ]
    write(
        TAB / "results_execution_summary.tex",
        rf"""\begin{{table}}[t]
\centering
\caption{{Execution summary for the Initial Value benchmark tier.}}
\label{{tab:results_execution_summary}}
\footnotesize
\setlength{{\tabcolsep}}{{4.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{lr}}
\toprule
Quantity & Value \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""",
    )


def write_method_table(method: pd.DataFrame) -> None:
    method = method.sort_values("within_1pct_rate", ascending=False)
    best_values = {
        "within_1pct_rate": float(method["within_1pct_rate"].max()),
        "within_5pct_rate": float(method["within_5pct_rate"].max()),
        "median_rmse_gap_pct": float(method["median_rmse_gap_pct"].min()),
        "q90_rmse_gap_pct": float(method["q90_rmse_gap_pct"].min()),
        "best_rmse_count": float(method["best_rmse_count"].max()),
        "fastest_within_1pct_count": float(method["fastest_within_1pct_count"].max()),
        "median_solver_time_sec": float(method["median_solver_time_sec"].min()),
        "median_iterations": float(method["median_iterations"].min()),
        "max_iter_rate": float(method["max_iter_rate"].min()),
    }

    def mark(text: str, value: float, best: float) -> str:
        if pd.notna(value) and np.isclose(float(value), best, rtol=1e-12, atol=1e-15):
            return rf"\textbf{{{text}}}"
        return text

    rows = []
    for _, row in method.iterrows():
        rows.append(
            " & ".join(
                [
                    tex_escape(row["method"]),
                    mark(pct(row["within_1pct_rate"]), row["within_1pct_rate"], best_values["within_1pct_rate"]),
                    mark(pct(row["within_5pct_rate"]), row["within_5pct_rate"], best_values["within_5pct_rate"]),
                    mark(num(row["median_rmse_gap_pct"], 1), row["median_rmse_gap_pct"], best_values["median_rmse_gap_pct"]),
                    mark(num(row["q90_rmse_gap_pct"], 1), row["q90_rmse_gap_pct"], best_values["q90_rmse_gap_pct"]),
                    mark(f"{int(row['best_rmse_count'])}", row["best_rmse_count"], best_values["best_rmse_count"]),
                    mark(
                        f"{int(row['fastest_within_1pct_count'])}",
                        row["fastest_within_1pct_count"],
                        best_values["fastest_within_1pct_count"],
                    ),
                    mark(
                        num(row["median_solver_time_sec"], 2),
                        row["median_solver_time_sec"],
                        best_values["median_solver_time_sec"],
                    ),
                    mark(num(row["median_iterations"], 1), row["median_iterations"], best_values["median_iterations"]),
                    mark(pct(row["max_iter_rate"]), row["max_iter_rate"], best_values["max_iter_rate"]),
                ]
            )
            + r" \\"
        )
    write(
        TAB / "results_method_ranking.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Method ranking on matched Initial Value runs. The RMSE gap is measured relative to the best final reprojection RMSE within each dataset.}}
\label{{tab:results_method_ranking}}
\footnotesize
\setlength{{\tabcolsep}}{{2.4pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lrrrrrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Method}} & Within & Within & Med. gap & Q90 gap & \multirow{{2}}{{*}}{{Best}} & Fastest & Med. time & Med. & Max-it. \\
 & 1\% & 5\% & (\%) & (\%) & & good & (s) & it. & (\%) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Bold values mark the best method in each metric column. Higher values are better for Within 1\%, Within 5\%, Best, and Fastest good; lower values are better for RMSE gaps, median time, median iterations, and Max-it.
\end{{minipage}}
\end{{table*}}
""",
    )


def compute_reference_recovery(force: bool = False) -> pd.DataFrame:
    output = TAB / "results_reference_recovery.csv"
    if output.exists() and not force:
        cached = pd.read_csv(output)
        if "camera_rotation_median_deg" in cached.columns:
            return cached

    summary = pd.read_csv(DIAG / "summary.csv")
    rows = []
    for base_dataset, group in summary.groupby("base_dataset", sort=True):
        original = CEP / "benchmark_initial_value_problems" / base_dataset / "original"
        reference_points = np.loadtxt(original / "XYZ.txt", dtype=np.float64)
        reference_pose = np.loadtxt(find_camera_file(original), dtype=np.float64)
        reference_euler = reference_pose[:, :3]
        reference_cameras = reference_pose[:, 3:6]

        quality = CEP / "benchmark_initial_value_problems" / base_dataset / "quality" / "Initial Value"
        initial_points = np.loadtxt(quality / "XYZ.txt", dtype=np.float64)
        initial_pose = np.loadtxt(find_camera_file(quality), dtype=np.float64)
        initial_euler = initial_pose[:, :3]
        initial_cameras = initial_pose[:, 3:6]
        initial_source = np.vstack([initial_cameras, initial_points])
        initial_target = np.vstack([reference_cameras, reference_points])
        initial_aligned, initial_scale, initial_alignment_rotation = similarity_transform(initial_source, initial_target)
        initial_camera_errors = np.linalg.norm(
            initial_aligned[: len(reference_cameras)] - reference_cameras, axis=1
        )
        initial_point_errors = np.linalg.norm(
            initial_aligned[len(reference_cameras) :] - reference_points, axis=1
        )
        initial_point_rmse = float(np.sqrt(np.mean(initial_point_errors**2)))
        initial_camera_rmse = float(np.sqrt(np.mean(initial_camera_errors**2)))
        initial_rotation_errors = camera_rotation_errors_deg(
            initial_euler, reference_euler, initial_alignment_rotation
        )
        initial_camera_rotation_median = float(np.median(initial_rotation_errors))

        for record in group.itertuples(index=False):
            run_root = DIAG / record.base_dataset / record.quality_dataset / str(record.method)
            pose_path = run_root / "FinalPose.txt"
            point_path = run_root / "Final3D.ply"
            if not pose_path.exists() or not point_path.exists():
                continue
            estimated_pose = np.loadtxt(pose_path, dtype=np.float64)
            estimated_euler = estimated_pose[:, :3]
            estimated_cameras = estimated_pose[:, 3:6]
            estimated_points = read_ply_vertices(point_path)
            if len(estimated_points) != len(reference_points) or len(estimated_cameras) != len(reference_cameras):
                raise ValueError(f"Size mismatch for {record.base_dataset} / {record.method}")
            source = np.vstack([estimated_cameras, estimated_points])
            target = np.vstack([reference_cameras, reference_points])
            aligned, scale, alignment_rotation = similarity_transform(source, target)
            camera_aligned = aligned[: len(reference_cameras)]
            point_aligned = aligned[len(reference_cameras) :]
            camera_error = np.linalg.norm(camera_aligned - reference_cameras, axis=1)
            point_error = np.linalg.norm(point_aligned - reference_points, axis=1)
            rotation_error = camera_rotation_errors_deg(
                estimated_euler, reference_euler, alignment_rotation
            )
            point_rmse = float(np.sqrt(np.mean(point_error**2)))
            camera_rmse = float(np.sqrt(np.mean(camera_error**2)))
            rows.append(
                {
                    "base_dataset": record.base_dataset,
                    "quality_dataset": record.quality_dataset,
                    "method": str(record.method),
                    "category": str(record.base_dataset).split("__", 1)[0],
                    "cameras": int(record.cameras),
                    "points": int(record.points),
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
    recovery.to_csv(output, index=False)
    return recovery


def write_reference_recovery_table(recovery: pd.DataFrame) -> None:
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
    summary.to_csv(TAB / "results_reference_recovery_summary.csv", index=False)
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
                    tex_escape(row["method"]),
                    f"{int(row['n'])}",
                    best_num(row["median_point_rmse"], best_values["median_point_rmse"], 3),
                    best_num(row["q75_point_rmse"], best_values["q75_point_rmse"], 3),
                    best_num(row["median_point_ratio"], best_values["median_point_ratio"], 3),
                    best_num(row["median_point_error"], best_values["median_point_error"], 3),
                    best_num(row["median_camera_rmse"], best_values["median_camera_rmse"], 3),
                    best_num(row["median_camera_rotation"], best_values["median_camera_rotation"], 3),
                ]
            )
            + r" \\"
        )
    write(
        TAB / "results_reference_recovery.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Original-reference reconstruction recovery after Sim(3) alignment to the BA Datasets reference reconstruction.}}
\label{{tab:results_reference_recovery}}
\footnotesize
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Method}} & \multirow{{2}}{{*}}{{Runs}} & Med. Pt. & Q75 Pt. & Rec. & Med. Pt. & Cam. Ctr. & Cam. Rot. \\
 & & RMSE & RMSE & ratio & err. & RMSE & (deg) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Lower values are better for all reported metrics; bold values mark the best method in each metric column. Med. Pt. RMSE is the median dataset-level object-point RMSE; Q75 Pt. RMSE is its 75th percentile across datasets; Rec. ratio is final/initial aligned point RMSE; Med. Pt. err. is the median per-point Euclidean error; Cam. Ctr. is camera-center RMSE; Cam. Rot. is the median camera-rotation error after Sim(3) alignment.
\end{{minipage}}
\end{{table*}}
""",
    )


def reference_recovery_tests(recovery: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    pivot = recovery.pivot(index="base_dataset", columns="method", values="point_rmse")
    pivot = pivot[METHOD_ORDER]
    friedman = stats.friedmanchisquare(*[pivot[m].to_numpy() for m in METHOD_ORDER])
    kendall_w = friedman.statistic / (pivot.shape[0] * (pivot.shape[1] - 1))
    pairs = [
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
    pvals = []
    for a, b in pairs:
        av = pivot[a].to_numpy(float)
        bv = pivot[b].to_numpy(float)
        log_ratio = np.log(av / bv)
        test = stats.wilcoxon(log_ratio, zero_method="wilcox", alternative="two-sided")
        pvals.append(float(test.pvalue))
        d = np.log(bv / av)  # positive means a has smaller reference RMSE.
        nonzero = np.abs(d) > 1e-15
        ranks = stats.rankdata(np.abs(d[nonzero]))
        rbc = float(np.sum(np.sign(d[nonzero]) * ranks) / np.sum(ranks))
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "a_better": int(np.sum(av < bv)),
                "b_better": int(np.sum(bv < av)),
                "median_ratio_a_over_b": float(np.median(av / bv)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial_a_better": rbc,
            }
        )
    order = np.argsort(pvals)
    adjusted = [None] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        value = min(1.0, pvals[idx] * (m - rank))
        running = max(running, value)
        adjusted[idx] = running
    pairwise = pd.DataFrame(rows)
    pairwise["holm_p_value"] = adjusted
    stats_info = {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "kendall_w": float(kendall_w),
        "n_blocks": int(pivot.shape[0]),
        "n_methods": int(pivot.shape[1]),
    }
    return pairwise, stats_info


def write_reference_statistics_table(pairwise: pd.DataFrame, stats_info: dict[str, float]) -> None:
    rows = [
        " & ".join(
            [
                tex_escape(r["method_a"]),
                tex_escape(r["method_b"]),
                f"{int(r['a_better'])}/{int(r['b_better'])}",
                num(r["median_ratio_a_over_b"], 3),
                num(r["rank_biserial_a_better"], 3),
                num(r["holm_p_value"], 2),
            ]
        )
        + r" \\"
        for _, r in pairwise.iterrows()
    ]
    write(
        TAB / "results_reference_pairwise_statistics.tex",
        table_env(
            "tab:results_reference_pairwise_statistics",
            (
                "Matched pairwise Wilcoxon signed-rank comparisons on Sim(3)-aligned reference point RMSE. "
                rf"The overall Friedman test gives $\chi^2={stats_info['friedman_statistic']:.1f}$, "
                rf"$p={stats_info['friedman_p_value']:.2e}$, Kendall's $W={stats_info['kendall_w']:.3f}$ "
                rf"for {stats_info['n_blocks']} matched datasets and {stats_info['n_methods']} methods."
            ),
            "llrrrr",
            r"Method A & Method B & A/B wins & Median RMSE ratio & Rank-biserial & Holm $p$",
            rows,
        ),
    )
    pairwise.to_csv(TAB / "results_reference_pairwise_statistics.csv", index=False)
    write(
        TAB / "results_reference_friedman_summary.txt",
        "\n".join(f"{k}: {v}" for k, v in stats_info.items()),
    )


def plot_reference_recovery(recovery: pd.DataFrame) -> None:
    recovery = recovery.copy()
    recovery["method"] = pd.Categorical(recovery["method"], METHOD_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=recovery,
        x="method",
        y="point_rmse",
        order=METHOD_ORDER,
        ax=ax,
        fliersize=0.8,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(m, "#777777") for m in METHOD_ORDER],
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Aligned point RMSE to reference")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "results_reference_point_rmse.pdf")
    plt.close(fig)

    summary = (
        recovery.groupby(["category", "method"], observed=False)["point_rmse_recovery_ratio"]
        .median()
        .reset_index()
    )
    pivot = summary.pivot(index="category", columns="method", values="point_rmse_recovery_ratio")
    pivot = pivot[METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    sns.heatmap(
        pivot,
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
    fig.savefig(FIG / "results_reference_recovery_ratio_heatmap.pdf")
    plt.close(fig)

    method_order = (
        recovery.groupby("method", observed=False)["point_rmse"]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    category_order = [c for c in ["Close-Range", "Oblique-5", "UAV", "Vehicle"] if c in recovery["category"].unique()]
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
        palette=[METHOD_COLORS.get(m, "#777777") for m in method_order],
        orient="h",
    )
    ax_box.set_xscale("log")
    ax_box.set_title("(a) Reference point RMSE", loc="left", fontsize=10, fontweight="bold")
    ax_box.set_xlabel("Reference point RMSE", fontsize=9.5)
    ax_box.set_ylabel("")
    ax_box.tick_params(axis="x", labelsize=8.5)
    ax_box.tick_params(axis="y", labelsize=7.8, pad=1.5)
    ax_box.grid(axis="x", alpha=0.25)

    positive = ratio_pivot.to_numpy(dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    ratio_ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ratio_tick_labels = ["1e-4", "1e-3", "1e-2", "0.1", "1", "10"]
    norm = matplotlib.colors.LogNorm(vmin=ratio_ticks[0], vmax=ratio_ticks[-1])
    heat = sns.heatmap(
        ratio_pivot,
        cmap="viridis_r",
        norm=norm,
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
    fig.savefig(FIG / "results_reference_recovery_combined.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def d144_quality_label(quality_dataset: str) -> str:
    for token, label in D144_QUALITY_ORDER:
        if token in quality_dataset:
            return label
    return quality_dataset


def d144_accuracy_heatmap_values() -> pd.DataFrame:
    accuracy = pd.read_csv(D144_ABSOLUTE / "checkpoint_absolute_accuracy_by_dataset_method.csv")
    accuracy["quality_label"] = accuracy["quality_dataset"].map(d144_quality_label)
    accuracy["quality_label"] = pd.Categorical(
        accuracy["quality_label"],
        [label for _, label in D144_QUALITY_ORDER],
        ordered=True,
    )
    accuracy["rmse_3d"] = accuracy["rmse_3d"].astype(float)
    accuracy["excess_mm"] = (
        accuracy["rmse_3d"]
        - accuracy.groupby("quality_label", observed=False)["rmse_3d"].transform("min")
    ) * 1000.0

    method_order = (
        accuracy.groupby("method", observed=False)["rmse_3d"]
        .mean()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    pivot = accuracy.pivot(index="method", columns="quality_label", values="excess_mm")
    pivot = pivot.loc[method_order, [label for _, label in D144_QUALITY_ORDER]]
    return np.log10(1.0 + pivot)


def plot_d144_control_layout(
    ax,
    title: str,
    *,
    text_scale: float = 1.0,
    marker_scale: float = 1.0,
    label_offsets: dict[str, tuple[float, float]] | None = None,
) -> None:
    gcp = pd.read_csv(D144_GCP_SPLIT)
    control = gcp[gcp["role"].astype(str).str.lower() == "control"].copy()
    check = gcp[gcp["role"].astype(str).str.lower() == "checkpoint"].copy()
    ax.scatter(
        control["x"],
        control["y"],
        s=46 * marker_scale,
        marker="o",
        color="#0072B2",
        edgecolor="white",
        linewidth=0.6 * marker_scale**0.5,
        label="GCP",
        zorder=3,
    )
    ax.scatter(
        check["x"],
        check["y"],
        s=58 * marker_scale,
        marker="^",
        color="#D55E00",
        edgecolor="white",
        linewidth=0.6 * marker_scale**0.5,
        label="Check point",
        zorder=3,
    )
    label_offsets = label_offsets or {}
    for _, row in gcp.iterrows():
        label = str(row["name"]).replace("GCP", "")
        offset = label_offsets.get(label, (3.0, 3.0))
        arrowprops = None
        if label in label_offsets:
            arrowprops = {
                "arrowstyle": "-",
                "color": "0.42",
                "linewidth": 0.35 * text_scale,
                "connectionstyle": "arc3,rad=0.12",
                "shrinkA": 1.0,
                "shrinkB": 2.5,
            }
        ax.annotate(
            label,
            (row["x"], row["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8 * text_scale,
            color="0.25",
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            arrowprops=arrowprops,
        )

    if title:
        ax.set_title(title, loc="left", fontsize=10 * text_scale, fontweight="bold")
    ax.set_xlabel("World X (km)", fontsize=9.2 * text_scale)
    ax.set_ylabel("World Y (km)", fontsize=9.2 * text_scale)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value / 1000:.1f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value / 1000:.1f}"))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=8.0 * text_scale)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6 * text_scale)


def plot_d144_checkpoint_heatmap() -> None:
    heatmap_values = d144_accuracy_heatmap_values()
    fig, ax_heat = plt.subplots(figsize=(7.2, 4.35))

    heat = sns.heatmap(
        heatmap_values,
        ax=ax_heat,
        cmap="YlOrRd",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"fraction": 0.07, "pad": 0.035},
    )
    ax_heat.set_title("Check-point 3D RMSE excess", loc="left", fontsize=10, fontweight="bold")
    ax_heat.set_xlabel("Initialization setting", fontsize=9.2)
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", labelrotation=25, labelsize=8.0)
    ax_heat.tick_params(axis="y", labelsize=7.5, pad=1.0)

    colorbar = heat.collections[0].colorbar
    color_ticks_mm = np.array([0, 1, 3, 10, 30, 60], dtype=float)
    colorbar.set_ticks(np.log10(1.0 + color_ticks_mm))
    colorbar.set_ticklabels(["0", "1", "3", "10", "30", "60"])
    colorbar.ax.tick_params(labelsize=7.5)
    colorbar.ax.set_title("mm", fontsize=8.0, pad=4)

    fig.subplots_adjust(left=0.24, right=0.95, bottom=0.18, top=0.92)
    fig.savefig(FIG / "results_d144_checkpoint_rmse_excess.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_d144_control_layout_only() -> None:
    fig, ax_layout = plt.subplots(figsize=(7.2, 5.4))
    label_offsets = {
        "0001": (10, -11),
        "0005": (8, 3),
        "0009": (-13, 9),
        "0010": (-20, 5),
        "0013": (10, 8),
        "0014": (10, 8),
        "0016": (10, 8),
        "0017": (10, 8),
        "0018": (10, -6),
        "0019": (12, 10),
        "0020": (10, 9),
        "0021": (-10, -8),
    }
    plot_d144_control_layout(
        ax_layout,
        "",
        text_scale=1.45,
        marker_scale=1.55,
        label_offsets=label_offsets,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.085, top=0.985)
    fig.savefig(FIG / "results_d144_control_layout_only.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_d144_control_layout_panel() -> None:
    fig, ax_layout = plt.subplots(figsize=(4.9, 4.15))
    label_offsets = {
        "0001": (10, -11),
        "0005": (8, 3),
        "0009": (-13, 9),
        "0010": (-20, 5),
        "0013": (10, 8),
        "0014": (10, 8),
        "0016": (10, 8),
        "0017": (10, 8),
        "0018": (10, -6),
        "0019": (12, 10),
        "0020": (10, 9),
        "0021": (-10, -8),
    }
    plot_d144_control_layout(
        ax_layout,
        "",
        text_scale=1.45,
        marker_scale=1.55,
        label_offsets=label_offsets,
    )
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.12, top=0.985)
    fig.savefig(FIG / "results_d144_control_layout_panel.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_d144_checkpoint_heatmap_panel() -> None:
    heatmap_values = d144_accuracy_heatmap_values()
    fig, ax_heat = plt.subplots(figsize=(5.1, 4.15))

    heat = sns.heatmap(
        heatmap_values,
        ax=ax_heat,
        cmap="YlOrRd",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"fraction": 0.075, "pad": 0.03},
    )
    ax_heat.set_xlabel("Initialization setting", fontsize=11.0)
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", labelrotation=35, labelsize=9.2)
    ax_heat.tick_params(axis="y", labelsize=8.7, pad=1.0)

    colorbar = heat.collections[0].colorbar
    color_ticks_mm = np.array([0, 1, 3, 10, 30, 60], dtype=float)
    colorbar.set_ticks(np.log10(1.0 + color_ticks_mm))
    colorbar.set_ticklabels(["0", "1", "3", "10", "30", "60"])
    colorbar.ax.tick_params(labelsize=8.6)
    colorbar.ax.set_title("mm", fontsize=9.0, pad=4)

    fig.subplots_adjust(left=0.36, right=0.94, bottom=0.21, top=0.98)
    fig.savefig(FIG / "results_d144_checkpoint_rmse_excess_panel.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def d144_quality_path(token: str) -> Path:
    matches = sorted((D144_INPUT_ROOT / "quality").glob(f"*{token}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot identify D144 quality folder for {token}")
    return matches[0]


def load_d144_geometry(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    points = np.loadtxt(folder / "XYZ.txt", dtype=np.float64)
    pose = np.loadtxt(find_camera_file(folder), dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if pose.ndim == 1:
        pose = pose.reshape(1, -1)
    return points[:, :3], pose[:, :6]


def draw_camera_frustums(
    ax,
    pose: np.ndarray,
    *,
    count: int = 55,
    scale_km: float = 0.16,
    color: str = "#D55E00",
) -> None:
    if len(pose) == 0:
        return
    indices = np.linspace(0, len(pose) - 1, min(count, len(pose)), dtype=int)
    centers = pose[indices, 3:6] / 1000.0
    rotations = euler_to_world_to_camera(pose[indices, :3])
    local_corners = np.array(
        [
            [-0.50, -0.33, 1.00],
            [0.50, -0.33, 1.00],
            [0.50, 0.33, 1.00],
            [-0.50, 0.33, 1.00],
        ],
        dtype=np.float64,
    )
    local_corners = local_corners / np.linalg.norm(local_corners, axis=1, keepdims=True)
    for center, rotation in zip(centers, rotations):
        corners = center + scale_km * (rotation.T @ local_corners.T).T
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=color, linewidth=0.45, alpha=0.62)
        for corner in corners:
            segment = np.vstack([center, corner])
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=0.35, alpha=0.50)


def set_3d_limits_km(ax, bounds_km: np.ndarray) -> None:
    mins = bounds_km[0]
    maxs = bounds_km[1]
    spans = maxs - mins
    pad = np.maximum(spans * 0.04, np.array([0.03, 0.03, 0.01]))
    mins = mins - pad
    maxs = maxs + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect((maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]))


def plot_d144_geometry_control_layout() -> None:
    panels = [
        (
            "(a) Original D144 geometry",
            FIG / "appendix_d144_viewer_original.png",
            FIG / "appendix_d144_viewer_original_cropped.png",
        ),
        (
            "(b) Init 50 px geometry",
            FIG / "appendix_d144_viewer_init50.png",
            FIG / "appendix_d144_viewer_init50_cropped.png",
        ),
    ]

    def viewer_content_bounds(path: Path) -> tuple[int, int, int, int]:
        image = plt.imread(path)
        height, width = image.shape[:2]
        rgb = image[..., :3].astype(np.float64)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        brightness = rgb.sum(axis=2)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        bright_points = (brightness > 0.82) & (spread < 0.42)
        camera_cyan = (rgb[..., 1] > 0.42) & (rgb[..., 2] > 0.38) & (rgb[..., 0] < 0.45)
        ys, xs = np.where(bright_points | camera_cyan)
        if len(xs) == 0:
            return 0, 0, width, height
        x0, x1 = np.quantile(xs, [0.005, 0.995])
        y0, y1 = np.quantile(ys, [0.005, 0.995])
        pad_x = 0.085 * (x1 - x0)
        pad_y = 0.14 * (y1 - y0)
        return (
            max(0, int(np.floor(x0 - pad_x))),
            max(0, int(np.floor(y0 - pad_y))),
            min(width, int(np.ceil(x1 + pad_x))),
            min(height, int(np.ceil(y1 + pad_y))),
        )

    def crop_viewer_image(path: Path, bounds: tuple[int, int, int, int]) -> np.ndarray:
        image = plt.imread(path)
        x0, y0, x1, y1 = bounds
        return image[y0:y1, x0:x1]

    content_bounds = [viewer_content_bounds(path) for _, path, _ in panels]
    shared_bounds = (
        min(bounds[0] for bounds in content_bounds),
        min(bounds[1] for bounds in content_bounds),
        max(bounds[2] for bounds in content_bounds),
        max(bounds[3] for bounds in content_bounds),
    )

    cropped_panels = []
    for title, path, cropped_path in panels:
        cropped = crop_viewer_image(path, shared_bounds)
        plt.imsave(cropped_path, cropped)
        cropped_panels.append((title, cropped))

    fig = plt.figure(figsize=(7.2, 6.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.65], hspace=0.28, wspace=0.045)
    image_axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    for ax, (title, cropped) in zip(image_axes, cropped_panels):
        ax.imshow(cropped, interpolation="none")
        ax.set_title(title, loc="left", fontsize=9.8, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("0.15")
    ax_layout = fig.add_subplot(grid[1, :])
    plot_d144_control_layout(ax_layout, "(c) Control/check-point layout")
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.06, top=0.95)
    fig.savefig(FIG / "results_d144_geometry_control_layout.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_family_table() -> None:
    variable = pd.read_csv(CLEAN / "analysis" / "variable_family_summary_new.csv")
    frame = pd.read_csv(CLEAN / "analysis" / "frame_family_summary_new.csv")

    best_cols = {
        "within_1pct_rate": "max",
        "within_5pct_rate": "max",
        "median_rmse_gap_pct": "min",
        "median_iterations": "min",
        "max_iter_rate": "min",
    }

    def block_bests(data: pd.DataFrame) -> dict[str, float]:
        return {
            col: float(data[col].max() if direction == "max" else data[col].min())
            for col, direction in best_cols.items()
        }

    def family_row(row: pd.Series, best: dict[str, float]) -> str:
        return (
            " & ".join(
                [
                    tex_escape(row.iloc[0]),
                    f"{int(row['n'])}",
                    best_pct(row["within_1pct_rate"], best["within_1pct_rate"]),
                    best_pct(row["within_5pct_rate"], best["within_5pct_rate"]),
                    best_num(row["median_rmse_gap_pct"], best["median_rmse_gap_pct"], 1),
                    best_num(row["median_iterations"], best["median_iterations"], 1),
                    best_pct(row["max_iter_rate"], best["max_iter_rate"]),
                ]
            )
            + r" \\"
        )

    variable_best = block_bests(variable)
    frame_best = block_bests(frame)
    rows = [r"\multicolumn{7}{@{}l}{\textit{Variable family}} \\"]
    for _, row in variable.sort_values("within_1pct_rate", ascending=False).iterrows():
        row = row.rename({"variable_family": "family"})
        rows.append(family_row(row[["family", "n", *best_cols.keys()]], variable_best))
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{@{}l}{\textit{Frame family}} \\")
    for _, row in frame.sort_values("within_1pct_rate", ascending=False).iterrows():
        row = row.rename({"frame_family": "family"})
        rows.append(family_row(row[["family", "n", *best_cols.keys()]], frame_best))
    body = "\n".join(rows)
    write(
        TAB / "results_factor_summary.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Factor-wise aggregation by object-point variable and reference-frame family.}}
\label{{tab:results_factor_summary}}
\footnotesize
\setlength{{\tabcolsep}}{{4.0pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Family}} & \multirow{{2}}{{*}}{{Runs}} & Within & Within & Med. gap & Med. & Max-it. \\
 & & 1\% & 5\% & (\%) & it. & (\%) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Bold values mark the best entry within each family block. Higher values are better for Within 1\% and Within 5\%; lower values are better for median RMSE gap, median iterations, and Max-it.
\end{{minipage}}
\end{{table*}}
""",
    )


def write_appendix_category_table() -> None:
    cat = pd.read_csv(CLEAN / "analysis" / "category_method_summary_new.csv")
    category_order = [c for c in ["Close-Range", "Oblique-5", "UAV", "Vehicle"] if c in set(cat["category"])]
    method_order = {method: i for i, method in enumerate(METHOD_ORDER)}
    cat["method_order"] = cat["method"].map(method_order)
    cat["category_order"] = cat["category"].map({c: i for i, c in enumerate(category_order)})
    cat = cat.sort_values(["category_order", "method_order"])

    rows = []
    for category in category_order:
        block = cat[cat["category"] == category]
        if block.empty:
            continue
        bests = {
            "within_1pct_rate": float(block["within_1pct_rate"].max()),
            "within_5pct_rate": float(block["within_5pct_rate"].max()),
            "median_rmse_gap_pct": float(block["median_rmse_gap_pct"].min()),
            "median_iterations": float(block["median_iterations"].min()),
            "max_iter_rate": float(block["max_iter_rate"].min()),
        }
        rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{tex_escape(category)}}}}} \\")
        for _, row in block.iterrows():
            rows.append(
                " & ".join(
                    [
                        "",
                        tex_escape(row["method"]),
                        f"{int(row['n'])}",
                        best_pct(row["within_1pct_rate"], bests["within_1pct_rate"]),
                        best_pct(row["within_5pct_rate"], bests["within_5pct_rate"]),
                        best_num(row["median_rmse_gap_pct"], bests["median_rmse_gap_pct"], 1),
                        best_num(row["median_iterations"], bests["median_iterations"], 1),
                        best_pct(row["max_iter_rate"], bests["max_iter_rate"]),
                    ]
                )
                + r" \\"
            )
    write(
        TAB / "appendix_category_method_summary.tex",
        rf"""\begingroup
\footnotesize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{longtable}}{{@{{}}llrrrrrr@{{}}}}
\caption{{Category-wise robustness and convergence summary for every method.}}
\label{{tab:appendix_category_method_summary}}\\
\toprule
\multirow{{2}}{{*}}{{Category}} & \multirow{{2}}{{*}}{{Method}} & \multirow{{2}}{{*}}{{Runs}} & Within & Within & Med. gap & Med. & Max-it. \\
 & & & 1\% & 5\% & (\%) & it. & (\%) \\
\midrule
\endfirsthead
\toprule
\multirow{{2}}{{*}}{{Category}} & \multirow{{2}}{{*}}{{Method}} & \multirow{{2}}{{*}}{{Runs}} & Within & Within & Med. gap & Med. & Max-it. \\
 & & & 1\% & 5\% & (\%) & it. & (\%) \\
\midrule
\endhead
\midrule
\multicolumn{{8}}{{@{{}}r}}{{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
{chr(10).join(rows)}
\end{{longtable}}
\par\smallskip
\footnotesize Bold values mark the best entry within each category block. Higher values are better for Within 1\% and Within 5\%; lower values are better for median RMSE gap, median iterations, and Max-it.
\endgroup
""",
    )


def write_appendix_convergence_case_table() -> None:
    diag = pd.read_csv(DIAG / "summary.csv")
    cases = REPRESENTATIVE_CONVERGENCE_CASES + [BOUNDARY_CONVERGENCE_CASE]
    case_order = {dataset: i for i, (dataset, _) in enumerate(cases)}
    method_order = {method: i for i, method in enumerate(CONVERGENCE_METHODS)}
    diag = diag[
        diag["base_dataset"].isin(case_order)
        & diag["method"].isin(CONVERGENCE_METHODS)
    ].copy()
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
            if np.isclose(float(value), best, rtol=1e-9, atol=1e-12):
                return rf"\textbf{{{text}}}"
            return text

        def mark_display_num(value: float, best: float, digits: int = 3) -> str:
            text = num(value, digits)
            if text == num(best, digits):
                return rf"\textbf{{{text}}}"
            return text

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
    body = "\n".join(rows)
    write(
        TAB / "appendix_convergence_case_summary.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Numerical summaries for the convergence-curve cases.}}
\label{{tab:appendix_convergence_case_summary}}
\footnotesize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Case}} & \multirow{{2}}{{*}}{{Method}} & Initial & Final & \multirow{{2}}{{*}}{{Iter.}} & \multirow{{2}}{{*}}{{Rejected}} & Final LM \\
 & & RMSE & RMSE & & & gain \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Bold values mark the best entry within each case block for final RMSE, iteration count, and rejected steps. Lower values are better for these three columns. Initial RMSE is common input information, and final LM gain is reported as a diagnostic value.
\end{{minipage}}
\end{{table*}}
""",
    )


def statistical_tests(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    pivot = clean.pivot(index="base_dataset", columns="method", values="final_rmse_px")
    pivot = pivot[METHOD_ORDER]
    friedman = stats.friedmanchisquare(*[pivot[m].to_numpy() for m in METHOD_ORDER])
    kendall_w = friedman.statistic / (pivot.shape[0] * (pivot.shape[1] - 1))
    pairs = [
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
    pvals = []
    for a, b in pairs:
        av = pivot[a].to_numpy(float)
        bv = pivot[b].to_numpy(float)
        log_ratio = np.log(av / bv)
        test = stats.wilcoxon(log_ratio, zero_method="wilcox", alternative="two-sided")
        pvals.append(float(test.pvalue))
        d = np.log(bv / av)  # positive means a has smaller RMSE.
        nonzero = np.abs(d) > 1e-15
        ranks = stats.rankdata(np.abs(d[nonzero]))
        rbc = float(np.sum(np.sign(d[nonzero]) * ranks) / np.sum(ranks))
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "a_better": int(np.sum(av < bv)),
                "b_better": int(np.sum(bv < av)),
                "median_ratio_a_over_b": float(np.median(av / bv)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "rank_biserial_a_better": rbc,
            }
        )
    order = np.argsort(pvals)
    adjusted = [None] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        value = min(1.0, pvals[idx] * (m - rank))
        running = max(running, value)
        adjusted[idx] = running
    pairwise = pd.DataFrame(rows)
    pairwise["holm_p_value"] = adjusted
    stats_info = {
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "kendall_w": float(kendall_w),
        "n_blocks": int(pivot.shape[0]),
        "n_methods": int(pivot.shape[1]),
    }
    return pairwise, stats_info


def write_statistics_table(pairwise: pd.DataFrame, stats_info: dict[str, float]) -> None:
    rows = [
        " & ".join(
            [
                tex_escape(r["method_a"]),
                tex_escape(r["method_b"]),
                f"{int(r['a_better'])}/{int(r['b_better'])}",
                num(r["median_ratio_a_over_b"], 3),
                num(r["rank_biserial_a_better"], 3),
                num(r["holm_p_value"], 2),
            ]
        )
        + r" \\"
        for _, r in pairwise.iterrows()
    ]
    write(
        TAB / "results_pairwise_statistics.tex",
        table_env(
            "tab:results_pairwise_statistics",
            (
                "Matched pairwise Wilcoxon signed-rank comparisons on final reprojection RMSE. "
                rf"The overall Friedman test gives $\chi^2={stats_info['friedman_statistic']:.1f}$, "
                rf"$p={stats_info['friedman_p_value']:.2e}$, Kendall's $W={stats_info['kendall_w']:.3f}$ "
                rf"for {stats_info['n_blocks']} matched datasets and {stats_info['n_methods']} methods."
            ),
            "llrrrr",
            r"Method A & Method B & A/B wins & Median RMSE ratio & Rank-biserial & Holm $p$",
            rows,
        ),
    )
    pairwise.to_csv(TAB / "results_pairwise_statistics.csv", index=False)
    write(TAB / "results_friedman_summary.txt", "\n".join(f"{k}: {v}" for k, v in stats_info.items()))


def collect_diagnostic_summaries(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    for path in DIAG.rglob("schur_summary.csv"):
        parts = path.relative_to(DIAG).parts
        if len(parts) < 4:
            continue
        frame = pd.read_csv(path)
        frame["method"] = parts[-2]
        schur_rows.append(frame)
    schur = pd.concat(schur_rows, ignore_index=True)
    for col in ["condition_number", "numerical_rank", "nullity"]:
        schur[col] = pd.to_numeric(schur[col], errors="coerce")
    schur_summary = (
        schur.groupby("method")
        .agg(
            schur_samples=("method", "size"),
            median_schur_condition=("condition_number", "median"),
            q90_schur_condition=("condition_number", lambda x: x.quantile(0.9)),
            median_schur_nullity=("nullity", "median"),
        )
        .reset_index()
    )

    point_rows = []
    for path in DIAG.rglob("point_block_conditioning.csv"):
        parts = path.relative_to(DIAG).parts
        if len(parts) < 4:
            continue
        frame = pd.read_csv(path, usecols=["condition_number", "numerical_rank"])
        frame["method"] = parts[-2]
        point_rows.append(frame)
    point = pd.concat(point_rows, ignore_index=True)
    point["condition_number"] = pd.to_numeric(point["condition_number"], errors="coerce")
    point["condition_number"] = point["condition_number"].replace([np.inf, -np.inf], np.nan)
    point["numerical_rank"] = pd.to_numeric(point["numerical_rank"], errors="coerce")
    point_summary = (
        point.groupby("method")
        .agg(
            point_samples=("method", "size"),
            median_point_condition=("condition_number", "median"),
            q90_point_condition=("condition_number", lambda x: x.quantile(0.9)),
            point_rank_def_rate=("numerical_rank", lambda x: (x < 3).mean()),
        )
        .reset_index()
    )

    diagnostic = diag_summary.merge(point_summary, on="method").merge(schur_summary, on="method")
    diagnostic["method"] = pd.Categorical(diagnostic["method"], METHOD_ORDER, ordered=True)
    diagnostic = diagnostic.sort_values("method")
    diagnostic.to_csv(TAB / "results_diagnostic_summary.csv", index=False)
    return diagnostic, point, schur


def write_diagnostic_table(diagnostic: pd.DataFrame) -> None:
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
    body = "\n".join(rows)
    write(
        TAB / "results_diagnostic_summary.tex",
        rf"""\begin{{table*}}[t]
\centering
\caption{{Strict diagnostic summary. Point-block and Schur-complement condition numbers are sampled at the final state and computed without LM damping.}}
\label{{tab:results_diagnostic_summary}}
\footnotesize
\setlength{{\tabcolsep}}{{2.6pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
\multirow{{2}}{{*}}{{Method}} & Med. & Rejected & LM & Med. & Q90 & Med. \\
 & it. & steps & gain & $\kappa(C_k)$ & $\kappa(C_k)$ & $\kappa^+(S)$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\par\smallskip
\begin{{minipage}}{{0.98\textwidth}}
\footnotesize Bold values mark the best method only for optimizer-effort and trust-region reliability columns: median iterations, rejected steps, and LM gain. Lower values are better for median iterations, rejected steps, and condition numbers; higher values are better for LM gain. Condition numbers are undamped structural diagnostics, not the damped LM systems used for individual trial steps, and are not bolded.
\end{{minipage}}
\end{{table*}}
""",
    )


def plot_method_ranking(method: pd.DataFrame) -> None:
    data = method.sort_values("within_1pct_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = [METHOD_COLORS.get(m, "#777777") for m in data["method"]]
    ax.barh(data["method"].astype(str), 100 * data["within_1pct_rate"], color=colors, alpha=0.9)
    ax.set_xlabel("Runs within 1% of dataset-best RMSE (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "results_method_within1pct.pdf")
    plt.close(fig)


def plot_pareto(method: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for _, row in method.iterrows():
        m = str(row["method"])
        ax.scatter(
            row["median_solver_time_sec"],
            100 * row["within_1pct_rate"],
            s=60 + 260 * row["max_iter_rate"],
            color=METHOD_COLORS.get(m, "#777777"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )
        if m.startswith("A2-") or m in {"A0-SphInvRange-W", "A1-XYInvZ-Ac"}:
            ax.annotate(m, (row["median_solver_time_sec"], 100 * row["within_1pct_rate"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Median solver time (s)")
    ax.set_ylabel("Runs within 1% of dataset-best RMSE (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "results_accuracy_time_pareto.pdf")
    plt.close(fig)


def plot_robustness_pareto_combined(method: pd.DataFrame) -> None:
    ranking = method.sort_values("within_1pct_rate", ascending=True)
    fig = plt.figure(figsize=(7.2, 4.55))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.04], wspace=0.30)
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_pareto = fig.add_subplot(grid[0, 1])

    colors = [METHOD_COLORS.get(m, "#777777") for m in ranking["method"]]
    ax_rank.barh(ranking["method"].astype(str), 100 * ranking["within_1pct_rate"], color=colors, alpha=0.9)
    ax_rank.set_title("(a) Robustness ranking", loc="left", fontsize=10, fontweight="bold")
    ax_rank.set_xlabel("Within-1% rate (%)", fontsize=9.5)
    ax_rank.set_ylabel("")
    ax_rank.set_xlim(0, 100)
    ax_rank.grid(axis="x", alpha=0.25)
    ax_rank.tick_params(axis="x", labelsize=8.5)
    ax_rank.tick_params(axis="y", labelsize=7.8, pad=1.5)

    labels = {
        "A2-Parallax-Mw": "A2-Parallax-Mw",
        "A2-Parallax-Mc": "A2-Parallax-Mc",
        "A0-SphInvRange-W": "A0-SphInvRange-W",
        "A1-XYInvZ-Ac": "A1-XYInvZ-Ac",
    }
    label_positions = {
        "A2-Parallax-Mw": (14.3, 95.3, "left", 0.0),
        "A2-Parallax-Mc": (14.3, 91.0, "left", 0.0),
        "A0-SphInvRange-W": (18.2, 67.0, "left", 0.18),
        "A1-XYInvZ-Ac": (15.2, 53.0, "left", -0.18),
    }
    for _, row in method.iterrows():
        m = str(row["method"])
        ax_pareto.scatter(
            row["median_solver_time_sec"],
            100 * row["within_1pct_rate"],
            s=42 + 170 * row["max_iter_rate"],
            color=METHOD_COLORS.get(m, "#777777"),
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
        )
        if m in labels:
            tx, ty, ha, rad = label_positions[m]
            ax_pareto.annotate(
                labels[m],
                (row["median_solver_time_sec"], 100 * row["within_1pct_rate"]),
                xytext=(tx, ty),
                textcoords="data",
                fontsize=7.6,
                ha=ha,
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.42",
                    "linewidth": 0.55,
                    "alpha": 0.9,
                    "connectionstyle": f"arc3,rad={rad}",
                    "shrinkA": 1,
                    "shrinkB": 3,
                },
                clip_on=False,
            )
    ax_pareto.set_title("(b) Accuracy-time trade-off", loc="left", fontsize=10, fontweight="bold")
    ax_pareto.set_xlabel("Median time (s)", fontsize=9.5)
    ax_pareto.set_ylabel("Within-1% rate (%)", fontsize=9.5)
    ax_pareto.set_ylim(0, 100)
    ax_pareto.set_xlim(10.6, 25.4)
    ax_pareto.grid(alpha=0.25)
    ax_pareto.tick_params(axis="both", labelsize=8.5)

    fig.subplots_adjust(left=0.185, right=0.975, bottom=0.14, top=0.92)
    fig.savefig(FIG / "results_robustness_pareto_combined.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_category_heatmap() -> None:
    cat = pd.read_csv(CLEAN / "analysis" / "category_method_summary_new.csv")
    pivot = cat.pivot(index="category", columns="method", values="within_1pct_rate")
    category_order = [c for c in ["Close-Range", "Oblique-5", "UAV", "Vehicle"] if c in pivot.index]
    category_labels = {
        "Close-Range": "Close",
        "Oblique-5": "Oblique",
        "UAV": "UAV",
        "Vehicle": "Vehicle",
    }
    pivot = (100 * pivot.loc[category_order, METHOD_ORDER]).T
    pivot = pivot.rename(columns=category_labels)
    fig, ax = plt.subplots(figsize=(6.7, 4.85))
    heat = sns.heatmap(
        pivot,
        annot=False,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Within-1% (%)", "fraction": 0.06, "pad": 0.04},
        ax=ax,
    )
    for y, method in enumerate(pivot.index):
        for x, category in enumerate(pivot.columns):
            value = pivot.loc[method, category]
            color = "white" if value >= 55 else "black"
            ax.text(x + 0.5, y + 0.5, f"{value:.0f}", ha="center", va="center", fontsize=8.2, color=color)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8.8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7.8, pad=1.5)
    colorbar = heat.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Within-1% (%)", fontsize=8.5)
    fig.subplots_adjust(left=0.285, right=0.94, bottom=0.10, top=0.98)
    fig.savefig(FIG / "results_category_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_conditioning(point: pd.DataFrame, schur: pd.DataFrame) -> None:
    point = point.copy()
    point["method"] = pd.Categorical(point["method"], METHOD_ORDER, ordered=True)
    schur = schur.copy()
    schur["method"] = pd.Categorical(schur["method"], METHOD_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=point.dropna(subset=["condition_number"]),
        x="method",
        y="condition_number",
        order=METHOD_ORDER,
        ax=ax,
        fliersize=0.5,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(m, "#777777") for m in METHOD_ORDER],
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(r"Undamped point-block condition number $\kappa(C_k)$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "results_point_conditioning.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=schur.dropna(subset=["condition_number"]),
        x="method",
        y="condition_number",
        order=METHOD_ORDER,
        ax=ax,
        fliersize=0.8,
        linewidth=0.6,
        palette=[METHOD_COLORS.get(m, "#777777") for m in METHOD_ORDER],
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(r"Sampled undamped Schur condition number $\kappa^+(S)$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "results_schur_conditioning.pdf")
    plt.close(fig)

    method_order = (
        point.dropna(subset=["condition_number"])
        .groupby("method", observed=False)["condition_number"]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    fig = plt.figure(figsize=(7.2, 4.65))
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
        palette=[METHOD_COLORS.get(m, "#777777") for m in method_order],
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
        palette=[METHOD_COLORS.get(m, "#777777") for m in method_order],
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

    fig.subplots_adjust(left=0.255, right=0.985, bottom=0.14, top=0.92)
    fig.savefig(FIG / "results_second_order_combined.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_convergence_cases() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), sharex=False, sharey=False)
    for index, (ax, (dataset, title)) in enumerate(
        zip(axes.ravel(), REPRESENTATIVE_CONVERGENCE_CASES)
    ):
        for method in CONVERGENCE_METHODS:
            path = DIAG / dataset / "Initial Value" / method / "convergence_strict.csv"
            if not path.exists():
                continue
            curve = accepted_curve(pd.read_csv(path))
            ax.plot(
                curve["accepted_step"],
                curve["rmse_px"],
                label=method,
                color=CONVERGENCE_COLORS.get(method, METHOD_COLORS.get(method, None)),
                linestyle=CONVERGENCE_LINESTYLES.get(method, "-"),
                linewidth=1.75,
            )
        row, col = divmod(index, 2)
        ax.set_title(title, fontsize=10.8)
        ax.set_yscale("log")
        y_low, y_high = ax.get_ylim()
        if y_low > 0 and y_high / y_low < 6:
            compact_ticks = [tick for tick in (0.6, 1.0, 2.0) if y_low <= tick <= y_high]
            if compact_ticks:
                ax.set_yticks(compact_ticks)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(compact_log_tick))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("Accepted LM step" if row == 1 else "", fontsize=10.0)
        ax.set_ylabel("RMSE (px)" if col == 0 else "", fontsize=10.0)
        ax.tick_params(axis="both", labelsize=9.0)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.6, frameon=False)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.16, top=0.94, wspace=0.25, hspace=0.30)
    fig.savefig(FIG / "results_representative_convergence_grid.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    dataset, title = REPRESENTATIVE_CONVERGENCE_CASES[-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for method in CONVERGENCE_METHODS:
        path = DIAG / dataset / "Initial Value" / method / "convergence_strict.csv"
        if not path.exists():
            continue
        curve = accepted_curve(pd.read_csv(path))
        ax.plot(
            curve["accepted_step"],
            curve["rmse_px"],
            label=method,
            color=CONVERGENCE_COLORS.get(method, METHOD_COLORS.get(method, None)),
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
    fig.savefig(FIG / "results_representative_convergence_vehicle.pdf")
    plt.close(fig)

    dataset, title = BOUNDARY_CONVERGENCE_CASE
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for method in CONVERGENCE_METHODS:
        path = DIAG / dataset / "Initial Value" / method / "convergence_strict.csv"
        if not path.exists():
            continue
        curve = accepted_curve(pd.read_csv(path))
        ax.plot(
            curve["accepted_step"],
            curve["rmse_px"],
            label=method,
            color=CONVERGENCE_COLORS.get(method, METHOD_COLORS.get(method, None)),
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
    fig.savefig(FIG / "appendix_convergence_kd4.pdf")
    plt.close(fig)

    plot_convergence_step_counts()


def plot_convergence_step_counts() -> None:
    cases = REPRESENTATIVE_CONVERGENCE_CASES + [BOUNDARY_CONVERGENCE_CASE]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.9), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    method_positions = {method: i for i, method in enumerate(CONVERGENCE_METHODS)}
    method_labels = [
        "Parallax-Mw",
        "Parallax-Mc",
        "SphInvRange-W",
        "XYInvZ-Ac",
        "XYZ-W",
    ]
    for index, (ax, (dataset, title)) in enumerate(zip(axes_flat, cases)):
        for method in CONVERGENCE_METHODS:
            path = DIAG / dataset / "Initial Value" / method / "convergence_strict.csv"
            if not path.exists():
                continue
            curve = trial_count_curve(pd.read_csv(path))
            if curve.empty:
                continue
            method_y = method_positions[method]
            method_color = CONVERGENCE_COLORS.get(method, METHOD_COLORS.get(method, None))
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
            if retry.empty:
                continue
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
        ax.set_ylabel("")
        ax.set_yticks(range(len(CONVERGENCE_METHODS)))
        if col == 0:
            ax.set_yticklabels(method_labels, fontsize=7.7)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(len(CONVERGENCE_METHODS) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=8.0)
        ax.grid(alpha=0.25)
        ax.grid(axis="y", alpha=0.16)
    legend_axes = axes_flat[len(cases):]
    if len(legend_axes) > 0:
        legend_axes[0].axis("off")
        legend_counts = [2, 3, 5]
        handles = [
            Line2D(
                [0],
                [0],
                color="#666666",
                linewidth=retry_line_width(np.array([count]))[0],
                label=f"{count} trials",
            )
            for count in legend_counts
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle="None",
                marker=">",
                markerfacecolor="none",
                markeredgecolor="#666666",
                markeredgewidth=1.15,
                markersize=7.2,
                label="Final accepted step",
            )
        )
        legend_axes[0].legend(
            handles=handles,
            loc="center",
            title="Event meaning",
            fontsize=8.3,
            title_fontsize=8.8,
            frameon=False,
            labelspacing=1.0,
        )
    for ax in legend_axes[1:]:
        ax.axis("off")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.955, wspace=0.24, hspace=0.42)
    fig.savefig(FIG / "appendix_convergence_step_counts.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_lm_damping_paths() -> None:
    cases = REPRESENTATIVE_CONVERGENCE_CASES + [BOUNDARY_CONVERGENCE_CASE]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.9), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    legend_handles: dict[str, Line2D] = {}
    for index, (ax, (dataset, title)) in enumerate(zip(axes_flat, cases)):
        for method in CONVERGENCE_METHODS:
            path = DIAG / dataset / "Initial Value" / method / "convergence_strict.csv"
            if not path.exists():
                continue
            curve = pd.read_csv(path)
            if "lm_damping" not in curve.columns or "iteration" not in curve.columns:
                continue
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
            method_color = CONVERGENCE_COLORS.get(method, METHOD_COLORS.get(method, None))
            (line,) = ax.plot(
                working["_iteration"],
                working["_lm_damping"],
                label=method,
                color=method_color,
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
    legend_axes = axes_flat[len(cases):]
    if len(legend_axes) > 0:
        legend_axes[0].axis("off")
        handles = [legend_handles[m] for m in CONVERGENCE_METHODS if m in legend_handles]
        labels = [h.get_label() for h in handles]
        legend_axes[0].legend(
            handles=handles,
            labels=labels,
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
    fig.savefig(FIG / "appendix_lm_damping_paths.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    clean, method, diag = load_inputs()
    recovery = compute_reference_recovery()
    write_execution_table(clean, diag)
    write_method_table(method)
    write_reference_recovery_table(recovery)
    reference_pairwise, reference_stats = reference_recovery_tests(recovery)
    write_reference_statistics_table(reference_pairwise, reference_stats)
    write_family_table()
    write_appendix_category_table()
    write_appendix_convergence_case_table()
    pairwise, stats_info = statistical_tests(clean)
    write_statistics_table(pairwise, stats_info)
    diagnostic, point, schur = collect_diagnostic_summaries(diag)
    write_diagnostic_table(diagnostic)
    plot_method_ranking(method)
    plot_pareto(method)
    plot_robustness_pareto_combined(method)
    plot_category_heatmap()
    plot_reference_recovery(recovery)
    plot_d144_checkpoint_heatmap()
    plot_d144_control_layout_only()
    plot_d144_geometry_control_layout()
    plot_conditioning(point, schur)
    plot_convergence_cases()
    plot_lm_damping_paths()
    print("Generated result assets in:")
    print(f"  {FIG}")
    print(f"  {TAB}")


if __name__ == "__main__":
    main()
