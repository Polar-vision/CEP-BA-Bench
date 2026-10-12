#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import checkpoint_absolute_accuracy as ca
import numpy as np


RUN_LEAF_SUFFIX = {
    "gcp-control": "__gcp-control",
    "free": "",
}


def fit_similarity(source, target):
    """Fit target = scale * rotation * source + translation (Umeyama)."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must be matching N x 3 arrays")
    if source.shape[0] < 3:
        raise ValueError("At least three control points are required for Sim(3) alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if not math.isfinite(variance) or variance <= np.finfo(float).eps:
        raise ValueError("Control points have insufficient spread for Sim(3) alignment")
    if np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("Control points are collinear; Sim(3) alignment is underdetermined")

    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(points, scale, rotation, translation):
    points = np.asarray(points, dtype=float)
    return scale * (points @ rotation.T) + translation


def fit_control_alignment(control_ids, gcp_records, observations, poses, calibrations):
    estimates = []
    truths = []
    for control_id in control_ids:
        if control_id not in observations:
            raise ValueError(f"Control point {control_id} has no image observations")
        estimate, _ = ca.triangulate(observations[control_id], poses, calibrations)
        estimates.append(estimate)
        truths.append(gcp_records[control_id]["xyz"])

    source = np.stack(estimates, axis=0)
    target = np.stack(truths, axis=0)
    scale, rotation, translation = fit_similarity(source, target)
    aligned = apply_similarity(source, scale, rotation, translation)
    errors = np.linalg.norm(aligned - target, axis=1)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "control_count": len(control_ids),
        "control_rmse_3d_m": rmse(errors),
    }


def rmse(values):
    values = list(values)
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarize(rows, keys, final_rmse_by_run_method):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    out = []
    for group_key, group_rows in sorted(grouped.items()):
        record = dict(zip(keys, group_key))
        control_counts = {
            int(row["alignment_control_count"]) for row in group_rows
        }
        if len(control_counts) != 1:
            raise ValueError(
                f"Inconsistent alignment control counts in group {group_key}: "
                f"{sorted(control_counts)}"
            )
        record.update(
            {
                "checkpoint_count": len(group_rows),
                "observation_count": sum(int(row["observations"]) for row in group_rows),
                "rmse_x": rmse(row["dx"] for row in group_rows),
                "rmse_y": rmse(row["dy"] for row in group_rows),
                "rmse_z": rmse(row["dz"] for row in group_rows),
                "rmse_horizontal": rmse(row["horizontal_error"] for row in group_rows),
                "rmse_3d": rmse(row["total_3d_error"] for row in group_rows),
                "mean_abs_z": mean(row["vertical_error"] for row in group_rows),
                "mean_reproj_rmse_px": mean(
                    row["checkpoint_reproj_rmse_px"] for row in group_rows
                ),
                "alignment_control_count": control_counts.pop(),
                "alignment_control_rmse_3d_m": mean(
                    row["alignment_control_rmse_3d_m"] for row in group_rows
                ),
                "similarity_scale": mean(
                    row["similarity_scale"] for row in group_rows
                ),
            }
        )
        matching_final = []
        if "quality_dataset" in keys and "method" in keys:
            matching_final = [
                final_rmse_by_run_method[(record["quality_dataset"], record["method"])]
            ]
        elif "method" in keys:
            method = record["method"]
            matching_final = [
                value
                for (_, run_method), value in final_rmse_by_run_method.items()
                if run_method == method
            ]
        if matching_final:
            record["mean_final_rmse_px"] = mean(matching_final)
        out.append(record)
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate held-out check points from final camera poses. Run-directory "
            "control and object-space alignment are selected independently so GCP-controlled "
            "and free-network runs can use the same post-hoc Sim(3) protocol."
        )
    )
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--quality-root", required=True)
    parser.add_argument("--gcp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--run-control-mode",
        choices=sorted(RUN_LEAF_SUFFIX),
        default="gcp-control",
        help=(
            "Run leaf convention: 'gcp-control' reads METHOD__gcp-control; "
            "'free' reads METHOD. Default: gcp-control."
        ),
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("direct", "control-sim3"),
        default="direct",
        help=(
            "Object-space comparison: 'direct' uses the run coordinates unchanged; "
            "'control-sim3' fits Sim(3) from all non-checkpoint measured controls. "
            "Default: direct."
        ),
    )
    args = parser.parse_args()

    if args.run_control_mode == "free" and args.alignment_mode == "direct":
        parser.error(
            "free-network runs require --alignment-mode control-sim3 because their "
            "gauge-fixed coordinates are not in the measured control frame"
        )

    diagnostic_root = Path(args.diagnostic_root)
    quality_root = Path(args.quality_root)
    out_dir = Path(args.out_dir)
    gcp_records = ca.read_gcp(args.gcp)
    checkpoint_ids = [
        control_id
        for control_id, record in sorted(gcp_records.items())
        if record["is_checkpoint"]
    ]
    control_ids = [
        control_id
        for control_id, record in sorted(gcp_records.items())
        if not record["is_checkpoint"]
    ]
    if args.alignment_mode == "control-sim3" and len(control_ids) < 3:
        raise ValueError(
            f"Sim(3) alignment requires at least three non-checkpoint controls; "
            f"found {len(control_ids)}"
        )

    summary_path = diagnostic_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    final_rmse_by_run_method = {}
    detail_rows = []
    for summary_row in csv.DictReader(summary_path.open()):
        if summary_row["status"] != "ok":
            continue
        base_dataset = summary_row["base_dataset"]
        quality_dataset = summary_row["quality_dataset"]
        method = summary_row["method"]
        quality_dir = quality_root / quality_dataset
        run_leaf = f"{method}{RUN_LEAF_SUFFIX[args.run_control_mode]}"
        run_dir = diagnostic_root / base_dataset / quality_dataset / run_leaf
        pose_path = run_dir / "FinalPose.txt"
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)

        observations = ca.read_gcp_observations(quality_dir / "gcp_observations.txt")
        calibrations = ca.read_calibration(quality_dir / "cal.txt")
        poses = ca.read_pose(pose_path)
        final_rmse_by_run_method[(quality_dataset, method)] = float(
            summary_row["final_rmse_px"]
        )

        if args.alignment_mode == "control-sim3":
            alignment = fit_control_alignment(
                control_ids, gcp_records, observations, poses, calibrations
            )
        else:
            alignment = {
                "scale": 1.0,
                "rotation": np.eye(3),
                "translation": np.zeros(3),
                "control_count": 0,
                "control_rmse_3d_m": float("nan"),
            }

        for checkpoint_id in checkpoint_ids:
            truth = gcp_records[checkpoint_id]["xyz"]
            checkpoint_observations = observations[checkpoint_id]
            raw_estimate, used = ca.triangulate(
                checkpoint_observations, poses, calibrations
            )
            estimate = apply_similarity(
                raw_estimate[None, :],
                alignment["scale"],
                alignment["rotation"],
                alignment["translation"],
            )[0]
            delta = estimate - truth
            horizontal = math.hypot(delta[0], delta[1])
            vertical = abs(delta[2])
            total = float(ca.np.linalg.norm(delta))
            detail_rows.append(
                {
                    "base_dataset": base_dataset,
                    "quality_dataset": quality_dataset,
                    "method": method,
                    "run_control_mode": args.run_control_mode,
                    "alignment_mode": args.alignment_mode,
                    "alignment_control_count": alignment["control_count"],
                    "alignment_control_rmse_3d_m": alignment["control_rmse_3d_m"],
                    "similarity_scale": alignment["scale"],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_name": gcp_records[checkpoint_id]["name"],
                    "observations": used,
                    "estimated_x": estimate[0],
                    "estimated_y": estimate[1],
                    "estimated_z": estimate[2],
                    "raw_estimated_x": raw_estimate[0],
                    "raw_estimated_y": raw_estimate[1],
                    "raw_estimated_z": raw_estimate[2],
                    "truth_x": truth[0],
                    "truth_y": truth[1],
                    "truth_z": truth[2],
                    "dx": delta[0],
                    "dy": delta[1],
                    "dz": delta[2],
                    "horizontal_error": horizontal,
                    "vertical_error": vertical,
                    "total_3d_error": total,
                    "checkpoint_reproj_rmse_px": ca.reprojection_rmse(
                        raw_estimate, checkpoint_observations, poses, calibrations
                    ),
                }
            )

    by_dataset_method = summarize(
        detail_rows,
        ["quality_dataset", "method", "run_control_mode", "alignment_mode"],
        final_rmse_by_run_method,
    )
    by_method = summarize(
        detail_rows,
        ["method", "run_control_mode", "alignment_mode"],
        final_rmse_by_run_method,
    )
    by_method.sort(key=lambda row: row["rmse_3d"])

    write_csv(out_dir / "checkpoint_absolute_accuracy_detail.csv", detail_rows)
    write_csv(out_dir / "checkpoint_absolute_accuracy_by_dataset_method.csv", by_dataset_method)
    write_csv(out_dir / "checkpoint_absolute_accuracy_by_method.csv", by_method)

    review_rows = [
        {
            "method": row["method"],
            "run_control_mode": row["run_control_mode"],
            "alignment_mode": row["alignment_mode"],
            "alignment_control_count": row["alignment_control_count"],
            "alignment_control_rmse_3d_m": row["alignment_control_rmse_3d_m"],
            "similarity_scale": row["similarity_scale"],
            "mean_final_rmse_px": row.get("mean_final_rmse_px", float("nan")),
            "checkpoint_count": row["checkpoint_count"],
            "checkpoint_observation_count": row["observation_count"],
            "checkpoint_rmse_horizontal_m": row["rmse_horizontal"],
            "checkpoint_rmse_z_m": row["rmse_z"],
            "checkpoint_rmse_3d_m": row["rmse_3d"],
            "checkpoint_reproj_rmse_px": row["mean_reproj_rmse_px"],
        }
        for row in by_method
    ]
    write_csv(out_dir / "checkpoint_review_table.csv", review_rows)

    print(out_dir / "checkpoint_review_table.csv")
    print(out_dir / "checkpoint_absolute_accuracy_by_method.csv")
    print(out_dir / "checkpoint_absolute_accuracy_by_dataset_method.csv")
    print(out_dir / "checkpoint_absolute_accuracy_detail.csv")


if __name__ == "__main__":
    main()
