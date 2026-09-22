#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import checkpoint_absolute_accuracy as ca


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--quality-root", required=True)
    parser.add_argument("--gcp", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    diagnostic_root = Path(args.diagnostic_root)
    quality_root = Path(args.quality_root)
    out_dir = Path(args.out_dir)
    gcp_records = ca.read_gcp(args.gcp)
    checkpoint_ids = [
        control_id
        for control_id, record in sorted(gcp_records.items())
        if record["is_checkpoint"]
    ]

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
        run_dir = diagnostic_root / base_dataset / quality_dataset / f"{method}__gcp-control"
        pose_path = run_dir / "FinalPose.txt"
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)

        observations = ca.read_gcp_observations(quality_dir / "gcp_observations.txt")
        calibrations = ca.read_calibration(quality_dir / "cal.txt")
        poses = ca.read_pose(pose_path)
        final_rmse_by_run_method[(quality_dataset, method)] = float(
            summary_row["final_rmse_px"]
        )

        for checkpoint_id in checkpoint_ids:
            truth = gcp_records[checkpoint_id]["xyz"]
            checkpoint_observations = observations[checkpoint_id]
            estimate, used = ca.triangulate(checkpoint_observations, poses, calibrations)
            delta = estimate - truth
            horizontal = math.hypot(delta[0], delta[1])
            vertical = abs(delta[2])
            total = float(ca.np.linalg.norm(delta))
            detail_rows.append(
                {
                    "base_dataset": base_dataset,
                    "quality_dataset": quality_dataset,
                    "method": method,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_name": gcp_records[checkpoint_id]["name"],
                    "observations": used,
                    "estimated_x": estimate[0],
                    "estimated_y": estimate[1],
                    "estimated_z": estimate[2],
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
                        estimate, checkpoint_observations, poses, calibrations
                    ),
                }
            )

    by_dataset_method = summarize(
        detail_rows,
        ["quality_dataset", "method"],
        final_rmse_by_run_method,
    )
    by_method = summarize(detail_rows, ["method"], final_rmse_by_run_method)
    by_method.sort(key=lambda row: row["rmse_3d"])

    write_csv(out_dir / "checkpoint_absolute_accuracy_detail.csv", detail_rows)
    write_csv(out_dir / "checkpoint_absolute_accuracy_by_dataset_method.csv", by_dataset_method)
    write_csv(out_dir / "checkpoint_absolute_accuracy_by_method.csv", by_method)

    review_rows = [
        {
            "method": row["method"],
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
