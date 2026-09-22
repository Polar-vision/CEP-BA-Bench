#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import numpy as np


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


def euler_to_world_to_camera(euler):
    ey, ex, ez = euler
    c1, c2, c3 = math.cos(ey), math.cos(ex), math.cos(ez)
    s1, s2, s3 = math.sin(ey), math.sin(ex), math.sin(ez)
    return np.array(
        [
            [c1 * c3 - s1 * s2 * s3, c2 * s3, s1 * c3 + c1 * s2 * s3],
            [-c1 * s3 - s1 * s2 * c3, c2 * c3, -s1 * s3 + c1 * s2 * c3],
            [-s1 * c2, -s2, c1 * c2],
        ],
        dtype=float,
    )


def read_gcp(path):
    records = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            control_id = int(parts[0])
            records[control_id] = {
                "name": parts[1],
                "xyz": np.array([float(parts[2]), float(parts[3]), float(parts[4])]),
                "h_acc": float(parts[5]),
                "v_acc": float(parts[6]),
                "is_checkpoint": int(parts[7]) != 0,
            }
    return records


def read_gcp_observations(path):
    tracks = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            control_id = int(parts[0])
            n = int(parts[1])
            obs = []
            cursor = 2
            for _ in range(n):
                obs.append(
                    {
                        "camera": int(parts[cursor]),
                        "u": float(parts[cursor + 1]),
                        "v": float(parts[cursor + 2]),
                    }
                )
                cursor += 3
            tracks[control_id] = obs
    return tracks


def read_calibration(path):
    values = []
    with Path(path).open() as f:
        rows = [[float(x) for x in line.split()] for line in f if line.strip()]
    if len(rows) % 3 != 0:
        raise ValueError(f"Calibration file should have 3 rows per camera: {path}")
    for i in range(0, len(rows), 3):
        values.append(np.array(rows[i : i + 3], dtype=float))
    return values


def read_pose(path):
    poses = []
    with Path(path).open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            poses.append(
                {
                    "euler": np.array([float(parts[0]), float(parts[1]), float(parts[2])]),
                    "center": np.array([float(parts[3]), float(parts[4]), float(parts[5])]),
                    "calib_index": int(float(parts[6])) - 1 if len(parts) > 6 else 0,
                }
            )
    return poses


def triangulate(observations, poses, calibrations):
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    used = 0
    for obs in observations:
        camera_index = obs["camera"]
        if camera_index < 0 or camera_index >= len(poses):
            continue
        pose = poses[camera_index]
        k = calibrations[pose["calib_index"]]
        pixel = np.array([obs["u"], obs["v"], 1.0])
        ray_camera = np.linalg.solve(k, pixel)
        ray_camera /= np.linalg.norm(ray_camera)
        rotation = euler_to_world_to_camera(pose["euler"])
        ray_world = rotation.T @ ray_camera
        ray_world /= np.linalg.norm(ray_world)
        center = pose["center"]
        projector = np.eye(3) - np.outer(ray_world, ray_world)
        a += projector
        b += projector @ center
        used += 1
    if used < 2:
        raise ValueError("At least two valid observations are required for triangulation")
    xyz = np.linalg.solve(a, b)
    return xyz, used


def project(point, pose, k):
    rotation = euler_to_world_to_camera(pose["euler"])
    point_camera = rotation @ (point - pose["center"])
    return np.array(
        [
            k[0, 0] * point_camera[0] / point_camera[2] + k[0, 2],
            k[1, 1] * point_camera[1] / point_camera[2] + k[1, 2],
        ]
    )


def reprojection_rmse(point, observations, poses, calibrations):
    sq = []
    for obs in observations:
        camera_index = obs["camera"]
        if camera_index < 0 or camera_index >= len(poses):
            continue
        pose = poses[camera_index]
        pred = project(point, pose, calibrations[pose["calib_index"]])
        meas = np.array([obs["u"], obs["v"]])
        sq.append(float(np.sum((pred - meas) ** 2)))
    return math.sqrt(sum(sq) / len(sq)) if sq else float("nan")


def find_run_name(diagnostic_root, dataset_filter):
    candidates = list(Path(diagnostic_root).glob(f"*/{dataset_filter}"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one diagnostic dataset directory, found {len(candidates)}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    diagnostic_root = Path(args.diagnostic_root)
    output_path = Path(args.out)
    records = read_gcp(dataset_dir / "gcp.txt")
    tracks = read_gcp_observations(dataset_dir / "gcp_observations.txt")
    calibrations = read_calibration(dataset_dir / "cal.txt")
    dataset_run_dir = find_run_name(diagnostic_root, dataset_dir.name)
    checkpoints = [cid for cid, record in records.items() if record["is_checkpoint"]]

    rows = []
    for method in METHOD_ORDER:
        pose_path = dataset_run_dir / f"{method}__gcp-control" / "FinalPose.txt"
        if not pose_path.exists():
            continue
        poses = read_pose(pose_path)
        for checkpoint_id in checkpoints:
            truth = records[checkpoint_id]["xyz"]
            observations = tracks[checkpoint_id]
            estimate, used = triangulate(observations, poses, calibrations)
            delta = estimate - truth
            horizontal = math.hypot(delta[0], delta[1])
            vertical = abs(delta[2])
            total = float(np.linalg.norm(delta))
            rows.append(
                {
                    "method": method,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_name": records[checkpoint_id]["name"],
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
                    "checkpoint_reproj_rmse_px": reprojection_rmse(
                        estimate, observations, poses, calibrations
                    ),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "method",
            "checkpoint_count",
            "observation_count",
            "rmse_x",
            "rmse_y",
            "rmse_z",
            "rmse_horizontal",
            "rmse_3d",
            "mean_abs_z",
            "mean_reproj_rmse_px",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            method_rows = [row for row in rows if row["method"] == method]
            if not method_rows:
                continue
            writer.writerow(
                {
                    "method": method,
                    "checkpoint_count": len(method_rows),
                    "observation_count": sum(int(row["observations"]) for row in method_rows),
                    "rmse_x": math.sqrt(
                        sum(row["dx"] ** 2 for row in method_rows) / len(method_rows)
                    ),
                    "rmse_y": math.sqrt(
                        sum(row["dy"] ** 2 for row in method_rows) / len(method_rows)
                    ),
                    "rmse_z": math.sqrt(
                        sum(row["dz"] ** 2 for row in method_rows) / len(method_rows)
                    ),
                    "rmse_horizontal": math.sqrt(
                        sum(row["horizontal_error"] ** 2 for row in method_rows)
                        / len(method_rows)
                    ),
                    "rmse_3d": math.sqrt(
                        sum(row["total_3d_error"] ** 2 for row in method_rows)
                        / len(method_rows)
                    ),
                    "mean_abs_z": sum(row["vertical_error"] for row in method_rows)
                    / len(method_rows),
                    "mean_reproj_rmse_px": sum(
                        row["checkpoint_reproj_rmse_px"] for row in method_rows
                    )
                    / len(method_rows),
                }
            )
    print(output_path)
    print(summary_path)


if __name__ == "__main__":
    main()
