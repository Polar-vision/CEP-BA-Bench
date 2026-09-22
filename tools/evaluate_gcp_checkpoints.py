#!/usr/bin/env python3
"""Evaluate PVL-BA GCP/checkpoint accuracy with a control-point similarity fit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def euler_to_world_to_camera(euler: np.ndarray) -> np.ndarray:
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


def non_comment_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def find_cam_file(root: Path) -> Path:
    cam = root / "Cam.txt"
    if cam.is_file():
        return cam
    matches = sorted(root.glob("Cam*.txt"))
    if not matches:
        raise FileNotFoundError(f"No Cam*.txt found under {root}")
    return matches[0]


def read_calibration(path: Path) -> Tuple[float, float, float, float]:
    rows = []
    for line in non_comment_lines(path):
        values = [float(x) for x in line.split()]
        if values:
            rows.append(values)
    if len(rows) < 2 or len(rows[0]) < 3 or len(rows[1]) < 3:
        raise ValueError(f"Unsupported calibration file: {path}")
    return rows[0][0], rows[1][1], rows[0][2], rows[1][2]


def read_cameras(path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    cameras: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for row_index, line in enumerate(non_comment_lines(path)):
        values = [float(x) for x in line.split()]
        if len(values) < 6:
            continue
        euler = np.array(values[:3], dtype=float)
        center = np.array(values[3:6], dtype=float)
        # PVL-BA GCP observations use zero-based image row indices. The last
        # Cam*.txt column is the calibration/photo-group id, not the image id.
        cameras[row_index] = (euler, center)
    if not cameras:
        raise ValueError(f"No cameras parsed from {path}")
    return cameras


def read_gcp(path: Path) -> Dict[int, dict]:
    points: Dict[int, dict] = {}
    for line in non_comment_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        point_id = int(parts[0])
        points[point_id] = {
            "name": parts[1],
            "xyz": np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=float),
            "is_checkpoint": int(parts[7]) == 1,
        }
    if not points:
        raise ValueError(f"No GCP records parsed from {path}")
    return points


def read_gcp_observations(path: Path) -> Dict[int, List[Tuple[int, float, float]]]:
    observations: Dict[int, List[Tuple[int, float, float]]] = {}
    for line in non_comment_lines(path):
        parts = line.split()
        if len(parts) < 2:
            continue
        point_id = int(parts[0])
        count = int(parts[1])
        triples = parts[2:]
        obs: List[Tuple[int, float, float]] = []
        for i in range(min(count, len(triples) // 3)):
            image_index = int(triples[3 * i])
            u = float(triples[3 * i + 1])
            v = float(triples[3 * i + 2])
            obs.append((image_index, u, v))
        observations[point_id] = obs
    if not observations:
        raise ValueError(f"No GCP observations parsed from {path}")
    return observations


def triangulate_point(
    obs: List[Tuple[int, float, float]],
    cameras: Dict[int, Tuple[np.ndarray, np.ndarray]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, int]:
    normal_matrix = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    used = 0
    for image_index, u, v in obs:
        camera = cameras.get(image_index)
        if camera is None:
            continue
        euler, center = camera
        rotation = euler_to_world_to_camera(euler)
        ray_camera = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
        ray_world = rotation.T @ ray_camera
        ray_world /= np.linalg.norm(ray_world)
        projector = np.eye(3) - np.outer(ray_world, ray_world)
        normal_matrix += projector
        rhs += projector @ center
        used += 1
    if used < 2:
        raise ValueError("Need at least two usable observations to triangulate a point")
    return np.linalg.solve(normal_matrix, rhs), used


def reprojection_rmse(
    xyz: np.ndarray,
    obs: List[Tuple[int, float, float]],
    cameras: Dict[int, Tuple[np.ndarray, np.ndarray]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> float:
    residuals = []
    for image_index, u, v in obs:
        camera = cameras.get(image_index)
        if camera is None:
            continue
        euler, center = camera
        point_camera = euler_to_world_to_camera(euler) @ (xyz - center)
        if abs(point_camera[2]) < 1e-12:
            continue
        predicted_u = fx * point_camera[0] / point_camera[2] + cx
        predicted_v = fy * point_camera[1] / point_camera[2] + cy
        residuals.extend([predicted_u - u, predicted_v - v])
    if not residuals:
        return float("nan")
    return math.sqrt(float(np.mean(np.square(residuals))))


def fit_similarity(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Need matching source/target arrays with at least three points")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def evaluate_run(run_root: Path, base_root: Path) -> dict:
    cam_path = find_cam_file(run_root)
    fx, fy, cx, cy = read_calibration(run_root / "cal.txt")
    cameras = read_cameras(cam_path)
    gcps = read_gcp(run_root / "gcp.txt")
    observations = read_gcp_observations(run_root / "gcp_observations.txt")

    triangulated: Dict[int, np.ndarray] = {}
    obs_counts: Dict[int, int] = {}
    reprojection: Dict[int, float] = {}
    for point_id, obs in observations.items():
        xyz, used = triangulate_point(obs, cameras, fx, fy, cx, cy)
        triangulated[point_id] = xyz
        obs_counts[point_id] = used
        reprojection[point_id] = reprojection_rmse(xyz, obs, cameras, fx, fy, cx, cy)

    control_ids = sorted(pid for pid, record in gcps.items() if not record["is_checkpoint"])
    checkpoint_ids = sorted(pid for pid, record in gcps.items() if record["is_checkpoint"])
    available_control_ids = [pid for pid in control_ids if pid in triangulated]
    available_checkpoint_ids = [pid for pid in checkpoint_ids if pid in triangulated]
    if len(available_control_ids) < 3:
        raise ValueError(f"{run_root}: fewer than three control points could be triangulated")
    if not available_checkpoint_ids:
        raise ValueError(f"{run_root}: no checkpoints could be triangulated")

    control_source = np.stack([triangulated[pid] for pid in available_control_ids], axis=0)
    control_target = np.stack([gcps[pid]["xyz"] for pid in available_control_ids], axis=0)
    scale, rotation, translation = fit_similarity(control_source, control_target)

    aligned_control = apply_similarity(control_source, scale, rotation, translation)
    control_errors = np.linalg.norm(aligned_control - control_target, axis=1)

    checkpoint_source = np.stack([triangulated[pid] for pid in available_checkpoint_ids], axis=0)
    checkpoint_target = np.stack([gcps[pid]["xyz"] for pid in available_checkpoint_ids], axis=0)
    raw_checkpoint_errors = np.linalg.norm(checkpoint_source - checkpoint_target, axis=1)
    aligned_checkpoint = apply_similarity(checkpoint_source, scale, rotation, translation)
    checkpoint_errors = np.linalg.norm(aligned_checkpoint - checkpoint_target, axis=1)

    run_name = "original" if run_root == base_root / "original" else run_root.name
    return {
        "base_dataset": base_root.name,
        "run": run_name,
        "camera_file": cam_path.name,
        "control_ids": " ".join(map(str, available_control_ids)),
        "checkpoint_ids": " ".join(map(str, available_checkpoint_ids)),
        "control_count": len(available_control_ids),
        "checkpoint_count": len(available_checkpoint_ids),
        "mean_gcp_obs": float(np.mean([obs_counts[pid] for pid in sorted(triangulated)])),
        "mean_gcp_reproj_rmse_px": float(np.mean([reprojection[pid] for pid in sorted(triangulated)])),
        "similarity_scale": scale,
        "control_rmse_m": math.sqrt(float(np.mean(control_errors * control_errors))),
        "control_max_m": float(np.max(control_errors)),
        "checkpoint_raw_rmse_m": math.sqrt(float(np.mean(raw_checkpoint_errors * raw_checkpoint_errors))),
        "checkpoint_rmse_m": math.sqrt(float(np.mean(checkpoint_errors * checkpoint_errors))),
        "checkpoint_mean_m": float(np.mean(checkpoint_errors)),
        "checkpoint_max_m": float(np.max(checkpoint_errors)),
        "checkpoint_errors_m": " ".join(f"{pid}:{err:.6f}" for pid, err in zip(available_checkpoint_ids, checkpoint_errors)),
    }


def discover_runs(base_root: Path, quality_filter: str | None) -> List[Path]:
    runs = [base_root / "original"]
    quality_root = base_root / "quality"
    if quality_root.is_dir():
        for run in sorted(quality_root.iterdir()):
            if not run.is_dir():
                continue
            if quality_filter and quality_filter not in run.name:
                continue
            runs.append(run)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_root", type=Path, help="PVL-BA problem directory containing original/quality")
    parser.add_argument("--quality-filter", help="Only evaluate quality runs whose name contains this text")
    parser.add_argument("--out", type=Path, help="CSV output path")
    args = parser.parse_args()

    rows = [evaluate_run(run, args.problem_root) for run in discover_runs(args.problem_root, args.quality_filter)]
    fieldnames = list(rows[0].keys())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
