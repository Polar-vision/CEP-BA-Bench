#!/usr/bin/env python3
"""Generate controlled BA problems in the CEP/PVL directory layout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Scenario:
    family: str
    trajectory: str
    target_parallax_deg: float
    depth_scale: float
    rotation_noise_deg: float
    translation_noise_ratio: float
    track_length: int
    coordinate_offset: float
    seed: int
    pose_level: str = "none"


def token(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def euler_to_rotation(euler: np.ndarray) -> np.ndarray:
    ey, ex, ez = euler
    c1, c2, c3 = np.cos([ey, ex, ez])
    s1, s2, s3 = np.sin([ey, ex, ez])
    return np.array(
        [
            [c1 * c3 - s1 * s2 * s3, c2 * s3, s1 * c3 + c1 * s2 * s3],
            [-c1 * s3 - s1 * s2 * c3, c2 * c3, -s1 * s3 + c1 * s2 * c3],
            [-s1 * c2, -s2, c1 * c2],
        ],
        dtype=np.float64,
    )


def project(
    point: np.ndarray,
    center: np.ndarray,
    euler: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[float, float]:
    fx, fy, cx, cy = intrinsics
    camera_point = euler_to_rotation(euler) @ (point - center)
    if camera_point[2] <= 1e-9:
        raise ValueError("Point is behind a camera")
    return (
        fx * camera_point[0] / camera_point[2] + cx,
        fy * camera_point[1] / camera_point[2] + cy,
    )


def make_scenarios(n_cameras: int) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seeds = range(3)
    full_track = n_cameras

    for trajectory in ("lateral", "forward"):
        for alpha in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            for depth_scale in (1.2, 2.0, 5.0, 10.0):
                for seed in seeds:
                    scenarios.append(
                        Scenario(
                            family="parallax_depth",
                            trajectory=trajectory,
                            target_parallax_deg=alpha,
                            depth_scale=depth_scale,
                            rotation_noise_deg=0.0,
                            translation_noise_ratio=0.0,
                            track_length=full_track,
                            coordinate_offset=0.0,
                            seed=seed,
                        )
                    )

    pose_levels = (
        ("mild", 0.5, 0.01),
        ("moderate", 3.0, 0.10),
        ("severe", 10.0, 0.50),
    )
    for trajectory in ("lateral", "forward"):
        for alpha in (0.1, 1.0, 10.0):
            for pose_level, rotation_noise, translation_noise in pose_levels:
                for seed in range(2):
                    scenarios.append(
                        Scenario(
                            family="pose_stress",
                            trajectory=trajectory,
                            target_parallax_deg=alpha,
                            depth_scale=1.5,
                            rotation_noise_deg=rotation_noise,
                            translation_noise_ratio=translation_noise,
                            track_length=full_track,
                            coordinate_offset=0.0,
                            seed=seed,
                            pose_level=pose_level,
                        )
                    )

    track_lengths = sorted({2, 3, 5, min(8, n_cameras), n_cameras})
    for alpha in (0.1, 1.0, 10.0):
        for track_length in track_lengths:
            for depth_scale in (2.0, 10.0):
                scenarios.append(
                    Scenario(
                        family="track_stress",
                        trajectory="lateral",
                        target_parallax_deg=alpha,
                        depth_scale=depth_scale,
                        rotation_noise_deg=0.0,
                        translation_noise_ratio=0.0,
                        track_length=track_length,
                        coordinate_offset=0.0,
                        seed=0,
                    )
                )

    for coordinate_offset in (0.0, 1e3, 1e6, 1e9):
        for seed in range(2):
            scenarios.append(
                Scenario(
                    family="coordinate_stress",
                    trajectory="lateral",
                    target_parallax_deg=1.0,
                    depth_scale=3.0,
                    rotation_noise_deg=0.0,
                    translation_noise_ratio=0.0,
                    track_length=full_track,
                    coordinate_offset=coordinate_offset,
                    seed=seed,
                )
            )
    return scenarios


def make_scale_validation_scenarios(n_cameras: int) -> list[Scenario]:
    return [
        Scenario(
            family="scale_validation",
            trajectory="lateral",
            target_parallax_deg=0.03,
            depth_scale=10.0,
            rotation_noise_deg=0.0,
            translation_noise_ratio=0.0,
            track_length=n_cameras,
            coordinate_offset=0.0,
            seed=0,
            pose_level="small-parallax",
        ),
        Scenario(
            family="scale_validation",
            trajectory="forward",
            target_parallax_deg=0.03,
            depth_scale=10.0,
            rotation_noise_deg=0.0,
            translation_noise_ratio=0.0,
            track_length=n_cameras,
            coordinate_offset=0.0,
            seed=0,
            pose_level="small-parallax",
        ),
        Scenario(
            family="scale_validation",
            trajectory="forward",
            target_parallax_deg=10.0,
            depth_scale=10.0,
            rotation_noise_deg=0.0,
            translation_noise_ratio=0.0,
            track_length=n_cameras,
            coordinate_offset=0.0,
            seed=2,
            pose_level="large-depth-error",
        ),
        Scenario(
            family="scale_validation",
            trajectory="lateral",
            target_parallax_deg=0.1,
            depth_scale=10.0,
            rotation_noise_deg=0.0,
            translation_noise_ratio=0.0,
            track_length=2,
            coordinate_offset=0.0,
            seed=0,
            pose_level="short-track",
        ),
        Scenario(
            family="scale_validation",
            trajectory="lateral",
            target_parallax_deg=1.0,
            depth_scale=3.0,
            rotation_noise_deg=0.0,
            translation_noise_ratio=0.0,
            track_length=n_cameras,
            coordinate_offset=1e9,
            seed=0,
            pose_level="large-coordinate",
        ),
        Scenario(
            family="scale_validation",
            trajectory="forward",
            target_parallax_deg=0.1,
            depth_scale=1.5,
            rotation_noise_deg=10.0,
            translation_noise_ratio=0.5,
            track_length=n_cameras,
            coordinate_offset=0.0,
            seed=1,
            pose_level="severe-pose",
        ),
    ]


def make_geometry(
    scenario: Scenario,
    n_cameras: int,
    n_points: int,
    depth: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    alpha = math.radians(scenario.target_parallax_deg)
    lateral_span = max(2.0 * depth * math.tan(alpha / 2.0), depth * 1e-6)
    t = np.linspace(-0.5, 0.5, n_cameras)

    centers = np.zeros((n_cameras, 3), dtype=np.float64)
    eulers = np.zeros((n_cameras, 3), dtype=np.float64)
    if scenario.trajectory == "lateral":
        centers[:, 0] = lateral_span * t
    elif scenario.trajectory == "forward":
        centers[:, 0] = lateral_span * t
        centers[:, 2] = depth * 0.25 * (t + 0.5)
    else:
        raise ValueError(f"Unknown trajectory: {scenario.trajectory}")

    points = np.column_stack(
        [
            rng.uniform(-0.10 * depth, 0.10 * depth, n_points),
            rng.uniform(-0.075 * depth, 0.075 * depth, n_points),
            depth * rng.uniform(0.90, 1.10, n_points),
        ]
    )

    if scenario.coordinate_offset != 0.0:
        offset = np.array(
            [
                scenario.coordinate_offset,
                -0.3 * scenario.coordinate_offset,
                0.2 * scenario.coordinate_offset,
            ],
            dtype=np.float64,
        )
        centers += offset
        points += offset
    return centers, eulers, points, lateral_span


def make_tracks(
    n_points: int,
    n_cameras: int,
    track_length: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    length = min(max(track_length, 2), n_cameras)
    tracks: list[np.ndarray] = []
    for _ in range(n_points):
        if length == n_cameras:
            cameras = np.arange(n_cameras, dtype=np.int32)
        else:
            start = int(rng.integers(0, n_cameras - length + 1))
            cameras = np.arange(start, start + length, dtype=np.int32)
        tracks.append(cameras)
    return tracks


def compute_parallax_statistics(
    points: np.ndarray,
    centers: np.ndarray,
    tracks: list[np.ndarray],
) -> dict[str, float]:
    max_angles: list[float] = []
    for point, camera_ids in zip(points, tracks):
        rays = point[None, :] - centers[camera_ids]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        maximum = 0.0
        for first, second in combinations(range(len(camera_ids)), 2):
            cosine = float(np.clip(np.dot(rays[first], rays[second]), -1.0, 1.0))
            maximum = max(maximum, math.degrees(math.acos(cosine)))
        max_angles.append(maximum)
    values = np.asarray(max_angles, dtype=np.float64)
    return {
        "parallax_min_deg": float(np.min(values)),
        "parallax_p10_deg": float(np.percentile(values, 10)),
        "parallax_median_deg": float(np.median(values)),
        "parallax_p90_deg": float(np.percentile(values, 90)),
        "parallax_max_deg": float(np.max(values)),
    }


def perturb_initial_state(
    scenario: Scenario,
    centers: np.ndarray,
    eulers: np.ndarray,
    points: np.ndarray,
    lateral_span: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_centers = centers.copy()
    initial_eulers = eulers.copy()
    initial_points = points.copy()

    reference = np.mean(centers, axis=0)
    initial_points = reference + scenario.depth_scale * (initial_points - reference)

    if scenario.rotation_noise_deg > 0.0:
        rotation_noise = rng.normal(
            0.0,
            math.radians(scenario.rotation_noise_deg),
            size=initial_eulers.shape,
        )
        initial_eulers += rotation_noise
    if scenario.translation_noise_ratio > 0.0:
        baseline = max(lateral_span, 1e-6)
        initial_centers += rng.normal(
            0.0,
            scenario.translation_noise_ratio * baseline,
            size=initial_centers.shape,
        )
    return initial_centers, initial_eulers, initial_points


def write_calibration(path: Path, intrinsics: tuple[float, float, float, float]) -> None:
    fx, fy, cx, cy = intrinsics
    path.write_text(
        f"{fx:.15f} 0 {cx:.15f}\n"
        f"0 {fy:.15f} {cy:.15f}\n"
        "0 0 1\n",
        encoding="ascii",
    )


def write_cameras(path: Path, eulers: np.ndarray, centers: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for euler, center in zip(eulers, centers):
            stream.write(
                " ".join(f"{value:.15f}" for value in (*euler, *center))
                + " 1\n"
            )


def write_points(path: Path, points: np.ndarray) -> None:
    np.savetxt(path, points, fmt="%.15f")


def write_features(
    path: Path,
    points: np.ndarray,
    centers: np.ndarray,
    eulers: np.ndarray,
    tracks: list[np.ndarray],
    intrinsics: tuple[float, float, float, float],
    image_noise_px: float,
    rng: np.random.Generator,
) -> int:
    observation_count = 0
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for point, camera_ids in zip(points, tracks):
            values: list[str] = [str(len(camera_ids))]
            for camera_id in camera_ids:
                u, v = project(
                    point,
                    centers[camera_id],
                    eulers[camera_id],
                    intrinsics,
                )
                if image_noise_px > 0.0:
                    u += float(rng.normal(0.0, image_noise_px))
                    v += float(rng.normal(0.0, image_noise_px))
                values.extend((str(int(camera_id)), f"{u:.15f}", f"{v:.15f}"))
                observation_count += 1
            stream.write(" ".join(values) + "\n")
    return observation_count


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def scenario_name(index: int, scenario: Scenario) -> str:
    return (
        f"SYN-{index:04d}-{scenario.family}-{scenario.trajectory}"
        f"-a{token(scenario.target_parallax_deg)}"
        f"-d{token(scenario.depth_scale)}"
        f"-r{token(scenario.rotation_noise_deg)}"
        f"-t{token(scenario.translation_noise_ratio)}"
        f"-l{scenario.track_length}"
        f"-o{token(scenario.coordinate_offset)}"
        f"-s{scenario.seed}"
    )


def generate_dataset(
    output_root: Path,
    index: int,
    scenario: Scenario,
    n_cameras: int,
    n_points: int,
    depth: float,
    image_noise_px: float,
) -> dict[str, object]:
    name = scenario_name(index, scenario)
    base_root = output_root / name
    original_root = base_root / "original"
    quality_name = name + "-quality"
    quality_root = base_root / "quality" / quality_name
    original_root.mkdir(parents=True, exist_ok=False)
    quality_root.mkdir(parents=True, exist_ok=False)

    geometry_rng = np.random.default_rng(10_000 + scenario.seed)
    observation_rng = np.random.default_rng(20_000 + scenario.seed)
    perturbation_rng = np.random.default_rng(30_000 + scenario.seed)
    centers, eulers, points, lateral_span = make_geometry(
        scenario,
        n_cameras,
        n_points,
        depth,
        geometry_rng,
    )
    tracks = make_tracks(
        n_points,
        n_cameras,
        scenario.track_length,
        geometry_rng,
    )
    initial_centers, initial_eulers, initial_points = perturb_initial_state(
        scenario,
        centers,
        eulers,
        points,
        lateral_span,
        perturbation_rng,
    )

    intrinsics = (1200.0, 1200.0, 1000.0, 750.0)
    camera_name = f"Cam-{n_cameras}-.txt"
    write_cameras(original_root / camera_name, eulers, centers)
    write_points(original_root / "XYZ.txt", points)
    write_calibration(original_root / "cal.txt", intrinsics)
    observations = write_features(
        original_root / "Feature.txt",
        points,
        centers,
        eulers,
        tracks,
        intrinsics,
        image_noise_px,
        observation_rng,
    )

    write_cameras(quality_root / camera_name, initial_eulers, initial_centers)
    write_points(quality_root / "XYZ.txt", initial_points)
    hardlink_or_copy(original_root / "cal.txt", quality_root / "cal.txt")
    hardlink_or_copy(original_root / "Feature.txt", quality_root / "Feature.txt")

    parallax = compute_parallax_statistics(points, centers, tracks)
    metadata: dict[str, object] = {
        "dataset_name": quality_name,
        "synthetic": True,
        "images": n_cameras,
        "points": n_points,
        "observations": observations,
        "image_noise_px": image_noise_px,
        "scene_depth": depth,
        "lateral_span": lateral_span,
        **asdict(scenario),
        **parallax,
    }
    (original_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )
    (quality_root / "noise_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )
    return {
        "base_dataset": name,
        "quality_dataset": quality_name,
        "base_root": str(base_root),
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\zuo\projects\CEP\synthetic_benchmark\datasets"),
    )
    parser.add_argument("--cameras", type=int, default=12)
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--depth", type=float, default=1000.0)
    parser.add_argument("--image-noise", type=float, default=0.5)
    parser.add_argument(
        "--profile",
        choices=("full", "scale-validation"),
        default="full",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists; use --force to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    scenarios = (
        make_scenarios(args.cameras)
        if args.profile == "full"
        else make_scale_validation_scenarios(args.cameras)
    )
    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios, start=1):
        rows.append(
            generate_dataset(
                output,
                index,
                scenario,
                args.cameras,
                args.points,
                args.depth,
                args.image_noise,
            )
        )
        if index % 25 == 0 or index == len(scenarios):
            print(f"Generated {index}/{len(scenarios)} scenarios", flush=True)

    manifest = output.parent / "scenario_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output.parent / "generation_config.json").write_text(
        json.dumps(
            {
                "scenario_count": len(scenarios),
                "method_run_count": len(scenarios) * 14,
                "cameras": args.cameras,
                "points": args.points,
                "depth": args.depth,
                "image_noise_px": args.image_noise,
                "profile": args.profile,
            },
            indent=2,
        ),
        encoding="ascii",
    )
    print(f"Datasets: {output}")
    print(f"Manifest: {manifest}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Planned method runs: {len(scenarios) * 14}")


if __name__ == "__main__":
    main()
