#!/usr/bin/env python3
"""Generate mechanism-level reproductions of the IJRR ParallaxBA simulations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from generate_synthetic_benchmark import (
    compute_parallax_statistics,
    euler_to_rotation,
    hardlink_or_copy,
    project,
    write_calibration,
    write_cameras,
    write_points,
)


INTRINSICS = (400.0, 400.0, 400.0, 400.0)
IMAGE_SIZE = (800.0, 800.0)


@dataclass(frozen=True)
class Experiment:
    name: str
    paper_simulation: str
    stress: str
    seed: int
    scale_upper: float = 1.0


def regular_grid(
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    resolution: tuple[int, int, int],
) -> np.ndarray:
    xs = np.linspace(*x_limits, resolution[0])
    ys = np.linspace(*y_limits, resolution[1])
    zs = np.linspace(*z_limits, resolution[2])
    return np.asarray(
        [(x, y, z) for x in xs for y in ys for z in zs],
        dtype=np.float64,
    )


def circle_centers(camera_count: int, step_distance: float) -> np.ndarray:
    radius = step_distance / (2.0 * math.sin(math.pi / camera_count))
    angles = np.linspace(0.0, 2.0 * math.pi, camera_count, endpoint=False)
    centers = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros(camera_count))
    )
    return centers - centers[0]


def line_centers(camera_count: int, step_distance: float) -> np.ndarray:
    centers = np.zeros((camera_count, 3), dtype=np.float64)
    centers[:, 2] = np.arange(camera_count, dtype=np.float64) * step_distance
    return centers


def square_centers(camera_count: int, step_distance: float) -> np.ndarray:
    increments = camera_count - 1
    side_counts = [increments // 4] * 4
    for index in range(increments % 4):
        side_counts[index] += 1
    directions = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
         (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    centers = [np.zeros(3, dtype=np.float64)]
    current = centers[0].copy()
    for direction, count in zip(directions, side_counts):
        for _ in range(count):
            current = current + step_distance * direction
            centers.append(current.copy())
    return np.asarray(centers, dtype=np.float64)


def scaled_relative_trajectory(
    centers: np.ndarray,
    scale_upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if scale_upper <= 1.0:
        return centers.copy()
    result = np.zeros_like(centers)
    result[0] = centers[0]
    for index in range(1, len(centers)):
        relative = centers[index] - centers[index - 1]
        result[index] = (
            result[index - 1]
            + rng.uniform(1.0, scale_upper) * relative
        )
    return result


def visible_track(
    point: np.ndarray,
    centers: np.ndarray,
    eulers: np.ndarray,
) -> np.ndarray:
    visible: list[int] = []
    for camera_id, (center, euler) in enumerate(zip(centers, eulers)):
        camera_point = euler_to_rotation(euler) @ (point - center)
        if camera_point[2] <= 1e-8:
            continue
        u = INTRINSICS[0] * camera_point[0] / camera_point[2] + INTRINSICS[2]
        v = INTRINSICS[1] * camera_point[1] / camera_point[2] + INTRINSICS[3]
        if 0.0 <= u < IMAGE_SIZE[0] and 0.0 <= v < IMAGE_SIZE[1]:
            visible.append(camera_id)
    return np.asarray(visible, dtype=np.int32)


def pixel_ray_world(
    u: float,
    v: float,
    euler: np.ndarray,
) -> np.ndarray:
    ray_camera = np.asarray(
        (
            (u - INTRINSICS[2]) / INTRINSICS[0],
            (v - INTRINSICS[3]) / INTRINSICS[1],
            1.0,
        ),
        dtype=np.float64,
    )
    ray_world = euler_to_rotation(euler).T @ ray_camera
    return ray_world / np.linalg.norm(ray_world)


def maximum_parallax_pair(
    camera_ids: np.ndarray,
    observations: dict[int, tuple[float, float]],
    centers: np.ndarray,
    eulers: np.ndarray,
) -> tuple[int, int]:
    rays = np.asarray(
        [
            pixel_ray_world(
                *observations[int(camera_id)],
                eulers[int(camera_id)],
            )
            for camera_id in camera_ids
        ],
        dtype=np.float64,
    )
    cosine = np.clip(rays @ rays.T, -1.0, 1.0)
    cosine[np.tril_indices(len(camera_ids))] = 2.0
    first_index, second_index = np.unravel_index(
        int(np.argmin(cosine)),
        cosine.shape,
    )
    return int(camera_ids[first_index]), int(camera_ids[second_index])


def triangulate_midpoint(
    first: int,
    second: int,
    observations: dict[int, tuple[float, float]],
    centers: np.ndarray,
    eulers: np.ndarray,
) -> np.ndarray:
    ray_first = pixel_ray_world(*observations[first], eulers[first])
    ray_second = pixel_ray_world(*observations[second], eulers[second])
    design = np.column_stack((ray_first, -ray_second))
    rhs = centers[second] - centers[first]
    depths, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
    point_first = centers[first] + depths[0] * ray_first
    point_second = centers[second] + depths[1] * ray_second
    return 0.5 * (point_first + point_second)


def make_problem(
    experiment: Experiment,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[dict[int, tuple[float, float]]],
    np.ndarray,
]:
    rng = np.random.default_rng(20260710 + experiment.seed)
    observation_rng = np.random.default_rng(20261710 + experiment.seed)
    eulers: np.ndarray

    if experiment.paper_simulation == "Simulation-1":
        centers = circle_centers(23, 5.0)
        eulers = np.zeros((len(centers), 3), dtype=np.float64)
        distant = regular_grid(
            (-5000.0, 5000.0), (-5000.0, 5000.0), (9995.0, 10005.0),
            (10, 10, 2),
        )
        nearby = regular_grid(
            (-25.0, 25.0), (-25.0, 25.0), (45.0, 55.0),
            (20, 20, 4),
        )
        candidate_points = np.vstack((distant, nearby))
        special_two_view_count = 0
    elif experiment.paper_simulation == "Simulation-2":
        centers = line_centers(21, 2.0)
        eulers = np.zeros((len(centers), 3), dtype=np.float64)
        nearby = regular_grid(
            (-25.0, 25.0), (-25.0, 25.0), (55.0, 75.0),
            (20, 20, 4),
        )
        axial = np.column_stack(
            (
                np.zeros(5),
                np.zeros(5),
                np.linspace(55.0, 75.0, 5),
            )
        )
        candidate_points = np.vstack((axial, nearby))
        special_two_view_count = len(axial)
    elif experiment.paper_simulation == "Simulation-3":
        centers = square_centers(66, 2.5)
        eulers = np.zeros((len(centers), 3), dtype=np.float64)
        distant = regular_grid(
            (-5000.0, 5000.0), (-5000.0, 5000.0), (9995.0, 10005.0),
            (10, 10, 2),
        )
        nearby = regular_grid(
            (-25.0, 25.0), (-25.0, 25.0), (55.0, 65.0),
            (20, 20, 4),
        )
        candidate_points = np.vstack((distant, nearby))
        special_two_view_count = 0
    else:
        raise ValueError(experiment.paper_simulation)

    initial_centers = scaled_relative_trajectory(
        centers,
        experiment.scale_upper,
        rng,
    )
    initial_eulers = eulers.copy()

    points: list[np.ndarray] = []
    tracks: list[np.ndarray] = []
    all_observations: list[dict[int, tuple[float, float]]] = []
    for point_index, point in enumerate(candidate_points):
        if point_index < special_two_view_count:
            camera_ids = np.asarray((0, 1), dtype=np.int32)
        else:
            camera_ids = visible_track(point, centers, eulers)
        if len(camera_ids) < 2:
            continue

        observations: dict[int, tuple[float, float]] = {}
        for camera_id in camera_ids:
            u, v = project(
                point,
                centers[int(camera_id)],
                eulers[int(camera_id)],
                INTRINSICS,
            )
            observations[int(camera_id)] = (
                u + float(observation_rng.normal(0.0, 0.1)),
                v + float(observation_rng.normal(0.0, 0.1)),
            )
        points.append(point)
        tracks.append(camera_ids)
        all_observations.append(observations)

    true_points = np.asarray(points, dtype=np.float64)
    initial_points: list[np.ndarray] = []
    for point, camera_ids, observations in zip(
        true_points,
        tracks,
        all_observations,
    ):
        first, second = maximum_parallax_pair(
            camera_ids,
            observations,
            initial_centers,
            initial_eulers,
        )
        estimate = triangulate_midpoint(
            first,
            second,
            observations,
            initial_centers,
            initial_eulers,
        )
        if not np.all(np.isfinite(estimate)):
            estimate = point.copy()
        initial_points.append(estimate)

    return (
        centers,
        eulers,
        true_points,
        initial_centers,
        initial_eulers,
        tracks,
        all_observations,
        np.asarray(initial_points, dtype=np.float64),
    )


def write_observations(
    path: Path,
    tracks: list[np.ndarray],
    observations: list[dict[int, tuple[float, float]]],
) -> int:
    count = 0
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for camera_ids, point_observations in zip(tracks, observations):
            values = [str(len(camera_ids))]
            for camera_id in camera_ids:
                u, v = point_observations[int(camera_id)]
                values.extend((str(int(camera_id)), f"{u:.15f}", f"{v:.15f}"))
                count += 1
            stream.write(" ".join(values) + "\n")
    return count


def generate_experiment(output: Path, experiment: Experiment) -> dict[str, object]:
    base_root = output / experiment.name
    original = base_root / "original"
    quality_name = experiment.name + "-quality"
    quality = base_root / "quality" / quality_name
    original.mkdir(parents=True)
    quality.mkdir(parents=True)

    (
        centers,
        eulers,
        points,
        initial_centers,
        initial_eulers,
        tracks,
        observations,
        initial_points,
    ) = make_problem(experiment)

    camera_name = f"Cam-{len(centers)}-.txt"
    write_cameras(original / camera_name, eulers, centers)
    write_points(original / "XYZ.txt", points)
    write_calibration(original / "cal.txt", INTRINSICS)
    observation_count = write_observations(
        original / "Feature.txt",
        tracks,
        observations,
    )

    write_cameras(quality / camera_name, initial_eulers, initial_centers)
    write_points(quality / "XYZ.txt", initial_points)
    hardlink_or_copy(original / "cal.txt", quality / "cal.txt")
    hardlink_or_copy(original / "Feature.txt", quality / "Feature.txt")

    point_error = np.linalg.norm(initial_points - points, axis=1)
    parallax = compute_parallax_statistics(points, centers, tracks)
    metadata: dict[str, object] = {
        "base_dataset": experiment.name,
        "quality_dataset": quality_name,
        "paper_simulation": experiment.paper_simulation,
        "stress": experiment.stress,
        "seed": experiment.seed,
        "scale_upper": experiment.scale_upper,
        "cameras": len(centers),
        "points": len(points),
        "observations": observation_count,
        "image_noise_px": 0.1,
        "initial_point_error_median": float(np.median(point_error)),
        "initial_point_error_p90": float(np.percentile(point_error, 90)),
        "initial_point_error_max": float(np.max(point_error)),
        **parallax,
    }
    (original / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )
    (quality / "noise_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="ascii",
    )
    return metadata


def experiments(replicates: int) -> list[Experiment]:
    result: list[Experiment] = []
    for seed in range(replicates):
        result.extend(
            (
                Experiment(
                    f"IJRR-S1-distant-s{seed:02d}",
                    "Simulation-1",
                    "distant-features",
                    seed,
                ),
                Experiment(
                    f"IJRR-S2-forward-two-view-s{seed:02d}",
                    "Simulation-2",
                    "forward-motion-two-view",
                    seed,
                ),
                Experiment(
                    f"IJRR-S3-scale1p2-s{seed:02d}",
                    "Simulation-3",
                    "relative-scale-1.2",
                    seed,
                    1.2,
                ),
                Experiment(
                    f"IJRR-S3-scale1p5-s{seed:02d}",
                    "Simulation-3",
                    "relative-scale-1.5",
                    seed,
                    1.5,
                ),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\zuo\projects\CEP\parallaxba_ijrr_reproduction\datasets"),
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"Output exists; use --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    rows = [
        generate_experiment(output, experiment)
        for experiment in experiments(args.replicates)
    ]
    manifest = output.parent / "scenario_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output.parent / "generation_config.json").write_text(
        json.dumps(
            {
                "source": "Zhao et al., ParallaxBA, IJRR",
                "reproduction_type": "mechanism-level",
                "replicates": args.replicates,
                "scenario_count": len(rows),
                "intrinsics": INTRINSICS,
                "image_noise_px": 0.1,
            },
            indent=2,
        ),
        encoding="ascii",
    )
    print(f"Generated {len(rows)} IJRR reproduction datasets")
    print(f"Datasets: {output}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
