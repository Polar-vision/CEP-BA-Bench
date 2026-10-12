#!/usr/bin/env python3
"""Exercise controlled BA, checkpoint evaluation, pose export, and resume.

This creates a tiny deterministic prepared network, not a substitute for the
paper's D144 results. Requires a built benchmark executable and NumPy.

    python tools/test_d144_runner.py --executable build/example_v2/Release/example.exe
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


BASE_NAME = "Synthetic__D144_control"
RUN_NAME = "Initial Value"
METHODS = ("A0-XYZ-W", "A2-Parallax-Mc")
CAMERA_CENTERS = np.array(
    [[-1.0, -0.6, 0.0], [1.0, -0.6, 0.0], [-1.0, 0.6, 0.1], [1.0, 0.6, -0.1]]
)
TIE_POINT_COUNT = 24
CONTROL_COUNT = 13
CHECKPOINT_COUNT = 10


def write_rows(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def numbers(values) -> str:
    return " ".join(f"{float(value):.15g}" for value in values)


def project(point: np.ndarray, camera_index: int) -> tuple[float, float]:
    relative = point - CAMERA_CENTERS[camera_index]
    return 800.0 * relative[0] / relative[2], 800.0 * relative[1] / relative[2]


def make_network(data_root: Path) -> Path:
    """Use a prepared Initial Value run to also detect duplicate discovery."""
    block = data_root / BASE_NAME
    tie_points = np.array(
        [
            [-1.5 + 0.6 * (i % 6), -0.8 + 0.5 * (i // 6), 6.0 + 0.3 * (i % 7)]
            for i in range(TIE_POINT_COUNT)
        ]
    )
    measured_points = np.array(
        [
            [-1.8 + 0.65 * (i % 6), -1.1 + 0.55 * ((i // 6) % 4), 6.4 + 0.4 * (i % 5)]
            for i in range(CONTROL_COUNT + CHECKPOINT_COUNT)
        ]
    )
    feature_rows = []
    for point in tie_points:
        observations = [
            f"{camera_index} {numbers(project(point, camera_index))}"
            for camera_index in range(len(CAMERA_CENTERS))
        ]
        feature_rows.append(f"{len(observations)} " + " ".join(observations))

    for is_initial, directory in (
        (False, block / "original"),
        (True, block / "quality" / RUN_NAME),
    ):
        directory.mkdir(parents=True)
        camera_rows = []
        for i, center in enumerate(CAMERA_CENTERS):
            angles = np.zeros(3)
            input_center = center.copy()
            if is_initial:
                angles += [0.002 * (i + 1), -0.0015 * (i + 1), 0.001 * (i + 1)]
                input_center += [0.025 * (i + 1), -0.02 * (i + 1), 0.015]
            camera_rows.append(numbers(np.concatenate((angles, input_center))) + " 1")
        write_rows(directory / "Cam.txt", camera_rows)
        write_rows(directory / "cal.txt", ["800 0 0", "0 800 0", "0 0 1"])
        write_rows(directory / "Feature.txt", feature_rows)
        xyz = tie_points + np.array([0.04, -0.03, 0.06]) if is_initial else tie_points
        write_rows(directory / "XYZ.txt", [numbers(point) for point in xyz])
        gcp_rows = []
        observation_rows = []
        for point_id, point in enumerate(measured_points):
            checkpoint = int(point_id >= CONTROL_COUNT)
            gcp_rows.append(f"{point_id} P{point_id} {numbers(point)} 0.01 0.02 {checkpoint}")
            observations = [
                f"{camera_index} {numbers(project(point, camera_index))}"
                for camera_index in range(len(CAMERA_CENTERS))
            ]
            observation_rows.append(f"{point_id} {len(observations)} " + " ".join(observations))
        write_rows(directory / "gcp.txt", gcp_rows)
        write_rows(directory / "gcp_observations.txt", observation_rows)
    return block


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ControlledRunnerTests(unittest.TestCase):
    executable: Path
    work_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_root = cls.work_root / "input"
        cls.block = make_network(cls.data_root)
        cls.output = cls.work_root / "controlled"
        # Remove inherited solver overrides so all checks use the published
        # default protocol, with a deterministic single-thread execution.
        cls.environment = {key: value for key, value in os.environ.items() if not key.startswith("CEP_")}
        cls.environment["CEP_NUM_THREADS"] = "1"
        for index, method in enumerate(METHODS):
            cls.run_ba(method, resume=index > 0)
        cls.baseline_rows = {row["method"]: row for row in read_csv(cls.output / "summary.csv")}
        cls.baseline_poses = {method: cls.pose_path(method).read_bytes() for method in METHODS}

    @classmethod
    def command(cls, arguments: list[str]) -> str:
        result = subprocess.run(
            arguments,
            cwd=cls.executable.parent,
            env=cls.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=60,
        )
        if result.returncode != 0:
            raise AssertionError(f"Command exited {result.returncode}: {arguments}\n{result.stdout}")
        return result.stdout

    @classmethod
    def pose_path(cls, method: str, output: Path | None = None) -> Path:
        return (output or cls.output) / BASE_NAME / RUN_NAME / f"{method}__gcp-control" / "FinalPose.txt"

    @classmethod
    def run_ba(
        cls, method: str, *, resume: bool = False,
        data_root: Path | None = None, output: Path | None = None,
    ) -> str:
        arguments = [
            str(cls.executable), "--problems", str(data_root or cls.data_root),
            "--out", str(output or cls.output), "--method", method,
            "--mode", "clean", "--use-gcp-control", "--export-final-pose",
        ]
        if resume:
            arguments.append("--resume")
        return cls.command(arguments)

    @classmethod
    def evaluate(cls, output: Path, block: Path) -> list[dict[str, str]]:
        evaluation = output / "absolute_accuracy"
        cls.command([
            sys.executable,
            str(Path(__file__).resolve().with_name("checkpoint_absolute_accuracy_batch.py")),
            "--diagnostic-root", str(output), "--quality-root", str(block / "quality"),
            "--gcp", str(block / "quality" / RUN_NAME / "gcp.txt"),
            "--out-dir", str(evaluation), "--run-control-mode", "gcp-control",
            "--alignment-mode", "direct",
        ])
        return read_csv(evaluation / "checkpoint_absolute_accuracy_by_method.csv")

    def assert_summary_complete(self, output: Path | None = None) -> None:
        rows = read_csv((output or self.output) / "summary.csv")
        self.assertEqual(len(rows), len(METHODS), "Prepared run discovered twice or resume duplicated rows")
        self.assertEqual({row["method"] for row in rows}, set(METHODS))
        keys = [(row["base_dataset"], row["quality_dataset"], row["method"], row["mode"]) for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["base_dataset"], BASE_NAME)
            self.assertEqual(row["quality_dataset"], RUN_NAME)
            self.assertEqual(int(row["num_threads"]), 1)
            self.assertEqual(int(row["observations"]), (TIE_POINT_COUNT + CONTROL_COUNT) * len(CAMERA_CENTERS))
            self.assertGreater(float(row["initial_cost"]), 1.0, "Input must require actual optimization")
            self.assertLess(float(row["final_rmse_px"]), 1e-4)

    def test_01_clean_pose_export_and_direct_accuracy(self) -> None:
        self.assert_summary_complete()
        for method in METHODS:
            pose = np.loadtxt(self.pose_path(method))
            self.assertEqual(pose.shape, (len(CAMERA_CENTERS), 7))
            self.assertTrue(np.isfinite(pose).all())
            run_dir = self.pose_path(method).parent
            self.assertFalse((run_dir / "convergence.csv").exists())
            self.assertFalse((run_dir / "report.txt").exists())
        evaluated = self.evaluate(self.output, self.block)
        self.assertEqual({row["method"] for row in evaluated}, set(METHODS))
        for row in evaluated:
            self.assertEqual(int(row["checkpoint_count"]), CHECKPOINT_COUNT)
            self.assertEqual(int(row["alignment_control_count"]), 0)
            self.assertEqual(row["alignment_mode"], "direct")
            self.assertEqual(float(row["similarity_scale"]), 1.0)
            self.assertLess(float(row["rmse_3d"]), 1e-4)
            print(f"  {row['method']}: direct CP RMSE = {float(row['rmse_3d']):.3g} m", flush=True)

    def test_02_resume_skips_complete_runs(self) -> None:
        for method in METHODS:
            before = self.pose_path(method).stat().st_mtime_ns
            output = self.run_ba(method, resume=True)
            self.assertIn("Attempted runs: 0", output)
            self.assertIn("Skipped completed runs: 1", output)
            self.assertEqual(self.pose_path(method).stat().st_mtime_ns, before)
        self.assert_summary_complete()

    def test_03_resume_repairs_missing_pose_without_duplicate_rows(self) -> None:
        method = METHODS[0]
        self.pose_path(method).unlink()
        output = self.run_ba(method, resume=True)
        self.assertIn("Attempted runs: 1", output)
        self.assertGreater(self.pose_path(method).stat().st_size, 0)
        self.assertEqual(self.pose_path(method).read_bytes(), self.baseline_poses[method])
        self.assert_summary_complete()

    def test_04_resume_repairs_empty_pose_without_duplicate_rows(self) -> None:
        method = METHODS[1]
        self.pose_path(method).write_bytes(b"")
        output = self.run_ba(method, resume=True)
        self.assertIn("Attempted runs: 1", output)
        self.assertGreater(self.pose_path(method).stat().st_size, 0)
        self.assertEqual(self.pose_path(method).read_bytes(), self.baseline_poses[method])
        self.assert_summary_complete()

    def test_05_checkpoints_do_not_influence_ba(self) -> None:
        changed_root = self.work_root / "changed-checkpoints"
        shutil.copytree(self.data_root, changed_root)
        changed_block = changed_root / BASE_NAME
        directory = changed_block / "quality" / RUN_NAME
        records = []
        for line in (directory / "gcp.txt").read_text().splitlines():
            fields = line.split()
            if int(fields[7]):
                fields[2:5] = [str(float(value) + 1000.0) for value in fields[2:5]]
            records.append(" ".join(fields))
        write_rows(directory / "gcp.txt", records)
        observations = []
        for line in (directory / "gcp_observations.txt").read_text().splitlines():
            fields = line.split()
            if int(fields[0]) >= CONTROL_COUNT:
                for position in range(3, len(fields), 3):
                    fields[position] = str(float(fields[position]) + 25.0)
                    fields[position + 1] = str(float(fields[position + 1]) - 40.0)
            observations.append(" ".join(fields))
        write_rows(directory / "gcp_observations.txt", observations)
        changed_output = self.work_root / "changed-output"
        for index, method in enumerate(METHODS):
            self.run_ba(method, resume=index > 0, data_root=changed_root, output=changed_output)
        self.assert_summary_complete(changed_output)
        for row in read_csv(changed_output / "summary.csv"):
            baseline = self.baseline_rows[row["method"]]
            for field in ("initial_cost", "final_cost", "iterations", "observations"):
                self.assertEqual(row[field], baseline[field], f"Held-out checkpoints changed {field}")
            self.assertEqual(self.pose_path(row["method"], changed_output).read_bytes(), self.baseline_poses[row["method"]])
        for row in self.evaluate(changed_output, changed_block):
            self.assertGreater(float(row["rmse_3d"]), 100.0, "Evaluator must measure the changed checkpoint data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, help="Preserve generated fixtures and outputs in a new directory")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable does not exist: {executable}")
    temporary = None
    if args.work_dir:
        work_root = args.work_dir.resolve()
        if work_root.exists():
            parser.error(f"--work-dir must be a new directory: {work_root}")
        work_root.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="cep-d144-test-")
        work_root = Path(temporary.name)
    ControlledRunnerTests.executable = executable
    ControlledRunnerTests.work_root = work_root
    print(f"Synthetic controlled-BA fixture: {work_root}", flush=True)
    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ControlledRunnerTests)
        )
        return 0 if result.wasSuccessful() else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
