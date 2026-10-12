#!/usr/bin/env python3
"""Validate D144 inputs, apply the published split, run BA, and evaluate CPs.

The data directory is read-only. No downloads or post-hoc alignment are made.
Use --prepare-only to inspect the generated split before running the solver.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
METHODS = (
    "A0-XYZ-W", "A0-XYInvZ-W", "A0-SphRange-W", "A0-SphInvRange-W",
    "A1-XYZ-Ac", "A1-XYInvZ-Ac", "A1-SphRange-Ac", "A1-SphInvRange-Ac",
    "A2-Parallax-Mc", "A1-XYZ-Aw", "A1-XYInvZ-Aw", "A1-SphRange-Aw",
    "A1-SphInvRange-Aw", "A2-Parallax-Mw",
)
SOLVER_ENV = {
    "CEP_NUM_THREADS": "1", "CEP_FIX_CONTROLLED_GAUGE": "0",
    "CEP_ANCHOR_POLICY": "current", "CEP_MAX_NUM_ITERATIONS": "100",
    "CEP_FUNCTION_TOLERANCE": "1e-6", "CEP_GRADIENT_TOLERANCE": "1e-10",
    "CEP_PARAMETER_TOLERANCE": "1e-8", "CEP_INITIAL_TRUST_REGION_RADIUS": "1e4",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_gcp(source, protocol):
    """Change only the role field; retain source IDs, names and coordinate tokens."""
    rows = []
    records = {}
    checkpoints = set(protocol["checkpoint_ids"])
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            rows.append(line)
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"Malformed GCP row in {source}: {line}")
        point_id = int(fields[0])
        if point_id in records:
            raise ValueError(f"Duplicate GCP ID {point_id} in {source}")
        if not all(math.isfinite(float(value)) for value in fields[2:7]):
            raise ValueError(f"Nonfinite surveyed coordinate/accuracy for ID {point_id}")
        fields[7] = "1" if point_id in checkpoints else "0"
        records[point_id] = fields
        rows.append(" ".join(fields))
    expected_ids = set(protocol["control_ids"]) | checkpoints
    if set(records) != expected_ids:
        raise ValueError(f"Expected exactly the 23 published D144 point IDs in {source}")
    return "\n".join(rows) + "\n", records


def observation_counts(path, protocol):
    tracks = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        point_id, count = map(int, fields[:2])
        if point_id in tracks or count < 2 or len(fields) != 2 + 3 * count:
            raise ValueError(f"Malformed/duplicate survey track {point_id} in {path}")
        for index in range(2, len(fields), 3):
            if not 0 <= int(fields[index]) < protocol["images"]:
                raise ValueError(f"Invalid image index in {path}")
            if not all(math.isfinite(float(v)) for v in fields[index + 1:index + 3]):
                raise ValueError(f"Nonfinite image measurement in {path}")
        tracks[point_id] = count
    if set(tracks) != set(protocol["control_ids"] + protocol["checkpoint_ids"]):
        raise ValueError(f"Missing/extra surveyed-point tracks in {path}")
    for role in ("control", "checkpoint"):
        actual = sum(tracks[i] for i in protocol[f"{role}_ids"])
        if actual != protocol[f"{role}_observations"]:
            raise ValueError(f"Expected {protocol[f'{role}_observations']} {role} observations; got {actual}")
    return tracks


def prepare_inputs(block, protocol):
    quality_root = block / "quality"
    original = block / "original"
    if not original.is_dir() or not quality_root.is_dir():
        raise ValueError("--problem-root must be the extracted block containing original/ and quality/")
    split_text, original_records = split_gcp(original / "gcp.txt", protocol)
    static_hashes = {name: sha256(original / name)
                     for name in ("Feature.txt", "cal.txt", "gcp_observations.txt")}
    observation_counts(original / "gcp_observations.txt", protocol)
    qualities = {}
    for directory in sorted(p for p in quality_root.iterdir() if p.is_dir()):
        metadata = json.loads((directory / "noise_metadata.json").read_text(encoding="utf-8-sig"))
        counts = json.loads((directory / "dataset_metadata.json").read_text(encoding="utf-8-sig"))
        target = metadata["target_rmse_px"]
        if target not in protocol["target_rmse_px"] or target in qualities:
            raise ValueError(f"Unexpected/duplicate quality target {target} in {directory}")
        expected_mode = "init-pose-triangulate" if target <= 100 else "init-pose-point"
        expected = {"seed": protocol["seed"], "images": protocol["images"],
                    "points": protocol["points"], "observations": protocol["tie_observations"],
                    "gcps": 23, "mode": expected_mode,
                    "feature_observations_perturbed": 0, "gcp_observations_perturbed": 0,
                    "scale_solver_sample_step": 4, "full_refine_iterations": 0}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"{directory.name}: expected {key}={value}, got {metadata.get(key)}")
        for key in ("images", "points", "observations", "gcps"):
            if counts.get(key) != expected[key]:
                raise ValueError(f"Inconsistent dataset_metadata {key} in {directory}")
        realized = float(metadata["actual_rmse_px"])
        if not math.isfinite(realized) or realized <= 0:
            raise ValueError(f"Invalid realized RMSE in {directory}")
        _, records = split_gcp(directory / "gcp.txt", protocol)
        if records != original_records:
            raise ValueError(f"Survey records differ from original in {directory}")
        for name, expected_hash in static_hashes.items():
            if sha256(directory / name) != expected_hash:
                raise ValueError(f"Static observations/calibration changed: {directory / name}")
        cameras = list(directory.glob("Cam*.txt"))
        if len(cameras) != 1 or not (directory / "XYZ.txt").is_file():
            raise ValueError(f"Expected one camera file and XYZ.txt in {directory}")
        qualities[target] = {"directory": directory.name, "target_rmse_px": target,
                             "actual_rmse_px": realized, "mode": expected_mode,
                             "camera_file": cameras[0].name,
                             "camera_sha256": sha256(cameras[0]),
                             "xyz_sha256": sha256(directory / "XYZ.txt"),
                             "noise_metadata_sha256": sha256(directory / "noise_metadata.json")}
    if set(qualities) != set(protocol["target_rmse_px"]):
        raise ValueError("Expected all eight published D144 quality sets")
    return split_text, qualities, static_hashes


def validate_outputs(out, block, qualities, methods, mode, protocol):
    expected = {(block.name, item["directory"], method, f"{mode}-gcp-control")
                for item in qualities for method in methods}
    with (out / "summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    keys = [(row["base_dataset"], row["quality_dataset"], row["method"], row["mode"])
            for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Expected {len(expected)} unique requested run keys; found {len(keys)} rows")
    for row in rows:
        if row["status"] != "ok" or not math.isfinite(float(row["final_rmse_px"])):
            raise ValueError(f"Unsuccessful/nonfinite run: {row['quality_dataset']} / {row['method']}")
        for field, expected_value in (("num_threads", 1), ("cameras", protocol["images"]),
                                      ("points", protocol["points"]),
                                      ("observations", protocol["tie_observations"] + protocol["control_observations"])):
            if int(row[field]) != expected_value:
                raise ValueError(f"Run {row['method']} has {field}={row[field]}, expected {expected_value}")
        pose = out / block.name / row["quality_dataset"] / (row["method"] + "__gcp-control") / "FinalPose.txt"
        if not pose.is_file() or pose.stat().st_size == 0:
            raise ValueError(f"Missing/empty final pose: {pose}")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-root", required=True, type=Path)
    parser.add_argument("--executable", type=Path, help="Built example.exe; required unless preparing/evaluating only")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--quality", type=int, choices=(2, 5, 10, 20, 50, 100, 200, 500))
    parser.add_argument("--mode", choices=("clean", "clean-logged", "diagnostic"), default="clean")
    parser.add_argument("--resume", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    block, out = args.problem_root.resolve(), args.out.resolve()
    if out == block or block in out.parents:
        parser.error("--out must be outside the input block")
    if not args.prepare_only and not args.evaluate_only and (
            args.executable is None or not args.executable.resolve().is_file()):
        parser.error("Supply --executable pointing to the built example.exe")
    protocol = json.loads((REPO / "protocols" / "d144_split.json").read_text())
    split_text, qualities, static_hashes = prepare_inputs(block, protocol)
    selected = [qualities[t] for t in protocol["target_rmse_px"] if args.quality is None or t == args.quality]
    methods = [args.method] if args.method else list(METHODS)
    # Dataset names encode the full network size, and can make nested output
    # paths exceed the Windows runtime's legacy limits. Fail before a long BA.
    if os.name == "nt":
        for quality in selected:
            for method in methods:
                leaf = out / block.name / quality["directory"] / (method + "__gcp-control")
                if len(str(leaf)) >= 248 or len(str(leaf / "FinalPose.txt")) >= 260:
                    raise ValueError("Output paths exceed Windows path limits; choose a shorter --out, e.g. C:/runs/d144")
    run_config = {"block": block.name, "mode": args.mode, "methods": methods,
                  "quality_directories": [q["directory"] for q in selected],
                  "solver_environment": SOLVER_ENV, "alignment_mode": "direct"}
    manifest = {"protocol": protocol, "run": run_config, "source_static_sha256": static_hashes,
                "source_gcp_sha256": sha256(block / "original" / "gcp.txt"),
                "split_sha256": hashlib.sha256(split_text.encode("utf-8")).hexdigest(),
                "qualities": [qualities[t] for t in protocol["target_rmse_px"]]}
    manifest_path = out / "protocol" / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing output uses different inputs/settings; choose a new --out")
    if (out / "summary.csv").exists() and not (args.resume or args.evaluate_only or args.prepare_only):
        raise ValueError("summary.csv already exists; use --resume or a new --out")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_path = manifest_path.parent / "gcp_13control_10checkpoint.txt"
    if split_path.exists() and split_path.read_text(encoding="utf-8") != split_text:
        raise ValueError("Existing split differs from the published protocol; choose a new --out")
    split_path.write_text(split_text, encoding="utf-8", newline="\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated eight tiers; 13 controls (201 observations), 10 held-out CPs (124 observations).", flush=True)
    print(f"Split: {split_path}", flush=True)
    if args.prepare_only:
        return
    if not args.evaluate_only:
        command = [str(args.executable.resolve()), "--problems", str(block.parent),
                   "--dataset", selected[0]["directory"] if args.quality else block.name,
                   "--out", str(out), "--mode", args.mode, "--use-gcp-control",
                   "--gcp", str(split_path), "--export-final-pose"]
        if args.method:
            command += ["--method", args.method]
        if args.resume:
            command += ["--resume"]
        env = os.environ.copy()
        env.update(SOLVER_ENV)
        env.pop("CEP_STRICT_VECTOR_DIAGNOSTICS", None)
        execution = {"command": command, "executable_sha256": sha256(args.executable.resolve()),
                     "solver_environment": SOLVER_ENV,
                     "strict_vector_diagnostics": args.mode == "diagnostic"}
        execution_path = manifest_path.parent / "last_execution.json"
        if (out / "summary.csv").exists() and execution_path.exists():
            previous = json.loads(execution_path.read_text())
            if previous["executable_sha256"] != execution["executable_sha256"]:
                raise ValueError("Executable changed since the previous solve; choose a new --out")
        execution_path.write_text(
            json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        subprocess.run(command, env=env, check=True)
    run_count = validate_outputs(out, block, selected, methods, args.mode, protocol)
    evaluation = out / "absolute_accuracy"
    subprocess.run([sys.executable, str(REPO / "tools" / "checkpoint_absolute_accuracy_batch.py"),
                    "--diagnostic-root", str(out), "--quality-root", str(block / "quality"),
                    "--gcp", str(split_path), "--out-dir", str(evaluation),
                    "--run-control-mode", "gcp-control", "--alignment-mode", "direct"], check=True)
    with (evaluation / "checkpoint_absolute_accuracy_by_dataset_method.csv").open(newline="") as stream:
        results = list(csv.DictReader(stream))
    if len(results) != run_count or any(int(r["checkpoint_count"]) != 10 or
            r["alignment_mode"] != "direct" or not math.isfinite(float(r["rmse_3d"])) for r in results):
        raise ValueError("Incomplete/nonfinite held-out checkpoint evaluation")
    print(f"Verified {run_count} unique runs and {10 * run_count} direct CP estimates: {evaluation}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"D144 reproduction failed: {exc}", file=sys.stderr)
        sys.exit(1)
