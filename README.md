# CEP-BA-Bench

CEP-BA-Bench is a C++ benchmark implementation for comparing object-point
parameterizations in bundle adjustment. It exposes 14 parameterizations under
one matched Ceres-based runner and separates clean timing runs from heavier
diagnostic runs.

This repository contains the source code, benchmark driver, protocol notes,
analysis tools, and a D144 reproduction runner. Large datasets, local build
products, external reference material, and generated benchmark results are not
versioned.

## What Is Included

```text
ba_v2/
  include/                  public headers and point/camera parameterizations
  src/                      bundle-adjustment implementation and exporter bridge
example_v2/
  example.cpp               command-line benchmark runner
docs/
  benchmark_protocol.md     timing and diagnostic protocol notes
  d144_reproduction.md      controlled D144 data, split, and execution protocol
  d144_validation.md        publication validation and its scope
tools/
  analyze_initial_value_benchmark.py
  run_d144.py               D144 preparation, adjustment, and checkpoint evaluation
  test_d144_runner.py       small controlled-network regression tests
```

## Implemented Methods

The runner accepts one of the following method IDs through `--method`. Without
`--method`, it runs all methods in this order:

```text
A0-XYZ-W
A0-XYInvZ-W
A0-SphRange-W
A0-SphInvRange-W
A1-XYZ-Ac
A1-XYInvZ-Ac
A1-SphRange-Ac
A1-SphInvRange-Ac
A2-Parallax-Mc
A1-XYZ-Aw
A1-XYInvZ-Aw
A1-SphRange-Aw
A1-SphInvRange-Aw
A2-Parallax-Mw
```

Frame suffixes:

- `W`: global world frame
- `Ac`: main-anchor camera frame
- `Aw`: anchor-origin world-aligned frame
- `Mc`: parallax main-anchor camera frame
- `Mw`: parallax main-anchor world-aligned frame

## Dependencies

The maintained build uses Windows/MSVC. It requires:

- CMake 3.21 or newer
- C++17 compiler
- Ceres Solver 2.3 or newer, with a sparse linear algebra backend
- Eigen 3.3 or newer
- the dependencies required by the selected Ceres installation

Install these libraries separately. CMake resolves their installed package
configurations; this project does not download dependencies. If they are not on
CMake's search path, supply `Ceres_DIR` and `Eigen3_DIR`, or use
`CMAKE_PREFIX_PATH`. For example, from the repository root:

```powershell
cmake -S .\example_v2 -B .\example_v2\build `
  -DCeres_DIR="C:/deps/ceres/lib/cmake/Ceres" `
  -DEigen3_DIR="C:/deps/eigen/share/eigen3/cmake"
```

Additional Ceres dependencies are resolved through `CeresConfig.cmake`. A Ceres
build configured with optional dependencies such as cuDSS requires those
packages to be discoverable as well.

The D144 runner requires Python 3.10 or newer and NumPy. Analysis and plotting
scripts additionally use pandas, SciPy, and Matplotlib.

## Build

```powershell
cmake -S .\example_v2 -B .\example_v2\build
cmake --build .\example_v2\build --config Release
```

The executable is written to:

```text
example_v2\build\Release\example.exe
```

Configuring from the repository root is also supported with
`cmake -S . -B build`; that build writes the executable to
`build\example_v2\Release\example.exe`.

## Dataset Layout

The reported benchmark uses the BA Datasets collection, specifically the
Initial Value tier. The runner can read this native layout directly:

The collection is organized into four scene categories:

| Category | Problems | Description | Dataset Access |
| --- | ---: | --- | --- |
| `Close-Range` | 12 | close-range indoor and outdoor photogrammetry scenes | [Google Drive](https://drive.google.com/drive/folders/1mvsQEFGBvZ-VcfxIJ3hqXiBV3tVTPMSe) |
| `Oblique-5` | 30 | five-camera oblique aerial blocks | [Google Drive](https://drive.google.com/drive/folders/1NDOMrSZocyTG7JEdLQ7KdujHUKDikmQa) |
| `UAV` | 34 | UAV image blocks for aerial reconstruction | [Google Drive](https://drive.google.com/drive/folders/1-VA-JrVe03PVZnswjuLAvx_EC7vqzSXN) |
| `Vehicle` | 29 | vehicle-mounted multi-camera sequences | [Google Drive](https://drive.google.com/drive/folders/1_GID2a5O5CSfUn5QfoWFhhnA_NwI5kmi) |

Together these folders provide 105 Initial Value benchmark problems. Each
problem contains one `Initial Value` folder used as the degraded initialization
input and one `Ground Truth` folder retained as the reference reconstruction.

```text
BA Datasets/
  Close-Range/
    CR1-problem-11-9611/
      Ground Truth/
        Cam.txt
        Feature.txt
        XYZ.txt
        cal.txt
      Initial Value/
        Cam.txt
        Feature.txt
        XYZ.txt
        cal.txt
  Oblique-5/
  UAV/
  Vehicle/
```

The benchmark uses `Initial Value` as the optimization input. `Ground Truth`
is retained as the reference reconstruction when reference-recovery analysis is
performed outside the clean timing run.

Expected files:

- `cal.txt`: camera intrinsic parameters, including focal length and principal point values.
- `Cam.txt` or `Cam-*.txt`: Euler-angle camera orientation, camera center, and camera ID records.
- `XYZ.txt`: object-point coordinates for the corresponding tier.
- `Feature.txt`: feature tracks, with image indices and image coordinates for each observation.

For compatibility with older run manifests, the runner also accepts this
prepared layout:

```text
<input-root>/
  <Category>__<Problem>/
    original/
      Cam.txt
      Feature.txt
      XYZ.txt
      cal.txt
    quality/
      Initial Value/
        Cam.txt
        Feature.txt
        XYZ.txt
        cal.txt
```

## Command-Line Usage

```text
example.exe [--problems <input-root>] [--out <results-root>]
            [--method <MethodId>] [--limit <N>] [--dataset <name>]
            [--mode clean|clean-logged|diagnostic] [--resume] [--no-xyz]
            [--export-final-pose]
            [--point-condition-sample <N>] [--schur-sample <N>]
            [--use-gcp-control]
            [--gcp <gcp.txt>] [--gcp-observations <gcp_observations.txt>]
```

Important options:

- `--problems`: BA Datasets root or a prepared benchmark root
- `--out`: output directory
- `--method`: run a single method; omitted means all 14 methods
- `--limit`: limit the number of discovered benchmark problems
- `--dataset`: run one category/problem/base name or run name
- `--mode clean`: timing and summary-output mode
- `--mode clean-logged`: clean solver run with lightweight per-iteration
  convergence and termination logs
- `--mode diagnostic`: export reports, convergence, poses, point clouds, and sampled diagnostics
- `--resume`: keep successful rows in an existing `summary.csv`, rerun failures, and skip completed runs
- `--export-final-pose`: also export `FinalPose.txt` in clean or clean-logged mode, without enabling diagnostic output
- `--no-xyz`: exclude `A0-XYZ-W`
- `--point-condition-sample`: sample point-block conditioning diagnostics
- `--schur-sample`: sample reduced camera Schur-system diagnostics
- `--use-gcp-control`: add fixed 3D GCP projection residuals from non-checkpoint records; checkpoint records are held out
- `--gcp`, `--gcp-observations`: override the per-run `gcp.txt` and `gcp_observations.txt` files used by `--use-gcp-control`

## Clean Run

Clean mode writes the aggregate `summary.csv` after each solve. Optional
`--export-final-pose` writes camera poses after the timed Ceres solve:

```powershell
$env:CEP_NUM_THREADS = "1"
Remove-Item Env:CEP_STRICT_VECTOR_DIAGNOSTICS -ErrorAction SilentlyContinue

.\example_v2\build\Release\example.exe `
  --problems ".\data\BA Datasets" `
  --out .\runs\initial_value_clean `
  --mode clean `
  --resume
```

Smoke test one problem and one method:

```powershell
.\example_v2\build\Release\example.exe `
  --problems ".\data\BA Datasets" `
  --out .\runs\initial_value_smoke `
  --limit 1 `
  --method A2-Parallax-Mw `
  --mode clean
```

The summary file records status, Ceres termination status, runtime,
linear-solver time, iteration counts, reprojection RMSE, final gradient norms,
termination message, actual thread count, and the report path when applicable.

## Logged Clean Run

The logged-clean mode preserves the clean solver configuration and does not
enable strict vector diagnostics. It writes `convergence.csv`, `report.txt`,
and `metrics.json` for each problem-method pair. The convergence file includes
Ceres' per-iteration RMSE and `iteration_time_sec`,
`step_solver_time_sec`, and `cumulative_time_sec` fields. The report preserves
the raw Ceres termination message and full solver report, which can distinguish
function, gradient, and parameter tolerance termination.

```powershell
$env:CEP_NUM_THREADS = "1"
Remove-Item Env:CEP_STRICT_VECTOR_DIAGNOSTICS -ErrorAction SilentlyContinue

.\example_v2\build\Release\example.exe `
  --problems ".\data\BA Datasets" `
  --out .\runs\initial_value_clean_logged `
  --mode clean-logged `
  --resume
```

Use a new output root when changing the summary schema. The logged-clean
`solver_time_sec` and per-iteration `cumulative_time_sec` are suitable for
time-to-target profiles; report-writing time is outside Ceres' solver time.

## Diagnostic Run

Diagnostic mode reruns the same problem-method pairs with extra outputs:

```powershell
$env:CEP_STRICT_VECTOR_DIAGNOSTICS = "1"
$env:CEP_NUM_THREADS = "1"

.\example_v2\build\Release\example.exe `
  --problems ".\data\BA Datasets" `
  --out .\runs\initial_value_diagnostic_strict `
  --mode diagnostic `
  --resume `
  --point-condition-sample 128 `
  --schur-sample 32
```

For each run, diagnostic mode can write:

```text
<out>/<base-dataset>/Initial Value/<MethodId>/
  report.txt
  metrics.json
  convergence.txt
  FinalPose.txt
  Final3D.ply
  point_block_conditioning.csv
  schur_summary.csv
  schur_spectrum.csv
```

Diagnostic timings should be kept separate from clean timing comparisons.

## Analysis

Generate Initial Value aggregate tables:

```powershell
python .\tools\analyze_initial_value_benchmark.py `
  --summary .\runs\initial_value_clean\summary.csv `
  --out .\runs\initial_value_clean\analysis
```

The script expects `base_dataset` values in the form
`<Category>__<Problem>`, which is what the runner emits for the native BA
Datasets layout.

## Controlled D144 Reproduction

The D144 experiment uses one ground-controlled block, eight initialization
quality sets, and the same 14 methods. Its data are distributed separately in
the [PVL-BA-Bench catalogue](https://pub-2c28bdf6e62548919c47727a9b969dda.r2.dev/index.html),
under `control_at_20` as
`shiyan-problem-i2823-p188150-o1073331-g23-c0`.

After downloading and extracting the block, run:

```powershell
python -m pip install numpy

python .\tools\run_d144.py `
  --problem-root ".\data\shiyan-problem-i2823-p188150-o1073331-g23-c0" `
  --executable .\example_v2\build\Release\example.exe `
  --out .\runs\d144_controlled
```

The runner creates the fixed 13-control/10-checkpoint split, uses one solver
thread and the GCP datum, exports final camera poses, and evaluates held-out
checkpoints by forward intersection without post-hoc Sim(3) alignment. A full
run comprises 112 adjustments. Use `--prepare-only` to validate inputs and
write the split without adjustment, or `--evaluate-only` to evaluate completed
runs. See [D144 reproduction](docs/d144_reproduction.md) for download, selection,
resume, output, and metric details.

## Protocol Notes

Recommended usage:

1. Run all Initial Value problems in `clean` mode for timing, robustness,
   convergence status, iteration counts, and final RMSE.
2. Run `diagnostic` mode separately for first-order, LM, point-block, and
   Schur-complement diagnostics.
3. Treat Ceres termination status and final-solution quality separately.

See `docs/benchmark_protocol.md` for additional metric definitions and
diagnostic guidance.

## Notes

- The benchmark fixes the Euler-angle camera representation across all methods;
  it compares object-point parameterizations, not camera-rotation
  parameterizations.
- Pure monocular BA fixes the first camera pose and the dominant baseline
  component of the second valid camera by default. Set
  `CEP_FIX_MONOCULAR_GAUGE=0` only when the formulation is constrained another
  way.
- GCP-controlled runs disable the free-network gauge by default because fixed
  control points define the datum. Set `CEP_FIX_CONTROLLED_GAUGE=1` only for
  a deliberately over-constrained diagnostic run.
- The dataset collection and generated outputs are intentionally kept outside
  Git because they are large and environment-specific.
