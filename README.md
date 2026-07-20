# CEP-BA-Bench

CEP-BA-Bench is a C++ benchmark implementation for comparing object-point
parameterizations in bundle adjustment. It exposes the 14 parameterizations
used in the accompanying CEP bundle-adjustment experiments, runs them on
PVL-BA-Bench-style datasets, and separates clean timing runs from heavier
diagnostic exports.

This repository contains the source code, benchmark driver, documentation, and
analysis scripts. Large datasets, local build products, external reference
material, and generated benchmark results are intentionally not versioned.

## What is included

```text
ba_v2/
  include/                  C++ public headers and point/camera parameterizations
  src/                      BA implementation and exporter bridge
example_v2/
  example.cpp               command-line benchmark runner
docs/
  benchmark_protocol.md     recommended timing/diagnostic protocol
  implementation_validation.md
tools/
  *.py, *.ps1               dataset extraction, synthetic generation, analysis, plots
```

## Implemented methods

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

The suffixes indicate the coordinate frame used by the point representation:

- `W`: world frame
- `Ac`: main-anchor camera frame
- `Aw`: anchor-origin world-aligned frame
- `Mc`: parallax main-anchor camera frame
- `Mw`: parallax main-anchor world-aligned frame

## Dependencies

The project is currently maintained as a Windows/MSVC CMake project. The
implementation uses:

- CMake 3.10 or newer
- C++17 compiler
- Ceres Solver
- Eigen
- Sophus headers
- vcpkg-installed transitive dependencies used by Ceres
- optional NVIDIA cuDSS when your Ceres build was configured with it

The CMake files keep the original local defaults but expose them as cache
variables, so another machine can override them without editing source files:

```powershell
cmake -S .\example_v2 -B .\example_v2\build `
  -DCEP_THIRDPARTY_DIR="D:\deps\3rdparty" `
  -DCEP_VCPKG_INSTALLED_DIR="D:\vcpkg\installed" `
  -DCEP_VCPKG_TARGET_TRIPLET="x64-windows" `
  -DCeres_DIR="D:\ceres-solver\install\lib\cmake\Ceres" `
  -Dcudss_DIR="D:\NVIDIA cuDSS\v0.5\lib\12\cmake\cudss"
```

If your Ceres build does not depend on cuDSS, omit `-Dcudss_DIR`.

## Build

From a Developer PowerShell or a shell where MSVC and CMake are available:

```powershell
cd E:\zuo\projects\CEP

cmake -S .\example_v2 -B .\example_v2\build -DCMAKE_BUILD_TYPE=Release
cmake --build .\example_v2\build --config Release
```

The benchmark executable is written to:

```text
example_v2\build\Release\example.exe
```

## Dataset layout

The runner expects PVL-BA-Bench-style datasets. Each base dataset should contain
an `original` folder and one or more `quality` subdatasets:

```text
PVL-BA-Bench/
  <base-dataset>/
    original/
      Cam-*.txt
      Feature.txt
      XYZ.txt
      cal.txt
    quality/
      <quality-dataset>/
        Cam-*.txt
        Feature.txt
        XYZ.txt
        cal.txt
        noise_metadata.json
```

`original` is used as the reference solution for the generated quality cases.
The benchmark compares optimization behavior on the quality datasets.

## Command-line usage

```text
example.exe [--problems <PVL-BA-Bench-root>] [--out <results-root>]
            [--method <MethodId>] [--limit <N>] [--dataset <base-name>]
            [--mode clean|diagnostic] [--resume] [--no-xyz]
            [--point-condition-sample <N>] [--schur-sample <N>]
```

Important options:

- `--problems`: root directory containing benchmark datasets
- `--out`: output directory
- `--method`: run a single method; omitted means all 14 methods
- `--limit`: limit the number of base datasets
- `--dataset`: run one base dataset by name
- `--mode clean`: authoritative timing mode
- `--mode diagnostic`: export reports, convergence, poses, point clouds, and diagnostics
- `--resume`: keep successful rows in an existing `summary.csv`, rerun failures, and skip completed runs
- `--no-xyz`: exclude `A0-XYZ-W` from the selected methods
- `--point-condition-sample`: sample point-block conditioning diagnostics
- `--schur-sample`: sample reduced camera Schur-system diagnostics

## Clean timing run

Use clean mode for runtime comparison tables. It writes only the aggregate
`summary.csv` and disables diagnostic exports during the timed solve:

```powershell
.\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_public_release_clean `
  --mode clean `
  --resume
```

Smoke test one dataset and one method:

```powershell
.\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_smoke `
  --limit 1 `
  --method A2-Parallax-Mw `
  --mode clean
```

The summary file includes status, Ceres termination status, runtime,
linear-solver time, iteration counts, reprojection RMSE, final gradient norms,
and the diagnostic report path when applicable.

## Diagnostic run

Diagnostic mode is for representative subsets, not primary timing comparisons:

```powershell
.\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_diagnostics `
  --limit 1 `
  --method A2-Parallax-Mc `
  --mode diagnostic `
  --point-condition-sample 16 `
  --schur-sample 16
```

For each run, diagnostic mode can write:

```text
<out>/<base-dataset>/<quality-dataset>/<MethodId>/
  report.txt
  metrics.json
  convergence.txt
  FinalPose.txt
  Final3D.ply
  point_block_conditioning.csv
  schur_summary.csv
  schur_spectrum.csv
```

Point-conditioning and Schur exports are sampled post-solve diagnostics. They
are intentionally disabled in clean timing mode.

## Analysis scripts

Generate aggregate CSV/LaTeX tables and figures:

```powershell
python .\tools\plot_benchmark_results.py `
  --summary E:\zuo\projects\CEP\benchmark_public_release_clean\summary.csv `
  --diagnostic-root E:\zuo\projects\CEP\benchmark_diagnostics `
  --out E:\zuo\projects\CEP\benchmark_figures `
  --solution-rmse-relative-tolerance 0.01
```

Other utility scripts under `tools/` support PVL package extraction, synthetic
benchmark generation, IJRR-inspired parallax reproduction, and specialized
summary analysis.

## Benchmark protocol

Recommended usage is:

1. Run every available quality dataset in `clean` mode for timing, success
   rate, convergence status, iteration counts, and final RMSE.
2. Run `diagnostic` mode only on a representative subset spanning problem size,
   geometry, and initial-error levels.
3. Treat Ceres termination status and final-solution equivalence separately.
   A run can report Ceres convergence while still landing at a worse local
   solution.

See `docs/benchmark_protocol.md` for the full protocol and
`docs/implementation_validation.md` for validation notes and known limitations.

## Notes

- The benchmark currently fixes the Euler-angle camera representation across
  all methods; it compares object-point parameterizations, not camera-rotation
  parameterizations.
- Pure monocular BA fixes the first camera pose and the dominant baseline
  component of the second valid camera by default. Set
  `CEP_FIX_MONOCULAR_GAUGE=0` only when the formulation is constrained another
  way.
- Datasets and generated results can be very large. Keep them outside Git or in
  release artifacts, not in the source tree.
