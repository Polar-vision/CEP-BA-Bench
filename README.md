# CEP-BA-Bench

CEP-BA-Bench is a C++ benchmark implementation for comparing object-point
parameterizations in bundle adjustment. It exposes 14 parameterizations under
one matched Ceres-based runner and separates clean timing runs from heavier
diagnostic runs.

This repository contains only the source code, benchmark driver, protocol
notes, and the Initial Value analysis script. Large datasets, local build
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
tools/
  analyze_initial_value_benchmark.py
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

The project is maintained as a Windows/MSVC CMake project. It uses:

- CMake 3.10 or newer
- C++17 compiler
- Ceres Solver 2.3-compatible headers and libraries
- Eigen
- Sophus headers
- vcpkg-installed transitive dependencies used by Ceres
- optional NVIDIA cuDSS when the local Ceres build was configured with it

The CMake files keep local defaults but expose them as cache variables:

```powershell
cmake -S .\example_v2 -B .\example_v2\build `
  -DCEP_THIRDPARTY_DIR="D:\deps\3rdparty" `
  -DCEP_VCPKG_INSTALLED_DIR="D:\vcpkg\installed" `
  -DCEP_VCPKG_TARGET_TRIPLET="x64-windows" `
  -DCeres_DIR="D:\ceres-solver\install\lib\cmake\Ceres" `
  -Dcudss_DIR="D:\NVIDIA cuDSS\v0.5\lib\12\cmake\cudss"
```

Omit `-Dcudss_DIR` if your Ceres build does not depend on cuDSS.

## Build

```powershell
cd E:\zuo\projects\CEP

cmake -S .\example_v2 -B .\example_v2\build
cmake --build .\example_v2\build --config Release
```

The executable is written to:

```text
example_v2\build\Release\example.exe
```

## Dataset Layout

The reported benchmark uses the BA Datasets collection, specifically the
Initial Value tier. The runner can read this native layout directly:

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
            [--mode clean|diagnostic] [--resume] [--no-xyz]
            [--point-condition-sample <N>] [--schur-sample <N>]
```

Important options:

- `--problems`: BA Datasets root or a prepared benchmark root
- `--out`: output directory
- `--method`: run a single method; omitted means all 14 methods
- `--limit`: limit the number of discovered benchmark problems
- `--dataset`: run one category/problem/base name or run name
- `--mode clean`: timing and summary-output mode
- `--mode diagnostic`: export reports, convergence, poses, point clouds, and sampled diagnostics
- `--resume`: keep successful rows in an existing `summary.csv`, rerun failures, and skip completed runs
- `--no-xyz`: exclude `A0-XYZ-W`
- `--point-condition-sample`: sample point-block conditioning diagnostics
- `--schur-sample`: sample reduced camera Schur-system diagnostics

## Clean Run

Clean mode writes only `summary.csv` during the timed solve:

```powershell
.\example_v2\build\Release\example.exe `
  --problems "E:\zuo\projects\CEP\BA Datasets" `
  --out E:\zuo\projects\CEP\benchmark_initial_value_clean `
  --mode clean `
  --resume
```

Smoke test one problem and one method:

```powershell
.\example_v2\build\Release\example.exe `
  --problems "E:\zuo\projects\CEP\BA Datasets" `
  --out E:\zuo\projects\CEP\benchmark_smoke `
  --limit 1 `
  --method A2-Parallax-Mw `
  --mode clean
```

The summary file records status, Ceres termination status, runtime,
linear-solver time, iteration counts, reprojection RMSE, final gradient norms,
and the diagnostic report path when applicable.

## Diagnostic Run

Diagnostic mode reruns the same problem-method pairs with extra outputs:

```powershell
$env:CEP_STRICT_VECTOR_DIAGNOSTICS = "1"
$env:CEP_NUM_THREADS = "4"

.\example_v2\build\Release\example.exe `
  --problems "E:\zuo\projects\CEP\BA Datasets" `
  --out E:\zuo\projects\CEP\benchmark_initial_value_diagnostic_strict `
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
  --summary E:\zuo\projects\CEP\benchmark_initial_value_clean\summary.csv `
  --out E:\zuo\projects\CEP\benchmark_initial_value_clean\analysis
```

The script expects `base_dataset` values in the form
`<Category>__<Problem>`, which is what the runner emits for the native BA
Datasets layout.

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
- The dataset collection and generated outputs are intentionally kept outside
  Git because they are large and environment-specific.
