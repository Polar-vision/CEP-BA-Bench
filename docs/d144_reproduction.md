# Reproduce the controlled D144 experiment

`tools/run_d144.py` reproduces the paper's ground-controlled D144 adjustment
and direct held-out checkpoint evaluation. It prepares the experiment-specific
13-control/10-checkpoint split, runs the selected parameterizations, exports
final camera poses, and evaluates checkpoints in the measured metric datum.
The complete experiment uses eight initialization sets and 14 methods: 112
adjustments. D144 is evaluated separately from the 105-problem Initial Value
free-network benchmark.

## Download the inputs

The data are distributed separately from this code repository:

- [PVL-BA-Bench catalogue](https://pub-2c28bdf6e62548919c47727a9b969dda.r2.dev/index.html)
- [D144 ZIP on Hugging Face](https://huggingface.co/datasets/Polar-vision/PVL-BA-Bench/resolve/main/packages/pvl-ba/control_at_20/shiyan-problem-i2823-p188150-o1073331-g23-c0.zip)

The catalogue identifies the block under `control_at_20` as
`shiyan-problem-i2823-p188150-o1073331-g23-c0`. It contains 2,823 cameras,
188,150 tie points, 1,073,331 tie-point image observations, and 23 surveyed
points. Extract the ZIP and pass the block directory itself to `--problem-root`:

```text
data/
  shiyan-problem-i2823-p188150-o1073331-g23-c0/
    original/
      Cam.txt or Cam-*.txt
      Feature.txt
      XYZ.txt
      cal.txt
      gcp.txt
      gcp_observations.txt
    quality/
      <quality-set directory>/
        Cam.txt or Cam-*.txt
        Feature.txt
        XYZ.txt
        cal.txt
        gcp.txt
        gcp_observations.txt
        noise_metadata.json
        dataset_metadata.json
```

The eight quality-set directory names include the block prefix and end with
the tier names below. The runner identifies them from their metadata; no
renaming or editing of the downloaded files is needed. Local archives that
name the same block `BA-problem-000144-i2823-p188150-o1073331-g23-c0` are also
accepted.

| Nominal target (px) | Tier | Initialization procedure |
| ---: | --- | --- |
| 2 | `init-rmse002p00px` | Perturb camera poses, then regenerate tie points by multi-ray least-squares triangulation |
| 5 | `init-rmse005p00px` | Same |
| 10 | `init-rmse010p00px` | Same |
| 20 | `init-rmse020p00px` | Same |
| 50 | `init-rmse050p00px` | Same |
| 100 | `init-rmse100p00px` | Same |
| 200 | `joint-rmse200p00px` | Perturb camera poses and tie-point coordinates without re-triangulation |
| 500 | `joint-rmse500p00px` | Same |

These are nominal targets. The paper's two representative settings have
realized initial RMSEs of 2.04 and 50.57 px. Generation uses seed `20260603`;
feature and surveyed-point image observations are unchanged. Each
`noise_metadata.json` records the realized RMSE and generation settings. The
runner uses the released initialization files directly.

## Build and Python dependency

The maintained build is Windows/MSVC with CMake 3.21 or newer, Ceres Solver
2.3 or newer, and Eigen 3.3 or newer. The wrapper requires Python 3.10 or newer
and NumPy. From the repository root:

```powershell
cmake -S .\example_v2 -B .\example_v2\build `
  -DCeres_DIR="C:/deps/ceres/lib/cmake/Ceres" `
  -DEigen3_DIR="C:/deps/eigen/share/eigen3/cmake"
cmake --build .\example_v2\build --config Release
python -m pip install numpy
```

Use paths to your installed CMake package directories. The two dependency
arguments can be omitted when CMake already discovers the packages; their
installation prefixes can instead be supplied through `CMAKE_PREFIX_PATH`.
Ceres must provide a sparse backend for `SPARSE_SCHUR`. Its own dependencies
are resolved through its installed package configuration. The resulting
executable is `example_v2\build\Release\example.exe`.

## Prepare and run

To validate the downloaded inputs and write the split and protocol manifest
without running BA:

```powershell
python .\tools\run_d144.py `
  --problem-root ".\data\shiyan-problem-i2823-p188150-o1073331-g23-c0" `
  --out .\runs\d144_preflight `
  --prepare-only
```

For the full eight-set, 14-method experiment, choose a new output directory:

Keep the output path short on Windows: the nested block and quality-set names
are long. If the wrapper reports a path-limit error, use a shorter absolute
output root such as `C:/runs/d144`.

```powershell
python .\tools\run_d144.py `
  --problem-root ".\data\shiyan-problem-i2823-p188150-o1073331-g23-c0" `
  --executable .\example_v2\build\Release\example.exe `
  --out .\runs\d144_controlled
```

The default mode is `clean`. Final poses are exported with
`--export-final-pose`, so checkpoint evaluation does not require a separate
diagnostic adjustment. The wrapper uses `CEP_NUM_THREADS=1`, disables the
free-network gauge for the controlled adjustment with
`CEP_FIX_CONTROLLED_GAUGE=0`, and removes `CEP_STRICT_VECTOR_DIAGNOSTICS` from
the solver environment. Fixed GCPs establish the metric datum. The common
solver configuration is Ceres LM with `SPARSE_SCHUR`, 100 nonlinear iterations,
function tolerance `1e-6`, gradient tolerance `1e-10`, parameter tolerance
`1e-8`, initial trust-region radius `10000`, minimum relative decrease `0.001`,
and Jacobi scaling enabled. The runner fixes the released `current` anchor
policy. See [benchmark_protocol.md](benchmark_protocol.md) for run-mode and
diagnostic guidance.

To run one method at one nominal target, use `--method` and `--quality`:

```powershell
python .\tools\run_d144.py `
  --problem-root ".\data\shiyan-problem-i2823-p188150-o1073331-g23-c0" `
  --executable .\example_v2\build\Release\example.exe `
  --out .\runs\d144_a2_2px `
  --method A2-Parallax-Mw `
  --quality 2
```

Accepted quality targets are `2`, `5`, `10`, `20`, `50`, `100`, `200`, and
`500`; they select the released tiers rather than generating new noise.
Omitting `--method` runs all 14 methods; omitting `--quality` runs all eight
tiers. `--mode clean-logged` adds convergence and termination logs.
`--mode diagnostic` writes the heavier per-run diagnostic outputs. Use clean
solver times for runtime comparisons and run timing experiments in isolation.
Removing `CEP_STRICT_VECTOR_DIAGNOSTICS` disables inherited diagnostic settings
for clean runs; in diagnostic mode, strict vector diagnostics use the solver's
default enabled setting.

Resume an interrupted experiment by repeating its command with `--resume`.
Keep the same inputs, method/quality selection, mode, and output root. To
recompute checkpoint tables from existing final poses without running BA,
repeat the experiment command with `--evaluate-only`. For example:

```powershell
python .\tools\run_d144.py `
  --problem-root ".\data\shiyan-problem-i2823-p188150-o1073331-g23-c0" `
  --out .\runs\d144_controlled `
  --evaluate-only
```

## Control and checkpoint protocol

The source package's `gcp.txt` marks all 23 surveyed points as controls. The
runner writes a separate experiment split to
`<out>/protocol/gcp_13control_10checkpoint.txt`, preserving the source point
IDs, names, and coordinates. The zero-based IDs held out as checkpoints are:

```text
0, 2, 3, 6, 11, 12, 15, 18, 20, 21
```

The remaining 13 points provide control. Their surveyed 3D coordinates are
fixed constants in camera-projection residuals; they are not optimized object
points. The residuals use image-coordinate errors in pixels with the same
unit weighting and no robust loss as the tie-point residuals. The horizontal
and vertical accuracy fields in `gcp.txt` are retained but are not used to
weight residuals or create 3D coordinate priors. The ten checkpoint coordinates
and their image observations do not enter the BA objective.

For each checkpoint, the evaluator uses its unchanged image observations,
the final optimized camera poses, and fixed calibration to perform multi-ray
least-squares forward intersection. It compares that coordinate directly to
the surveyed coordinate. There is **no post-hoc Sim(3) alignment** in this
controlled evaluation. This differs from the similarity-aligned reference
recovery used for the primary free-network benchmark.

## Outputs and units

`<out>/protocol/manifest.json` records the experiment configuration, selected
inputs, split, execution protocol, and SHA256 fingerprints of the initialization
and static input files. `<out>/protocol/last_execution.json` records the most
recent solver command, executable SHA256, and solver environment. Resume rejects
changed inputs, settings, or executable; use a new output root for those changes.
The solver writes `summary.csv` and a
`FinalPose.txt` for each selected problem-method pair. Checkpoint tables are
written under `<out>/absolute_accuracy/`:

| File | Contents |
| --- | --- |
| `checkpoint_absolute_accuracy_detail.csv` | Individual checkpoint estimates, coordinate differences, 3D errors, and reprojection RMSE |
| `checkpoint_absolute_accuracy_by_dataset_method.csv` | Ten-checkpoint RMSE for each initialization-method pair |
| `checkpoint_absolute_accuracy_by_method.csv` | Checkpoint error aggregated over the selected initialization sets for each method |
| `checkpoint_review_table.csv` | Compact method-level review table |

A full run has 112 initialization-method pairs and 1,120 checkpoint-detail
rows. Each method's pooled `checkpoint_count=80` represents the same ten
physical checkpoints evaluated under eight initializations.

Coordinates, horizontal errors, vertical errors, and 3D RMSE are in **metres**.
Image reprojection RMSE is in **pixels**. Horizontal RMSE is the root mean
square of `sqrt(dx^2 + dy^2)`; vertical RMSE is the root mean square of `dz`;
3D RMSE is the root mean square of `sqrt(dx^2 + dy^2 + dz^2)`. The paper's
excess-checkpoint-RMSE figure converts the difference from the best method in
each tier to millimetres.

The controlled solver's `summary.csv` image RMSE includes tie-point and GCP
image residuals: `sqrt(2 * cost / observations)`, with one observation counting
as one two-coordinate residual block. D144 contributes 201 control image
observations, so the complete controlled objective contains 1,073,532 image
observations. Checkpoint error is evaluated independently of that objective.

## Validation

The small regression network checks control/CP separation, clean-mode final
pose export, direct checkpoint accuracy, and resume with missing or empty pose
files:

```powershell
python .\tools\test_d144_runner.py `
  --executable .\example_v2\build\Release\example.exe
```

See [d144_validation.md](d144_validation.md) for the publication checks and
their scope, including a real D144 adjustment and the archived full evaluation.
