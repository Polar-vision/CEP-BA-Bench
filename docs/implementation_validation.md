# CEP Benchmark Implementation and Validation

## Scope completed

- Implemented all 14 paper-defined object-point parameterizations.
- Added `Ac` camera-frame and `Mc` camera-frame parallax residuals.
- Corrected `XYInvZ` to use `[xi, eta, 1]^T / rho`.
- Corrected the parallax sine-rule expression in the manuscript to
  `sin(beta + omega) / sin(omega)` for the stated baseline and ray convention.
- Added clean timing and diagnostic execution modes.
- Added CSV, JSON, convergence, point-block conditioning, and sampled Schur
  spectrum outputs.
- Added Python aggregation, LaTeX table, rank, failure-rate, runtime,
  convergence, point-conditioning, and Schur-spectrum plots.
- Kept all solver changes inside CEP; `ceres-solver` source was not modified.

## Build

```powershell
cmake --build E:\zuo\projects\CEP\example_v2\build --config Release
```

The verified executable is:

```text
E:\zuo\projects\CEP\example_v2\build\Release\example.exe
```

## Clean timing run

```powershell
E:\zuo\projects\CEP\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_14methods_verified `
  --limit 1 `
  --mode clean
```

Validation dataset:

```text
abs-problem-i370-p24349-o103103-g9-c0-i370-p24349-o103103-g9-init-rmse005p00px
```

The dataset contains 370 cameras, 24,349 tie points, and 103,103 image
observations. All 14 methods completed successfully. The initial-cost spread
across parameterizations was approximately `2.1e-9`, confirming equivalent
initial geometry. The final reprojection-RMSE spread was approximately
`4.23e-7 px`, confirming that all methods reached effectively equivalent
solutions on the smoke-test case.

Clean mode generated exactly one file:

```text
benchmark_14methods_verified\summary.csv
```

No report, convergence, point-cloud, Jacobian, conditioning, or Schur files
were generated during the timing run. Use `solver_time_sec` and
`linear_solver_time_sec` for the paper's runtime comparisons.

## Diagnostic run

```powershell
E:\zuo\projects\CEP\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_schur_smoke `
  --limit 1 `
  --method A2-Parallax-Mc `
  --mode diagnostic `
  --point-condition-sample 16 `
  --schur-sample 16
```

This run generated:

- `report.txt`
- `metrics.json`
- `convergence.txt`
- `FinalPose.txt`
- `Final3D.ply`
- `point_block_conditioning.csv`
- `schur_summary.csv`
- `schur_spectrum.csv`

For the 16 sampled Parallax-Mc point blocks, all sampled blocks were full rank.
The sampled point-block condition numbers were approximately 2.3 to 19.8.
The sampled reduced-camera Schur system had 348 columns and a non-zero-spectrum
condition number of approximately 113. The large sampled-system nullity is
expected because the point-induced camera subgraph is underconstrained and
retains gauge freedoms; it must not be interpreted as the nullity of the full
BA problem.

## Plot generation

```powershell
python E:\zuo\projects\CEP\tools\plot_benchmark_results.py `
  --summary E:\zuo\projects\CEP\benchmark_14methods_verified\summary.csv `
  --diagnostic-root E:\zuo\projects\CEP\benchmark_schur_smoke `
  --out E:\zuo\projects\CEP\benchmark_figures_schur
```

The script generates aggregate CSV and LaTeX tables together with runtime,
failure-rate, iteration, RMSE, convergence, point-conditioning, and Schur
spectrum figures.

## Full benchmark versus diagnostic subset

Run on all quality datasets:

- success and termination status;
- initial/final cost and reprojection RMSE;
- solver and linear-solver time;
- iteration and accepted/rejected-step counts;
- final gradient norms.

Run only on a representative diagnostic subset:

- per-iteration gain ratio, damping, gradients, and step norms;
- sampled point-block eigenvalues and condition numbers;
- sampled reduced-camera Schur singular spectrum;
- future gradient-direction and local-Lipschitz diagnostics.

## All-quality validation

The single base dataset currently stored under `PVL-BA-Bench` contains seven
quality datasets:

- pose-perturbation levels at 5, 10, 20, 50, and 100 px;
- joint camera-and-point perturbation levels at 200 and 500 px.

The authoritative clean run covered all 14 methods at all seven levels:

```powershell
E:\zuo\projects\CEP\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_all_quality_clean `
  --mode clean
```

The mechanism-analysis run used the same 98 matched cases with 16 sampled
point blocks and 16 sampled points for the reduced Schur system:

```powershell
E:\zuo\projects\CEP\example_v2\build\Release\example.exe `
  --problems E:\zuo\projects\CEP\PVL-BA-Bench `
  --out E:\zuo\projects\CEP\benchmark_all_quality_diagnostics `
  --mode diagnostic `
  --point-condition-sample 16 `
  --schur-sample 16
```

All 98 processes returned normally. Using Ceres convergence plus a final-RMSE
tolerance of 1% relative to the best matched method, nine runs were classified
as solution failures. One occurred at 50 px and eight occurred at 100 px.
The 5, 10, and 20 px cases were therefore insufficient by themselves to expose
the main differences in convergence basin.

The following methods reached an equivalent solution at all seven levels:

- `A1-XYInvZ-Ac`;
- `A1-SphInvRange-Ac`;
- `A1-XYInvZ-Aw`;
- `A1-SphInvRange-Aw` (Civera-type IDP);
- `A2-Parallax-Mc`;
- `A2-Parallax-Mw`.

`A0-XYInvZ-W` reached the maximum iteration count at both 50 and 100 px.
Several XYZ and spherical-range variants reported Ceres convergence at 100 px
but terminated at a worse local solution, demonstrating why termination status
and final-solution equivalence must be reported separately.

The Schur numerical-rank threshold was changed from a boundary-sensitive
relative tolerance of `1e-10` to `1e-8`. The previous threshold lay directly
on the near-zero spectrum of `A0-XYInvZ-W`, causing the reported condition
number to switch between approximately `1e2` and `1e10` under small numerical
changes. With the explicit `1e-8` retained-spectrum threshold, all 98 sampled
Schur condition numbers lie between approximately 111.70 and 113.21. This is
consistent with the expected near-invariance of the reduced camera system to
invertible object-point reparameterization.

Paper-ready outputs are generated with:

```powershell
python E:\zuo\projects\CEP\tools\plot_benchmark_results.py `
  --summary E:\zuo\projects\CEP\benchmark_all_quality_clean\summary.csv `
  --diagnostic-root E:\zuo\projects\CEP\benchmark_all_quality_diagnostics `
  --out E:\zuo\projects\CEP\benchmark_figures_all_quality `
  --max-curves 200 `
  --solution-rmse-relative-tolerance 0.01
```

The output includes quality-level performance curves, solution and solver
failure rates, point-conditioning heatmaps, Schur-condition heatmaps, and
representative/all-quality Schur spectra. Convergence plots are restricted to
quality levels that contain at least one negative LM gain-ratio step; the
selection is recorded in `diagnostic_quality_selection.csv`. For the current
validation dataset, this selects the 50 and 100 px pose-perturbation cases.
The script first compares methods within four groups: A0, A1-Ac, A1-Aw, and
A2. It then selects one winner per group using the following lexicographic
priority: solution-success rate, Ceres convergence rate, median iterations,
and median solver time. The selected rows are written to `group_winners.csv`
and compared in the main convergence figure. Separate group figures support
the within-group analysis.

A frame-ablation figure compares matched Ac/Aw or Mc/Mw pairs for XYZ,
spherical range, spherical inverse range, and parallax. The
`A1-XYInvZ-Ac` and `A1-XYInvZ-Aw` are included as a matched frame-ablation
pair because both optimize normalized planar direction coordinates and inverse
$Z$ in their respective frames. The declared pairs are stored in
`frame_ablation_pairs.csv`.
A complete 14-method figure is retained for supplementary review. Metrics and
raw convergence logs remain available for all seven quality levels.

## Remaining limitations

- The validation dataset contains 9 GCPs and no independent check points.
  Its GCP observations are stored separately from the tie-point BA files, so
  independent object-space check-point accuracy was not validated here.
- GCP-constrained absolute BA and check-point RMSE require a separate control
  point residual path and should not be inferred from `original/XYZ.txt`.
- The current Euler-angle camera representation is intentionally held fixed
  across all methods; this benchmark does not compare camera-rotation
  parameterizations.

## Unified point-block and gauge correction

The benchmark implementation was corrected before restarting the public
release experiment:

- `A2-Parallax-Mw` now registers azimuth, elevation, and parallax angle as one
  three-dimensional Schur point block.
- `A1-SphRange-Aw` and `A1-SphInvRange-Aw` now likewise use one
  three-dimensional point block instead of separate direction and range
  blocks.
- Pure monocular BA fixes the first camera pose and the dominant baseline
  component of the second valid camera by default. Set
  `CEP_FIX_MONOCULAR_GAUGE=0` only for a separately constrained formulation.
- Exact ray-baseline collinearity is regularized at machine scale in the
  parallax residuals to prevent `sqrt(0)` from producing NaN automatic
  derivatives.

The old partial public-release run was stopped after 644 rows. Its timings,
iterations, conditioning diagnostics, and failure rates are not comparable
with the corrected implementation and must not be used in the paper.

## Corrected synthetic validation

All methods produced the same initial reprojection RMSE in the focused
regression scene, and all 14 runs completed normally.

The IJRR-inspired stress suite showed a clear convergence distinction:

- distant features: median iterations were 10 for `A0-XYZ-W`, 5 for
  `A1-SphInvRange-Aw`, and 5 for both A2 methods;
- forward two-view motion: median iterations were 30 for `A0-XYZ-W`, 25 for
  `A1-SphInvRange-Aw`, and 14 for both A2 methods.

The corrected 212-scene benchmark contains 2,968 method runs. All processes
completed normally. The principal aggregate results are:

- `A0-XYZ-W`: 92.45% solution success, 77.83% formal convergence, and
  39 median iterations;
- `A2-Parallax-Mc`: 95.28% solution success, 87.26% formal convergence, and
  17 median iterations;
- `A2-Parallax-Mw`: 96.23% solution success, 91.04% formal convergence, and
  19 median iterations.

Within the 144 parallax-depth scenes, both A2 methods achieved 100% solution
success, compared with 97.22% for `A0-XYZ-W`. The corrected
`A2-Parallax-Mw` median relative time decreased from approximately 12.0x in
the split-block implementation to approximately 1.76x after unification.
This confirms that the former timing disadvantage was primarily an
implementation artifact.

The enlarged 12-camera, 600-point suite contains 84 runs. A previously failing
short-track `A2-Parallax-Mc` case now completes without non-finite Jacobians.
The enlarged suite remains deliberately difficult: solution success is 83.33%
for `A0-XYZ-W`, 50.00% for `A2-Parallax-Mc`, and 66.67% for
`A2-Parallax-Mw`. Thus, the corrected evidence supports an A2 convergence
advantage in weak-depth cases, but not universal superiority across short
tracks, severe pose perturbations, or extreme coordinate scales.

Current paper-ready synthetic outputs are stored in:

- `synthetic_benchmark/analysis/method_summary.csv`;
- `synthetic_benchmark/analysis/family_summary.csv`;
- `synthetic_benchmark/analysis/conclusions_zh.md`;
- `parallaxba_ijrr_reproduction/analysis/`.
