# Object-Point Parameterization Benchmark Protocol

## Run modes

`clean` mode is the authoritative source for runtime comparisons. It disables
per-run console diagnostics and does not write reports, convergence curves,
poses, or point clouds. The runner writes only the aggregate `summary.csv`
after each solve has finished.

`diagnostic` mode writes `report.txt`, `metrics.json`, `convergence.txt`,
`FinalPose.txt`, and `Final3D.ply`. With
`--point-condition-sample N`, it also evaluates sampled point-block Hessians.
With `--schur-sample N`, it exports the singular spectrum of a sampled reduced
camera Schur system.
It is intended for representative subsets and mechanism analysis, not for the
primary runtime table.

## Metrics for the full benchmark

The following metrics are inexpensive Ceres summary values and can be
collected for all quality datasets:

- execution status, Ceres termination status, and solution failure rate;
- initial and final reprojection RMSE;
- LM iteration count and accepted/rejected trial steps;
- total solver time and linear solver time;
- total linear-solver iterations;
- final maximum gradient component and gradient 2-norm.

Execution success only means that the benchmark process returned normally.
Ceres convergence additionally requires `termination_type == CONVERGENCE`.
For matched runs on the same quality dataset, an equivalent solution requires
both Ceres convergence and a final reprojection RMSE no more than 1% above the
best final RMSE obtained by any compared method. The relative tolerance must
be reported and sensitivity to this threshold should be checked in the final
paper.

For datasets containing independent check points, object-space RMSE should be
reported separately from the reprojection metrics. GCPs used as constraints
must not also be treated as independent check points.

## Representative diagnostic subset

The following diagnostics should be computed only for a stratified subset of
datasets covering small/medium/large problems, weak/strong geometry, and
several initial-error levels:

- per-iteration RMSE, gradient norms, step norms, and cost decrease;
- LM gain ratio, trust-region radius, and damping parameter;
- point-block and reduced-camera-system condition estimates;
- extremal eigenvalues or singular values of selected Schur systems;
- gradient direction quality and local Lipschitz estimates;
- expensive Jacobian, spectrum, or Schur matrix exports.

These diagnostics must remain disabled in clean timing runs.

All available quality levels should be included in the clean benchmark because
low-error cases alone can hide differences in convergence basin and failure
rate. Expensive diagnostics do not need to be run for every dataset in the
full corpus. They should cover every quality level on a small representative
set of base datasets, or at least low, medium, and high initial-error levels
when the full cross-product is too expensive.

The sampled Schur system is formed from a point-induced camera subgraph, so it
usually contains a larger null space than the full BA system. Report its
non-zero singular spectrum, numerical rank, and non-zero-spectrum condition
number. CEP defines the retained spectrum using
`sigma_i > 1e-8 * sigma_max` and records both the relative tolerance and
absolute threshold in `schur_summary.csv`. Do not interpret the raw nullity as
the nullity of the full dataset.

Point-block condition numbers depend on the local scaling of each
parameterization. They are useful for diagnosing numerical behavior within the
implemented coordinates, but they are not coordinate-invariant geometric
quality measures.

## Results presentation

Recommended main Results figures and tables:

- success/failure rate by parameterization;
- solver-time and linear-solver-time boxplots;
- iteration-count and final-RMSE boxplots;
- mean-rank plots across matched quality datasets;
- check-point object-space accuracy for datasets with independent checks;
- a compact table of median performance and statistical ranks.

Recommended Discussion figures:

- representative convergence curves;
- gain-ratio and LM-damping trajectories;
- gradient decay and iterations-to-gradient-threshold;
- conditioning or Schur-spectrum plots for representative cases;
- frame/anchor ablations explaining differences between W, Aw/Ac, and Mw/Mc.

## Implemented methods

The CEP runner exposes all 14 methods in the paper:

- `A0-XYZ-W`
- `A0-XYInvZ-W`
- `A0-SphRange-W`
- `A0-SphInvRange-W`
- `A1-XYZ-Ac`
- `A1-XYInvZ-Ac`
- `A1-SphRange-Ac`
- `A1-SphInvRange-Ac`
- `A2-Parallax-Mc`
- `A1-XYZ-Aw`
- `A1-XYInvZ-Aw`
- `A1-SphRange-Aw`
- `A1-SphInvRange-Aw` (Civera-type IDP)
- `A2-Parallax-Mw`

The `Ac` methods store the object point in the main-anchor camera frame and
therefore depend on the anchor rotation during reconstruction. The `Aw`
methods translate the origin to the anchor while keeping axes parallel to the
global world frame. `A2-Parallax-Mc` and `A2-Parallax-Mw` use the same
parallax angle but express the main-anchor bearing in different frames.
