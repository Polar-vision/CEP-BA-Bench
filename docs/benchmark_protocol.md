# Object-Point Parameterization Benchmark Protocol

## Run modes

`clean` mode is the authoritative source for runtime comparisons. It disables
per-run console diagnostics and does not write reports, convergence curves,
poses, or point clouds. The runner writes only the aggregate `summary.csv`
after each solve has finished.
With `--export-final-pose`, it additionally exports final camera poses after
the timed Ceres solve, for held-out checkpoint evaluation without diagnostic
instrumentation. Use `solver_time_sec` for comparisons; `wall_time_sec` also
includes setup and any requested export work.

`clean-logged` mode uses the same solver settings and does not enable strict
vector diagnostics. It additionally writes lightweight per-run logs after
`Solve()` returns: Ceres' per-iteration RMSE and wall-time fields,
`termination_message`, the solver tolerances, and the full solver report.
The Ceres-reported solver time remains separate from report-writing time.

`diagnostic` mode writes `report.txt`, `metrics.json`, `convergence.txt`,
`FinalPose.txt`, and `Final3D.ply`. With
`--point-condition-sample N`, it also evaluates sampled point-block Hessians.
With `--schur-sample N`, it exports the singular spectrum of a sampled reduced
camera Schur system.
It is intended for representative subsets and mechanism analysis, not for the
primary runtime table.

## Metrics for the full benchmark

The following metrics are inexpensive Ceres summary values and can be
collected for all Initial Value problems:

- execution status, Ceres termination status, and solution failure rate;
- initial and final reprojection RMSE;
- LM iteration count and accepted/rejected trial steps;
- total solver time and linear solver time;
- total linear-solver iterations;
- final maximum gradient component and gradient 2-norm.

Execution success only means that the benchmark process returned normally.
Ceres convergence additionally requires `termination_type == CONVERGENCE`.
For matched runs on the same problem, the paper reports final reprojection
RMSE within 1% and 5% of the problem-best final RMSE, independently of Ceres
termination status. A `CONVERGENCE` run may still stop at high RMSE. Strict
first-passage profiles use the earliest accepted logged iterate reaching the
same RMSE target and normalize by the fastest time to that target; unsuccessful
pairs have infinite ratio. Endpoint profiles compare final outputs and use the
fastest total solve as denominator, including runs missing the accuracy band.
The tolerance is on RMSE, so the objective ratios are 1.0201 and 1.1025 with
unchanged observations and identity loss.

For datasets containing independent check points, object-space RMSE should be
reported separately from the reprojection metrics. GCPs used as constraints
must not also be treated as independent check points.

When fixed GCP control residuals are enabled, the control points define the
network datum and the free-network monocular gauge fix should be disabled.
Retain the free-network gauge only for unconstrained runs or explicit
over-constrained diagnostics.

The controlled D144 protocol and its released runner are documented in
[d144_reproduction.md](d144_reproduction.md). Its direct checkpoint evaluation
does not apply a similarity transform.

## Common solver settings

The paper uses Ceres 2.3.0, `SPARSE_SCHUR`, LM, fixed intrinsics, unit image
residual weights, no robust loss, and one thread. Its iteration limit is 100,
with function tolerance `1e-6`, gradient tolerance `1e-10`, parameter tolerance
`1e-8`, initial trust-region radius `1e4`, minimum relative decrease `1e-3`,
and Jacobi scaling. The D144 wrapper explicitly sets supported environment
overrides to these values so inherited settings cannot alter the protocol.
General C++ runs default to the host's thread count unless `CEP_NUM_THREADS`
is set. Diagnostic vector logging is enabled by default only in diagnostic
mode and is excluded from primary timing comparisons.

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

All Initial Value problems should be included in the clean benchmark because
small or easy cases alone can hide differences in convergence basin and failure
rate. Expensive diagnostics can be run either for the full corpus or for a
representative subset that covers the four scene categories and a range of
problem sizes.

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
- mean-rank plots across matched Initial Value problems;
- check-point object-space accuracy for datasets with independent checks;
- a compact table of median performance and statistical ranks.

Recommended Discussion figures:

- representative convergence curves;
- gain-ratio and LM-damping trajectories;
- gradient decay and iterations-to-gradient-threshold;
- conditioning or Schur-spectrum plots for representative cases;
- frame/anchor ablations explaining differences between W, Aw/Ac, and Mw/Mc.

## Implemented methods

The CEP runner exposes all 14 evaluated methods:

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
