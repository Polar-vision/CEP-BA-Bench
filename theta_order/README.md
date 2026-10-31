# Reproduction package

Code for *Gradient order, not information: a Θ-order classification of
object-point parameterizations for bundle adjustment*.

Every number, table and figure in the manuscript is produced by the five scripts
below. No input data are needed: all scenes are generated in code from analytic
geometry. Nothing is compiled, nothing is downloaded.

## Requirements

Python 3.9 or newer with NumPy and Matplotlib. No other dependencies.

```
python -m pip install -r requirements.txt
```

Tested on Windows 11 with Python 3.13.5, NumPy 2.3.2 and Matplotlib 3.10.5. The
code uses nothing outside the NumPy and Matplotlib APIs that has changed since
NumPy 1.22, so older stacks should work, but only the above was verified.

## Run order

| # | script | wall time | writes | produces |
|---|---|---|---|---|
| 1 | `verify_scenarios.py` | 1 s | stdout | Table 4, Table 5, the data behind Fig 3 |
| 2 | `mechanism.py` | 6 s | `mechanism.json` | Table 7, Fig 6 |
| 3 | `mechanism2.py` | 15 s | `mechanism2.json` | Table 8, Fig 7 |
| 4 | `mechanism3.py` | 212 s | `mechanism3.json` | Table 9, Fig 8 |
| 5 | `make_figs.py` | 5 s | `figs/*.pdf`, `figdata.json` | all eight figures |

```
python verify_scenarios.py
python mechanism.py
python mechanism2.py
python mechanism3.py
python make_figs.py
```

Steps 1–4 are independent. Step 5 needs the four `*.json` files, which are
shipped pre-computed so that the figures can be rebuilt without re-running the
optimisation. Total cold run: about four minutes, of which `mechanism3.py`
accounts for 86 per cent (forty random starts times five charts times an IRLS
loop with a graduated non-convexity schedule).

## Determinism

`verify_scenarios.py` contains no random numbers at all: the analytic checks and
the float32 experiments are deterministic functions of the geometry.

The three mechanism scripts seed an explicit generator and never touch the global
NumPy state:

- `default_rng(11)` — Monte Carlo measurement noise in `mechanism.exp_E`, and the
  contamination mask in `RobustProblem`;
- `default_rng(7)` — the default initialisation perturbation in `init_state`;
- `default_rng(9000 + start)` — the initialisation parameters and the inlier noise
  for start number `start`, drawn once and reused by all five charts, so a given
  start index means the same perturbation and the same noise in every chart. This
  pairing is what makes the chart-to-chart comparison in Tables 8 and 9 valid.

Two runs on the same machine reproduce each `*.json` bit for bit except for the
recorded wall-clock field `secs`. The figure PDFs come out byte-identical in size
and differ only in the `CreationDate` that matplotlib writes into the file. Across
machines the last few digits of a few hundred Levenberg–Marquardt iterations may
differ in the last ulp; nothing quoted in the paper is closer than four
significant digits to that level.

## What to check after a run

If the package is working, the following appear in the output. Each is quoted in
the manuscript.

| quantity | value | where |
|---|---|---|
| analytic self-check of the parallax bound, ratio measured/predicted | 1.0000 | `verify_scenarios.py`, Table 5 |
| spread of the covariance block across the five charts | ≤ 1.8e-12 | `make_figs.py`, Fig 5 |
| same, weighted, under the robust cost | ≤ 2.8e-12 | `mechanism3.py`, `I.dev` |
| radial depth standard deviation, all five charts | 0.028465 m | `mechanism3.py`, `I.sigma_depth` |
| median relative depth error, four reciprocal charts, least squares, 10 per cent gross errors | 11.61 % | `mechanism3.py`, `J_deg.l2` |
| same, inlier reprojection RMS | 4.31 px | `mechanism3.py`, `J_deg.l2` |
| same, Tukey biweight | 0.88 % / 0.64 px | `mechanism3.py`, `J_deg.tukey` |
| Cartesian chart, Tukey biweight | 3.64 % / 0.89 px | `mechanism3.py`, `J_deg.tukey` |
| starts reaching the lowest-cost basin, short baseline, 10 per cent gross errors | 40/40 reciprocal, 0/40 Cartesian | `mechanism3.py`, `K_deg_contam` (cost criterion) |

The four reciprocal charts agree with one another to the digits shown. That
agreement is the point of the experiment, not a bug: the estimator is blind to
the chart, so it fails in exactly the same way in all four.

## File map

```
verify_scenarios.py   Scene class, analytic Jacobians, the eight application
                      scenarios, and the float32 experiments
mechanism.py          gradient order vs conditioning: single-step error, budget
                      sweep, start-quality sweep
mechanism2.py         joint adjustment: Levenberg--Marquardt with Nielsen gain
                      ratio acceptance; rejected steps and basins
mechanism3.py         robust estimation: Tukey biweight and Huber kernels, IRLS,
                      graduated non-convexity, weighted CRB
make_figs.py          the eight figures, all data taken from the scripts above
*.json                pre-computed results, one file per mechanism script
figs/                 figure output (PDF, vector)
```

`verify_scenarios.py` defines `Scene`, the single geometry object that every
other script imports, so the scenes behind the tables and the scenes behind the
figures are the same objects by construction rather than by careful copying.

## Note on what is not here

An outlier-magnitude sweep (gross errors from 0 to 3000 px) was run during
development and is not reported in the paper; the manuscript uses a single
contamination level of 30–300 px, stated in Appendix A. The sweep is therefore
not part of this package.
