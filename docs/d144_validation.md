# D144 publication validation

Validated on 2026-10-12 using Windows/MSVC 2022 x64, Ceres 2.3.0,
Eigen 3.4.0, Python 3.11.7, and a single solver thread.

## Build and regression checks

The root and `example_v2` CMake entry points both configured and built in
Release mode using installed dependency packages. The final source was rebuilt
from clean objects through the root entry point.

`tools/test_d144_runner.py` passed all five integration tests on a small,
deterministic network with four cameras, 24 tie points, 13 controls, and ten
held-out checkpoints. Both A0-XYZ-W and A2-Parallax-Mc performed actual BA:

- Clean mode exported final poses without diagnostic output.
- Direct checkpoint RMSE was below `1e-4` m for both methods.
- Resume skipped complete runs and repaired missing or empty pose files without
  duplicate summary rows.
- Changing only checkpoint coordinates and image observations left BA costs,
  iteration counts, and final camera poses unchanged, while checkpoint errors
  changed. This verifies that checkpoint information is held out of adjustment.

## D144 data and numerical checks

The wrapper validated all eight released D144 tiers, their generation metadata,
unchanged image observations/calibration, and the published ID-based split.
It preserved the source point IDs, names, coordinate tokens, and accuracy
metadata, changing only checkpoint-role flags. The split has 201 control and
124 checkpoint image observations.

A fresh controlled adjustment was executed with the published wrapper for
`A2-Parallax-Mw` and the nominal 2 px tier in `clean` mode. Its checked endpoint
metrics and direct checkpoint errors exactly matched the paper's archived
single-thread result:

| Metric | Value |
| --- | ---: |
| Final image RMSE (px) | 1.298839299192032 |
| Recorded iterations | 3 |
| Direct checkpoint 3D RMSE (m) | 0.16297291603364156 |

Repeating this wrapper command with `--resume` attempted zero solves and
retained exactly one successful row.

The full wrapper evaluation was also run on copies of the paper's archived
112 final poses and corresponding summary. It verified 112 unique
initialization-method pairs and 1,120 checkpoint estimates. All five coordinate
error metrics (X, Y, Z, horizontal, and 3D RMSE) exactly matched the archived
direct evaluator's values.

The 112 adjustments were **not all rerun** during publication validation.
The fresh solve checks the complete runner path; the archived evaluation checks
the full eight-tier, 14-method output coverage. Timing values and numerical
paths across different dependency builds or hosts require separate assessment.
