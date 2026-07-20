#!/usr/bin/env python3
"""Run CEP diagnostic mode for synthetic boundary cases."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\zuo\projects\CEP\synthetic_benchmark"),
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path(r"E:\zuo\projects\CEP\example_v2\build\Release\example.exe"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    selection = pd.read_csv(root / "analysis" / "diagnostic_selection.csv")
    data_root = root / "datasets"
    output_root = root / "diagnostic_results"
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"

    with log_path.open("a", encoding="utf-8") as log:
        for index, row in selection.iterrows():
            quality = row["quality_dataset"]
            print(f"[{index + 1}/{len(selection)}] {quality}", flush=True)
            command = [
                str(args.exe),
                "--problems",
                str(data_root),
                "--out",
                str(output_root),
                "--dataset",
                quality,
                "--mode",
                "diagnostic",
                "--resume",
            ]
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=log,
                text=True,
                check=False,
            )
            log.flush()
            if completed.returncode != 0:
                raise SystemExit(
                    f"Diagnostic benchmark failed for {quality}: "
                    f"{completed.returncode}"
                )
    print(f"Diagnostic results: {output_root}")


if __name__ == "__main__":
    main()
