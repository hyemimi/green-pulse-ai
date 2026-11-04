"""Create a causally imputed dataset for real-time anomaly detection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ["timestamp", "operating_regime", "reactor_id"]
TARGET_COLUMNS = ["fault_type", "efficiency_loss_pct", "time_to_fault_min"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(input_path: Path, output_path: Path, metadata_path: Path, max_gap: int) -> None:
    data = pd.read_csv(input_path, parse_dates=["timestamp"])
    data = data.sort_values(["reactor_id", "timestamp"], kind="stable").reset_index(drop=True)

    sensor_columns = [
        column
        for column in data.columns
        if column not in ID_COLUMNS + TARGET_COLUMNS
    ]
    original = data.copy()
    missing_before = original[sensor_columns].isna().sum()

    for column in sensor_columns:
        data[f"{column}__was_missing"] = original[column].isna().astype("int8")

    # Causal imputation: only prior values from the same reactor are available.
    data[sensor_columns] = (
        data.groupby("reactor_id", sort=False)[sensor_columns]
        .ffill(limit=max_gap)
    )

    missing_after = data[sensor_columns].isna().sum()

    observed_unchanged = True
    for column in sensor_columns:
        observed = original[column].notna()
        observed_unchanged &= np.allclose(
            original.loc[observed, column],
            data.loc[observed, column],
            equal_nan=True,
        )

    if len(data) != len(original):
        raise RuntimeError("Row count changed during preprocessing.")
    if not original[ID_COLUMNS + TARGET_COLUMNS].equals(data[ID_COLUMNS + TARGET_COLUMNS]):
        raise RuntimeError("Identifier or target columns changed during preprocessing.")
    if not observed_unchanged:
        raise RuntimeError("At least one originally observed sensor value changed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    data.to_csv(temporary_output, index=False, compression="gzip")
    temporary_output.replace(output_path)

    metadata = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "policy": {
            "mode": "realtime_causal",
            "group_key": "reactor_id",
            "method": "forward_fill",
            "max_gap_minutes": max_gap,
            "uses_future_values": False,
            "missing_flags_added": True,
            "targets_imputed": False,
        },
        "shape": {"rows": int(len(data)), "columns": int(data.shape[1])},
        "sensor_columns": sensor_columns,
        "target_columns": TARGET_COLUMNS,
        "missing_before": {k: int(v) for k, v in missing_before.items()},
        "missing_after": {k: int(v) for k, v in missing_after.items()},
        "filled_values": {
            k: int(missing_before[k] - missing_after[k]) for k in sensor_columns
        },
        "validation": {
            "row_count_preserved": True,
            "identifiers_and_targets_unchanged": True,
            "observed_sensor_values_unchanged": True,
        },
        "sha256": {
            "input": sha256(input_path),
            "output": sha256(output_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Filled sensor values: {int((missing_before - missing_after).sum()):,}")
    print(f"Remaining sensor NaNs: {int(missing_after.sum()):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chemical_process_timeseries.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chemical_process_timeseries_realtime_ffill.csv.gz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/chemical_process_timeseries_realtime_ffill.metadata.json"),
    )
    parser.add_argument("--max-gap", type=int, default=6)
    args = parser.parse_args()

    preprocess(args.input, args.output, args.metadata, args.max_gap)


if __name__ == "__main__":
    main()
