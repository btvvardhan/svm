#!/usr/bin/env python3
"""
Classify terrain from a single CSV file using the trained window-based model.

Usage:
  conda activate svm
  python predict_csv.py path/to/your_recording.csv
  python predict_csv.py path/to/file.csv --model svm_window_based.joblib --dt 0.002
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

# Reuse live classifier logic for feature extraction and buffering
from predict_live import LiveTerrainClassifier, FEATURE_COLS


def infer_dt(df: pd.DataFrame) -> float:
    """Infer sampling interval from 'time' column if present."""
    if "time" not in df.columns:
        return 0.002
    t = pd.to_numeric(df["time"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(t) < 2:
        return float(t[0]) if len(t) == 1 and t[0] > 0 else 0.002
    if np.nanstd(t) < 1e-12 and t[0] > 0:
        return float(t[0])
    diffs = np.diff(t)
    pos = diffs[diffs > 0]
    if len(pos) > 0:
        dt = float(np.median(pos))
        if dt > 0:
            return dt
    return 0.002


def classify_csv(
    csv_path: Path,
    model_path: Path,
    dt: float | None = None,
) -> tuple[str, list[str], float]:
    """
    Run window-based classifier on a CSV and return majority terrain and per-window predictions.

    Returns:
        (majority_label, list_of_window_predictions, dt_used)
    """
    df = pd.read_csv(csv_path)
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}. Required: {FEATURE_COLS}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("CSV has no valid rows after dropping missing FSR values.")

    if dt is None:
        dt = infer_dt(df)

    clf = LiveTerrainClassifier(model_path, dt=dt)
    preds = []
    for _, row in df.iterrows():
        clf.push(row["fsr1"], row["fsr2"], row["fsr3"], row["fsr4"])
        p = clf.predict_if_ready(step=True)
        if p is not None:
            preds.append(p)

    if not preds:
        raise ValueError(
            f"Not enough samples for one window (need at least {clf.window_size} rows; got {len(df)})."
        )

    majority = max(set(preds), key=preds.count)
    return majority, preds, dt


def main():
    p = argparse.ArgumentParser(
        description="Classify terrain (hard_ground vs grass) from a single CSV file.",
    )
    p.add_argument(
        "csv",
        type=Path,
        help="Path to CSV file with columns fsr1, fsr2, fsr3, fsr4.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "svm_window_based.joblib",
        help="Path to trained model (default: svm_window_based.joblib in this directory).",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Sampling interval in seconds. If omitted, inferred from 'time' column.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-window counts (hard_ground / grass).",
    )
    args = p.parse_args()

    if not args.csv.exists():
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if not args.model.exists():
        print(f"Error: model not found: {args.model}. Run train_svm.py first.", file=sys.stderr)
        sys.exit(1)

    try:
        majority, preds, dt_used = classify_csv(args.csv, args.model, args.dt)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Terrain: {majority}")
    if args.verbose:
        n_hard = preds.count("hard_ground")
        n_grass = preds.count("grass")
        print(f"Windows: {len(preds)} (hard_ground: {n_hard}, grass: {n_grass})")
        print(f"dt: {dt_used:.6f}")


if __name__ == "__main__":
    main()
