#!/usr/bin/env python3
"""
Live terrain classifier for robot: hard_ground vs grass.

Use the window-based SVM model to classify terrain from a stream of FSR
readings (fsr1, fsr2, fsr3, fsr4). Push samples in real time; get a prediction
once a full window is buffered (and optionally every step_size new samples).

Usage:
  # From Python (e.g. on robot):
  from predict_live import LiveTerrainClassifier
  clf = LiveTerrainClassifier("svm_window_based.joblib", dt=0.002)
  clf.push(fsr1, fsr2, fsr3, fsr4)
  label = clf.predict_if_ready()  # None until window full, then "hard_ground" or "grass"

  # Test from a CSV file (simulates streaming):
  python predict_live.py --csv grass/grass_1.csv
  python predict_live.py --csv hard_ground/hard_ground_1.csv
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import joblib

# Must match train_svm.py exactly for correct features
FEATURE_COLS = ["fsr1", "fsr2", "fsr3", "fsr4"]


def _series_features(x: np.ndarray, dt: float, prefix: str) -> dict:
    """Compute time-dynamic features for one 1D signal window (same as train_svm.py)."""
    x = x.astype(float)
    eps = 1e-9

    feats = {}
    feats[f"{prefix}mean"] = float(np.mean(x))
    feats[f"{prefix}std"] = float(np.std(x, ddof=0))
    feats[f"{prefix}min"] = float(np.min(x))
    feats[f"{prefix}max"] = float(np.max(x))
    feats[f"{prefix}median"] = float(np.median(x))
    feats[f"{prefix}ptp"] = float(np.ptp(x))
    feats[f"{prefix}rms"] = float(np.sqrt(np.mean(x * x)))

    if len(x) >= 2 and dt > 0:
        dx = np.diff(x) / dt
        adx = np.abs(dx)
        feats[f"{prefix}d_mean_abs"] = float(np.mean(adx))
        feats[f"{prefix}d_max_abs"] = float(np.max(adx))
        feats[f"{prefix}d_std"] = float(np.std(dx, ddof=0))
        feats[f"{prefix}total_variation"] = float(np.sum(adx))
        feats[f"{prefix}sharpness_norm"] = float(np.max(adx) / (np.ptp(x) + eps))
    else:
        feats[f"{prefix}d_mean_abs"] = 0.0
        feats[f"{prefix}d_max_abs"] = 0.0
        feats[f"{prefix}d_std"] = 0.0
        feats[f"{prefix}total_variation"] = 0.0
        feats[f"{prefix}sharpness_norm"] = 0.0

    if len(x) >= 3 and dt > 0:
        dx = np.diff(x) / dt
        ddx = np.diff(dx) / dt
        feats[f"{prefix}dd_max_abs"] = float(np.max(np.abs(ddx)))
        feats[f"{prefix}dd_std"] = float(np.std(ddx, ddof=0))
    else:
        feats[f"{prefix}dd_max_abs"] = 0.0
        feats[f"{prefix}dd_std"] = 0.0

    return feats


def _window_to_features(win: np.ndarray, dt: float, feature_cols: list) -> dict:
    """Build one row of features from a (window_size, n_cols) array. Same order as training."""
    feat = {}
    for i, c in enumerate(feature_cols):
        feat.update(_series_features(win[:, i], dt, prefix=f"{c}_"))
    total = np.sum(win, axis=1)
    feat.update(_series_features(total, dt, prefix="total_"))
    return feat


class LiveTerrainClassifier:
    """
    Buffers FSR samples and runs the window-based SVM for live terrain classification.
    """

    def __init__(self, model_path: str | Path, dt: float = 0.002):
        """
        Load the window-based model and set sampling interval.

        Args:
            model_path: Path to svm_window_based.joblib (from train_svm.py).
            dt: Sampling interval in seconds (default 0.002 = 500 Hz). Must match your robot's FSR rate.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.window_size = int(bundle["window_size"])
        self.step_size = int(bundle.get("step_size", self.window_size))
        self.feature_names = list(bundle["feature_names"])
        self.feature_cols = list(bundle.get("feature_cols", FEATURE_COLS))
        self.dt = float(dt)

        self._buffer = []  # list of [fsr1, fsr2, fsr3, fsr4]
        self._last_step = -1  # index of last sample used for a prediction (for step_size)

    def push(self, fsr1: float, fsr2: float, fsr3: float, fsr4: float) -> None:
        """Append one FSR sample. Call this at your sensor rate (e.g. every 2 ms)."""
        self._buffer.append([float(fsr1), float(fsr2), float(fsr3), float(fsr4)])

        # Keep only the last window_size samples to avoid unbounded growth
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def push_row(self, row: dict | list) -> None:
        """Append one sample from a dict with keys fsr1..fsr4 or a list [fsr1,fsr2,fsr3,fsr4]."""
        if isinstance(row, dict):
            self.push(
                row["fsr1"], row["fsr2"], row["fsr3"], row["fsr4"]
            )
        else:
            self.push(row[0], row[1], row[2], row[3])

    def ready(self) -> bool:
        """True if we have at least window_size samples and can predict."""
        return len(self._buffer) >= self.window_size

    def predict_if_ready(self, step: bool = True) -> str | None:
        """
        If buffer has at least window_size samples, compute features and return class.
        If step=True (default), only return a new prediction every step_size new samples
        (sliding window like in training); otherwise predict every time ready() is True.

        Returns:
            "hard_ground" or "grass", or None if not ready or (step=True and not at step boundary).
        """
        if not self.ready():
            return None

        n = len(self._buffer)
        # When step=True, predict only when we have advanced by step_size since last prediction
        if step and self._last_step >= 0 and (n - 1 - self._last_step) < self.step_size:
            return None

        win = np.array(self._buffer[-self.window_size :], dtype=float)
        feat = _window_to_features(win, self.dt, self.feature_cols)
        row = pd.DataFrame([feat]).fillna(0.0)
        # Ensure column order matches training
        row = row[self.feature_names]
        pred = self.pipeline.predict(row)[0]
        if step:
            self._last_step = n - 1
        return str(pred)

    def reset(self) -> None:
        """Clear the buffer (e.g. after a stop or terrain change)."""
        self._buffer.clear()
        self._last_step = -1


def run_csv_test(csv_path: Path, model_path: Path, dt: float) -> None:
    """Simulate live streaming from a CSV and print predictions."""
    df = pd.read_csv(csv_path)
    for c in FEATURE_COLS:
        if c not in df.columns:
            print(f"Error: {csv_path} missing column {c}", file=sys.stderr)
            sys.exit(1)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)

    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce").dropna()
        if len(t) >= 2 and np.nanstd(t) >= 1e-12:
            dt = float(np.median(np.diff(t)))
        elif len(t) >= 1 and t.iloc[0] > 0:
            dt = float(t.iloc[0])
    print(f"Using dt={dt:.6f}")

    clf = LiveTerrainClassifier(model_path, dt=dt)
    preds = []
    for _, row in df.iterrows():
        clf.push(row["fsr1"], row["fsr2"], row["fsr3"], row["fsr4"])
        p = clf.predict_if_ready(step=True)
        if p is not None:
            preds.append(p)

    if not preds:
        print("No predictions (not enough samples for one window).")
        return
    majority = max(set(preds), key=preds.count)
    print(f"Predictions: {len(preds)} windows -> majority: {majority}")
    print(f"  (hard_ground: {preds.count('hard_ground')}, grass: {preds.count('grass')})")


def main():
    p = argparse.ArgumentParser(description="Live terrain classifier (hard_ground vs grass).")
    p.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "svm_window_based.joblib",
                   help="Path to svm_window_based.joblib")
    p.add_argument("--csv", type=Path, default=None,
                   help="Test on a CSV file (stream rows and print majority prediction).")
    p.add_argument("--dt", type=float, default=0.002, help="Sampling interval in seconds (default 0.002).")
    args = p.parse_args()

    if args.csv is not None:
        run_csv_test(args.csv, args.model, args.dt)
    else:
        print("Usage: python predict_live.py --csv path/to/file.csv")
        print("Or import LiveTerrainClassifier and use push() / predict_if_ready() in your robot code.")


if __name__ == "__main__":
    main()
