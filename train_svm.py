#!/usr/bin/env python3
"""
SVM surface classifier: hard_ground vs grass.

Key point:
SVM cannot directly "see" time-evolution unless we convert the time series
into features that represent fluctuation over time. This script does that
using sliding windows + derivative-based features.

- Train/val split by FILE (no leakage).
- Window-based model (recommended for spike shape).
- File-summary model (one sample per file) is kept as a lightweight baseline.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# --------- Config ----------
DATA_ROOT = Path(__file__).resolve().parent  # expects: ./grass/*.csv and ./hard_ground/*.csv
OUT_DIR = DATA_ROOT

CLASSES = ["hard_ground", "grass"]
FEATURE_COLS = ["fsr1", "fsr2", "fsr3", "fsr4"]

MODEL_WINDOW = OUT_DIR / "svm_window_based.joblib"
MODEL_FILE_SUMMARY = OUT_DIR / "svm_file_summary.joblib"
METRICS_FILE = OUT_DIR / "validation_metrics.txt"
METRICS_JSON = OUT_DIR / "validation_results.json"


def load_data(data_root: Path) -> pd.DataFrame:
    """Load all CSVs; label from folder name; file_id from filename stem."""
    dfs = []
    all_csvs = []
    for cls in CLASSES:
        folder = data_root / cls
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        for csv_path in sorted(folder.glob("*.csv")):
            all_csvs.append((cls, csv_path))

    for cls, csv_path in tqdm(all_csvs, desc="Loading CSVs", unit="file"):
        df = pd.read_csv(csv_path)

        missing = set(FEATURE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")

        # numeric conversion
        for c in FEATURE_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # optional time column
        if "time" in df.columns:
            df["time"] = pd.to_numeric(df["time"], errors="coerce")

        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        df["surface"] = cls
        df["file_id"] = csv_path.stem
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    return data


def infer_dt(df: pd.DataFrame) -> float:
    """
    Infer sampling interval dt.
    Handles two common cases:
    1) time column is absolute and increasing -> dt = median(diff(time))
    2) time column stores a constant dt (e.g., 0.002 repeated) -> dt = that value
    """
    if "time" not in df.columns:
        return 1.0

    t = df["time"].dropna().to_numpy(dtype=float)
    if len(t) < 2:
        return float(t[0]) if len(t) == 1 and t[0] > 0 else 1.0

    # Case: constant dt stored in "time" column (repeated value)
    if np.nanstd(t) < 1e-12 and t[0] > 0:
        return float(t[0])

    # Case: absolute time increasing
    diffs = np.diff(t)
    pos = diffs[diffs > 0]
    if len(pos) > 0:
        dt = float(np.median(pos))
        if dt > 0:
            return dt

    return 1.0


def series_features(x: np.ndarray, dt: float, prefix: str) -> dict:
    """Compute time-dynamic features for one 1D signal window."""
    x = x.astype(float)
    eps = 1e-9

    feats = {}
    feats[f"{prefix}mean"] = float(np.mean(x))
    feats[f"{prefix}std"] = float(np.std(x, ddof=0))
    feats[f"{prefix}min"] = float(np.min(x))
    feats[f"{prefix}max"] = float(np.max(x))
    feats[f"{prefix}median"] = float(np.median(x))
    feats[f"{prefix}ptp"] = float(np.ptp(x))  # max-min
    feats[f"{prefix}rms"] = float(np.sqrt(np.mean(x * x)))

    # Derivative features = the "spike sharpness" signals
    if len(x) >= 2 and dt > 0:
        dx = np.diff(x) / dt
        adx = np.abs(dx)
        feats[f"{prefix}d_mean_abs"] = float(np.mean(adx))
        feats[f"{prefix}d_max_abs"] = float(np.max(adx))
        feats[f"{prefix}d_std"] = float(np.std(dx, ddof=0))
        feats[f"{prefix}total_variation"] = float(np.sum(adx))
        # Normalize sharpness by amplitude (helps when amplitudes vary)
        feats[f"{prefix}sharpness_norm"] = float(np.max(adx) / (np.ptp(x) + eps))
    else:
        feats[f"{prefix}d_mean_abs"] = 0.0
        feats[f"{prefix}d_max_abs"] = 0.0
        feats[f"{prefix}d_std"] = 0.0
        feats[f"{prefix}total_variation"] = 0.0
        feats[f"{prefix}sharpness_norm"] = 0.0

    # Second derivative (optional but often useful)
    if len(x) >= 3 and dt > 0:
        dx = np.diff(x) / dt
        ddx = np.diff(dx) / dt
        feats[f"{prefix}dd_max_abs"] = float(np.max(np.abs(ddx)))
        feats[f"{prefix}dd_std"] = float(np.std(ddx, ddof=0))
    else:
        feats[f"{prefix}dd_max_abs"] = 0.0
        feats[f"{prefix}dd_std"] = 0.0

    return feats


def build_window_dataset(data: pd.DataFrame, window_size: int, step_size: int):
    """
    Convert each file into many window samples.
    Each sample = features computed over a time window.
    """
    rows = []
    y = []
    groups = []

    for file_id, df in tqdm(list(data.groupby("file_id")), desc="Building window dataset", unit="file"):
        surface = df["surface"].iloc[0]
        dt = infer_dt(df)

        arr = df[FEATURE_COLS].to_numpy(dtype=float)
        n = len(arr)

        if n < window_size:
            # Skip too-short files
            continue

        for start in range(0, n - window_size + 1, step_size):
            win = arr[start : start + window_size, :]  # shape (window_size, 4)

            feat = {}
            # Per-leg features
            for i, c in enumerate(FEATURE_COLS):
                feat.update(series_features(win[:, i], dt, prefix=f"{c}_"))

            # Total load features (often very informative)
            total = np.sum(win, axis=1)
            feat.update(series_features(total, dt, prefix="total_"))

            rows.append(feat)
            y.append(surface)
            groups.append(file_id)

    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(y, name="surface")
    groups = pd.Series(groups, name="file_id")

    if len(X) == 0:
        raise RuntimeError("No window samples were built. Check window_size vs file lengths.")

    return X, y, groups


def majority_vote_by_file(file_ids: pd.Series, y_true: pd.Series, y_pred: np.ndarray):
    df = pd.DataFrame({"file_id": file_ids.values, "y_true": y_true.values, "y_pred": y_pred})
    file_true = df.groupby("file_id")["y_true"].first()
    file_pred = df.groupby("file_id")["y_pred"].agg(lambda s: s.value_counts().idxmax())
    return file_true, file_pred


def main():
    p = argparse.ArgumentParser(description="Train SVM surface classifier (hard_ground vs grass).")
    p.add_argument("--window-size", type=int, default=50,
                   help="Window size in samples (default 50). If dt=0.002, 50 samples ~ 0.1s.")
    p.add_argument("--step-size", type=int, default=25,
                   help="Step size in samples (default 25).")
    p.add_argument("--skip-window", action="store_true",
                   help="Skip the window-based model (only run file-summary baseline).")
    args = p.parse_args()

    metrics_lines = []
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "window_size": args.window_size,
            "step_size": args.step_size,
            "feature_cols": FEATURE_COLS,
        },
        "model_paths": {
            "window_based": str(MODEL_WINDOW),
            "file_summary": str(MODEL_FILE_SUMMARY),
        },
    }

    data = load_data(DATA_ROOT)
    print(f"Total rows: {len(data)}")
    print(f"Total files: {data['file_id'].nunique()}\n")

    # --------- 1) WINDOW-BASED MODEL (recommended) ----------
    if not args.skip_window:
        Xw, yw, groups = build_window_dataset(data, window_size=args.window_size, step_size=args.step_size)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(Xw, yw, groups=groups))

        X_train, X_val = Xw.iloc[train_idx], Xw.iloc[val_idx]
        y_train, y_val = yw.iloc[train_idx], yw.iloc[val_idx]
        groups_val = groups.iloc[val_idx]

        print("=" * 60)
        print("1) WINDOW-BASED SVM (captures spike sharpness via derivatives)")
        print("=" * 60)
        print(f"Train windows: {len(X_train)} | Val windows: {len(X_val)}")
        print(f"Train files: {groups.iloc[train_idx].nunique()} | Val files: {groups_val.nunique()}")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
        ])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        acc_win = accuracy_score(y_val, y_pred)
        cm_win = confusion_matrix(y_val, y_pred, labels=CLASSES)

        print(f"\nWindow-level accuracy: {acc_win:.4f}")
        print("Confusion matrix (windows):")
        print(f"              pred_hard  pred_grass")
        print(f"true_hard     {cm_win[0,0]:>10}  {cm_win[0,1]:>10}")
        print(f"true_grass    {cm_win[1,0]:>10}  {cm_win[1,1]:>10}")
        print("\nClassification report (windows):")
        print(classification_report(y_val, y_pred, target_names=CLASSES))

        file_true, file_pred = majority_vote_by_file(groups_val, y_val, y_pred)
        acc_file = accuracy_score(file_true, file_pred)
        cm_file = confusion_matrix(file_true, file_pred, labels=CLASSES)

        print("\nFile-level accuracy (majority vote over windows):", f"{acc_file:.4f}")
        print("Confusion matrix (files):")
        print(f"              pred_hard  pred_grass")
        print(f"true_hard     {cm_file[0,0]:>10}  {cm_file[0,1]:>10}")
        print(f"true_grass    {cm_file[1,0]:>10}  {cm_file[1,1]:>10}")

        # Save model + feature list + window params (important for consistent prediction later)
        joblib.dump({
            "pipeline": model,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "feature_names": list(Xw.columns),
            "feature_cols": FEATURE_COLS,
        }, MODEL_WINDOW)
        print(f"\nWindow-based model saved to: {MODEL_WINDOW}")

        validation_results["window_based"] = {
            "window_level_accuracy": float(acc_win),
            "confusion_matrix_windows": cm_win.tolist(),
            "file_level_accuracy": float(acc_file),
            "confusion_matrix_files": cm_file.tolist(),
            "classification_report_windows": classification_report(y_val, y_pred, target_names=CLASSES),
        }

        metrics_lines.extend([
            "",
            "=" * 60,
            "1) WINDOW-BASED SVM",
            "=" * 60,
            f"Window-level accuracy: {acc_win:.4f}",
            "Confusion matrix (windows):",
            f"              pred_hard  pred_grass",
            f"true_hard     {cm_win[0,0]:>10}  {cm_win[0,1]:>10}",
            f"true_grass    {cm_win[1,0]:>10}  {cm_win[1,1]:>10}",
            "",
            "Classification report (windows):",
            classification_report(y_val, y_pred, target_names=CLASSES),
            "",
            f"File-level accuracy (majority vote over windows): {acc_file:.4f}",
            "Confusion matrix (files):",
            f"              pred_hard  pred_grass",
            f"true_hard     {cm_file[0,0]:>10}  {cm_file[0,1]:>10}",
            f"true_grass    {cm_file[1,0]:>10}  {cm_file[1,1]:>10}",
        ])

    # --------- 2) FILE-SUMMARY BASELINE (improved with derivatives) ----------
    print("\n" + "=" * 60)
    print("2) FILE-SUMMARY FEATURES (baseline, one sample per file)")
    print("=" * 60)

    feat_rows = []
    for file_id, df in tqdm(list(data.groupby("file_id")), desc="Building file-summary features", unit="file"):
        dt = infer_dt(df)
        row = {"file_id": file_id, "surface": df["surface"].iloc[0]}

        for c in FEATURE_COLS:
            x = df[c].to_numpy(dtype=float)
            row.update(series_features(x, dt, prefix=f"{c}_"))

        total = df[FEATURE_COLS].sum(axis=1).to_numpy(dtype=float)
        row.update(series_features(total, dt, prefix="total_"))

        feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows).fillna(0.0)
    Xf = feat_df.drop(columns=["file_id", "surface"])
    yf = feat_df["surface"]

    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        Xf, yf, test_size=0.2, random_state=42, stratify=yf
    )

    model_f = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
    ])
    model_f.fit(X_train_f, y_train_f)
    pred_f = model_f.predict(X_val_f)

    acc_f = accuracy_score(y_val_f, pred_f)
    cm_f = confusion_matrix(y_val_f, pred_f, labels=CLASSES)

    print(f"Accuracy: {acc_f:.4f}")
    print("Confusion matrix (files):")
    print(f"              pred_hard  pred_grass")
    print(f"true_hard     {cm_f[0,0]:>10}  {cm_f[0,1]:>10}")
    print(f"true_grass    {cm_f[1,0]:>10}  {cm_f[1,1]:>10}")
    print("\nClassification report (files):")
    print(classification_report(y_val_f, pred_f, target_names=CLASSES))

    joblib.dump({"pipeline": model_f, "feature_names": list(Xf.columns)}, MODEL_FILE_SUMMARY)
    print(f"\nFile-summary model saved to: {MODEL_FILE_SUMMARY}")

    validation_results["file_summary"] = {
        "accuracy": float(acc_f),
        "confusion_matrix": cm_f.tolist(),
        "classification_report": classification_report(y_val_f, pred_f, target_names=CLASSES),
    }

    metrics_lines.extend([
        "",
        "=" * 60,
        "2) FILE-SUMMARY FEATURES (baseline)",
        "=" * 60,
        f"Accuracy: {acc_f:.4f}",
        "Confusion matrix (files):",
        f"              pred_hard  pred_grass",
        f"true_hard     {cm_f[0,0]:>10}  {cm_f[0,1]:>10}",
        f"true_grass    {cm_f[1,0]:>10}  {cm_f[1,1]:>10}",
        "",
        "Classification report (files):",
        classification_report(y_val_f, pred_f, target_names=CLASSES),
    ])

    METRICS_FILE.write_text("\n".join(metrics_lines))
    with open(METRICS_JSON, "w") as f:
        json.dump(validation_results, f, indent=2)

    print(f"\nValidation metrics saved to: {METRICS_FILE}")
    print(f"Validation results saved to: {METRICS_JSON}")


if __name__ == "__main__":
    main()