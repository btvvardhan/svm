#!/usr/bin/env python3
"""
SVM surface classifier: hard_ground vs grass.
- Split train/val by FILE to avoid data leakage (no same-file in both).
- Two modes: row-based (with file-level eval) and file-summary features.
"""
from pathlib import Path
import argparse
import json
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# --------- Config ----------
DATA_ROOT = Path(__file__).resolve().parent  # repo root: grass/ and hard_ground/
OUT_DIR = DATA_ROOT  # where to save model and metrics (change to e.g. DATA_ROOT / "output" if you prefer)
CLASSES = ["hard_ground", "grass"]
FEATURE_COLS = ["fsr1", "fsr2", "fsr3", "fsr4"]
MODEL_ROW = OUT_DIR / "svm_row_based.joblib"
MODEL_FILE_SUMMARY = OUT_DIR / "svm_file_summary.joblib"
METRICS_FILE = OUT_DIR / "validation_metrics.txt"
METRICS_JSON = OUT_DIR / "validation_results.json"


def load_data(data_root: Path):
    """Load all CSVs; label from folder name; file_id from filename."""
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
        needed = set(FEATURE_COLS)
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        df["surface"] = cls
        df["file_id"] = csv_path.stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description="Train SVM surface classifier (hard_ground vs grass).")
    p.add_argument("--quick", action="store_true", help="Only run file-summary model (fast); skip slow row-based model.")
    args = p.parse_args()

    metrics_lines = []
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "model_paths": {"row_based": str(MODEL_ROW), "file_summary": str(MODEL_FILE_SUMMARY)},
    }

    data = load_data(DATA_ROOT)
    print(f"Total rows: {len(data)}, files: {data['file_id'].nunique()}\n")

    # --------- 1) Row-based SVM, split by file ----------
    if not args.quick:
        X = data[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        y = data["surface"]
        groups = data["file_id"]

        mask = X.notna().all(axis=1)
        X, y, groups = X[mask], y[mask], groups[mask]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_val = groups.iloc[val_idx]
        print(f"Row-based: train {len(X_train)} rows, val {len(X_val)} rows")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", verbose=True)),
        ])
        for _ in tqdm([1], desc="Fitting row-based SVM", unit="model"):
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        print("\n" + "=" * 60)
        print("1) ROW-BASED SVM (train/val split by file)")
        print("=" * 60)
        acc_row = accuracy_score(y_val, y_pred)
        cm_row = confusion_matrix(y_val, y_pred, labels=CLASSES)
        print(f"Row-level accuracy: {acc_row:.4f}")
        print("Confusion matrix (rows):")
        print(f"              pred_hard  pred_grass")
        print(f"true_hard     {cm_row[0,0]:>10}  {cm_row[0,1]:>10}")
        print(f"true_grass    {cm_row[1,0]:>10}  {cm_row[1,1]:>10}")
        print("\nClassification report (rows):")
        print(classification_report(y_val, y_pred, target_names=CLASSES))

        val_pred_df = pd.DataFrame({
            "file_id": groups_val.values,
            "y_true": y_val.values,
            "y_pred": y_pred,
        })
        file_true = val_pred_df.groupby("file_id")["y_true"].first()
        file_pred = val_pred_df.groupby("file_id")["y_pred"].agg(
            lambda s: s.value_counts().idxmax()
        )
        acc_file = accuracy_score(file_true, file_pred)
        cm_file = confusion_matrix(file_true, file_pred, labels=CLASSES)
        print("File-level (majority vote):")
        print(f"  Accuracy: {acc_file:.4f}")
        print("  Confusion matrix (files):")
        print(f"              pred_hard  pred_grass")
        print(f"true_hard     {cm_file[0,0]:>10}  {cm_file[0,1]:>10}")
        print(f"true_grass    {cm_file[1,0]:>10}  {cm_file[1,1]:>10}")

        joblib.dump(model, MODEL_ROW)
        print(f"\nRow-based model saved to: {MODEL_ROW}")
        validation_results["row_based"] = {
            "row_level_accuracy": float(acc_row),
            "confusion_matrix_rows": cm_row.tolist(),
            "file_level_accuracy": float(acc_file),
            "confusion_matrix_files": cm_file.tolist(),
            "classification_report": classification_report(y_val, y_pred, target_names=CLASSES),
        }
        metrics_lines.extend([
            "",
            "=" * 60,
            "1) ROW-BASED SVM (train/val split by file)",
            "=" * 60,
            f"Row-level accuracy: {acc_row:.4f}",
            "Confusion matrix (rows):",
            f"              pred_hard  pred_grass",
            f"true_hard     {cm_row[0,0]:>10}  {cm_row[0,1]:>10}",
            f"true_grass    {cm_row[1,0]:>10}  {cm_row[1,1]:>10}",
            "",
            "Classification report (rows):",
            classification_report(y_val, y_pred, target_names=CLASSES),
            "",
            f"File-level (majority vote) accuracy: {acc_file:.4f}",
            "Confusion matrix (files):",
            f"              pred_hard  pred_grass",
            f"true_hard     {cm_file[0,0]:>10}  {cm_file[0,1]:>10}",
            f"true_grass    {cm_file[1,0]:>10}  {cm_file[1,1]:>10}",
        ])

    # --------- 2) File-summary features ----------
    print("\n" + "=" * 60)
    print("2) FILE-SUMMARY FEATURES (one sample per file)")
    print("=" * 60)

    grouped = list(data.groupby("file_id"))
    feat_rows = []
    for file_id, df in tqdm(grouped, desc="Building file-summary features", unit="file"):
        row = {"file_id": file_id, "surface": df["surface"].iloc[0]}
        for c in FEATURE_COLS:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            row[f"{c}_mean"] = s.mean()
            row[f"{c}_std"] = s.std()
            row[f"{c}_min"] = s.min()
            row[f"{c}_max"] = s.max()
            row[f"{c}_median"] = s.median()
        feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows)
    Xf = feat_df.drop(columns=["file_id", "surface"])
    yf = feat_df["surface"]
    Xf = Xf.fillna(0)

    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        Xf, yf, test_size=0.2, random_state=42, stratify=yf
    )
    print(f"File-summary: train {len(X_train_f)} files, val {len(X_val_f)} files")

    model_f = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", verbose=True)),
    ])
    for _ in tqdm([1], desc="Fitting file-summary SVM", unit="model"):
        model_f.fit(X_train_f, y_train_f)
    pred_f = model_f.predict(X_val_f)

    print("\nFile-summary metrics:")
    acc_f = accuracy_score(y_val_f, pred_f)
    cm_f = confusion_matrix(y_val_f, pred_f, labels=CLASSES)
    print(f"Accuracy: {acc_f:.4f}")
    print("Confusion matrix (files):")
    print(f"              pred_hard  pred_grass")
    print(f"true_hard     {cm_f[0,0]:>10}  {cm_f[0,1]:>10}")
    print(f"true_grass    {cm_f[1,0]:>10}  {cm_f[1,1]:>10}")
    print("\nClassification report (files):")
    print(classification_report(y_val_f, pred_f, target_names=CLASSES))

    joblib.dump(model_f, MODEL_FILE_SUMMARY)
    print(f"\nFile-summary model saved to: {MODEL_FILE_SUMMARY}")

    validation_results["file_summary"] = {
        "accuracy": float(acc_f),
        "confusion_matrix": cm_f.tolist(),
        "classification_report": classification_report(y_val_f, pred_f, target_names=CLASSES),
    }

    metrics_lines.extend([
        "",
        "=" * 60,
        "2) FILE-SUMMARY FEATURES (one sample per file)",
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
    print(f"Validation metrics (text) saved to: {METRICS_FILE}")
    print(f"Validation results (JSON) saved to: {METRICS_JSON}")


if __name__ == "__main__":
    main()
