# Terrain classifier (hard_ground vs grass)

SVM-based classifier for robot terrain recognition using FSR (force-sensitive resistor) readings from four sensors (`fsr1`, `fsr2`, `fsr3`, `fsr4`). Trained on windowed time-series features including derivatives to capture spike shape; supports live streaming and batch prediction from CSV.

---

## Environment setup

### Create conda environment (recommended)

```bash
# Create env named "svm" with Python 3
conda create -n svm python=3.10 -y

# Activate it
conda activate svm

# Install dependencies from project root
cd /path/to/svm
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- `pandas`, `scikit-learn`, `tqdm` (see `requirements.txt`); `joblib` is included with scikit-learn.

---

## Data layout

- **Training data**: Two folders next to the scripts:
  - `grass/` — CSV files recorded on grass (e.g. `grass_1.csv`, `grass_2.csv`, …)
  - `hard_ground/` — CSV files recorded on hard ground (e.g. `hard_ground_1.csv`, …)

- **CSV format**: Each file must have columns **`fsr1`, `fsr2`, `fsr3`, `fsr4`** (numeric). An optional `time` column is used to infer sampling interval `dt`; if missing, a default is used.

Example:

```text
time,fsr1,fsr2,fsr3,fsr4,label
0.0,0,0,0,3968,grass_1
0.002,...,...,...,...,...
```

---

## Training

1. Activate the environment and go to the project directory:

   ```bash
   conda activate svm
   cd /path/to/svm
   ```

2. Run training (reads `grass/*.csv` and `hard_ground/*.csv`):

   ```bash
   python train_svm.py
   ```

3. Optional arguments:

   ```bash
   python train_svm.py --window-size 50 --step-size 25   # default
   python train_svm.py --skip-window                      # only train file-summary baseline
   ```

4. Outputs:
   - **`svm_window_based.joblib`** — main model for live and CSV prediction (use this).
   - **`svm_file_summary.joblib`** — baseline (one sample per file).
   - **`validation_metrics.txt`** and **`validation_results.json`** — validation accuracy and confusion matrices.

Train/validation split is **by file** (no leakage between files). The window-based model uses sliding windows (default 50 samples, step 25) and derivative-based features.

---

## Predicting terrain from a new CSV

To classify a **single CSV file** (e.g. a new recording) and get a terrain label:

```bash
conda activate svm
cd /path/to/svm

python predict_csv.py path/to/your_file.csv
```

Example:

```bash
python predict_csv.py grass/grass_5.csv
# Terrain: grass

python predict_csv.py /data/new_walk.csv
# Terrain: hard_ground
```

The script uses the window-based model, runs sliding windows over the CSV, and prints the **majority vote** over windows as the final terrain. Optional: `--model` to point to another joblib file, `--dt` if you need to set sampling interval manually.

---

## Live classification (on the robot)

For real-time classification from a stream of FSR samples:

1. Use the **window-based** model: `svm_window_based.joblib`.

2. In your robot code:

   ```python
   from predict_live import LiveTerrainClassifier

   clf = LiveTerrainClassifier("svm_window_based.joblib", dt=0.002)  # dt = your sampling period in seconds

   # Each time you read a new FSR sample:
   clf.push(fsr1, fsr2, fsr3, fsr4)
   label = clf.predict_if_ready()   # None until a full window is ready, then "hard_ground" or "grass"
   if label is not None:
       # use label for control/logging
   ```

3. Test the live pipeline from a CSV (simulates streaming):

   ```bash
   python predict_live.py --csv grass/grass_1.csv
   python predict_live.py --csv hard_ground/hard_ground_1.csv
   ```

Ensure `dt` matches your robot’s FSR sampling period (e.g. 0.002 for 500 Hz).

---

## Project files

| File / folder           | Purpose |
|-------------------------|--------|
| `train_svm.py`          | Train window-based and file-summary SVM models. |
| `predict_csv.py`        | Classify one CSV file → single terrain label (majority over windows). |
| `predict_live.py`       | Live classifier class + CSV streaming test; use on robot for real-time prediction. |
| `grass/`, `hard_ground/`| Training CSVs (required columns: `fsr1`–`fsr4`). |
| `svm_window_based.joblib` | Trained model for live and CSV prediction. |
| `svm_file_summary.joblib` | Baseline per-file model. |
| `validation_metrics.txt`, `validation_results.json` | Last run’s validation metrics. |
| `requirements.txt`      | Python dependencies. |

---

## Important notes

- **Classes**: Only **hard_ground** and **grass**. Adding new terrains requires new data and retraining.
- **Validation**: File-level accuracy can look very high with few files; window-level accuracy (~83% in default setup) is a better indicator. Collect more files for reliable validation.
- **CSV for prediction**: Any new CSV must have the same columns `fsr1`, `fsr2`, `fsr3`, `fsr4`. Extra columns are ignored. Use `predict_csv.py` for one-off file classification and `predict_live.py` for streaming or integration in robot code.
