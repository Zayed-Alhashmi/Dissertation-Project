import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from load_ct import load_dicom_series
from score_patient import total_agatston


# Run classical pipeline on all patients in a data folder and compare to ground truth.
def run_validation(data_root: str, scores_csv: str):
    gt_df = pd.read_csv(scores_csv)
    gt_df.columns = gt_df.columns.str.strip()  # remove accidental whitespace from headers
    gt_df["patient_id"] = gt_df["filename"].str.strip().str.replace("A", "", regex=False)

    patient_folders = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))],
        key=lambda x: int(x) if x.isdigit() else -1
    )

    results = []

    for pid in patient_folders:
        folder = os.path.join(data_root, pid)

        gt_row = gt_df[gt_df["patient_id"] == pid]
        if gt_row.empty:
            print(f"  Patient {pid}: no ground truth found, skipping.")
            continue

        gt_score = float(gt_row["total"].values[0])

        try:
            series = load_dicom_series(folder)
        except FileNotFoundError:
            print(f"  Patient {pid}: no DICOM files found, skipping.")
            continue

        pred_score = total_agatston(series, verbose=False)  # suppress per-slice logs

        results.append({
            "patient": pid,
            "predicted": round(pred_score, 1),
            "ground_truth": round(gt_score, 1),
            "error": round(pred_score - gt_score, 1),
        })

        status = "OVER" if pred_score > gt_score else ("UNDER" if pred_score < gt_score else "EXACT")
        print(f"  {pid:>4} | pred: {pred_score:7.1f} | truth: {gt_score:7.1f} | "
              f"err: {pred_score - gt_score:+8.1f}  [{status}]")

    print()

    if not results:
        print("No results to evaluate.")
        return

    predicted    = np.array([r["predicted"]    for r in results])
    ground_truth = np.array([r["ground_truth"] for r in results])
    errors       = predicted - ground_truth

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r, p = pearsonr(predicted, ground_truth) if len(results) >= 3 else (float("nan"), float("nan"))

    over  = sum(1 for e in errors if e > 0)
    under = sum(1 for e in errors if e < 0)
    exact = sum(1 for e in errors if e == 0)

    print("=" * 55)
    print(f"  Patients evaluated : {len(results)}")
    print(f"  MAE                : {mae:.1f}")
    print(f"  RMSE               : {rmse:.1f}")
    print(f"  Pearson r          : {r:.3f}  (p = {p:.4f})")
    print(f"  Over-scoring       : {over} patients")
    print(f"  Under-scoring      : {under} patients")
    print(f"  Exact (±0)         : {exact} patients")
    print("=" * 55)

    return results


if __name__ == "__main__":
    from cli import pick_folder
    from tkinter import filedialog
    import tkinter as tk

    print("Select the patient DATA ROOT folder (e.g. SampleData/)")
    data_root = pick_folder("Select data root folder (contains patient subfolders)")

    # look for scores.csv in the same directory automatically
    scores_csv = os.path.join(data_root, "scores.csv")
    if not os.path.isfile(scores_csv):
        print("Select the scores.csv ground truth file")
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        scores_csv = filedialog.askopenfilename(
            title="Select scores.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        if not scores_csv:
            print("No file selected. Exiting.")
            sys.exit(1)
    else:
        print(f"Found scores.csv in {data_root}")

    print(f"\nData root : {data_root}")
    print(f"Scores    : {scores_csv}")
    print("-" * 55)

    run_validation(data_root, scores_csv)
