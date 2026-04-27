import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from classical.load_ct import load_dicom_series
from classical.score_patient import total_agatston


# Run classical or hybrid pipeline on all patients and compare to ground truth.
def run_validation(data_root: str, scores_csv: str, mode: str = "classical",
                   arch: str = "resnet18"):
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

        pred_score = total_agatston(series, verbose=False, mode=mode,
                                    arch=arch)  # suppress per-slice logs

        results.append({
            "patient":      pid,
            "predicted":    round(pred_score, 1),
            "ground_truth": round(gt_score, 1),
            "error":        round(pred_score - gt_score, 1),
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


# Run BOTH classical and hybrid pipelines and print a side-by-side comparison per patient.
# arch controls which CNN checkpoint is used for the hybrid column.
def run_compare(data_root: str, scores_csv: str, arch: str = "resnet18"):
    gt_df = pd.read_csv(scores_csv)
    gt_df.columns = gt_df.columns.str.strip()
    gt_df["patient_id"] = gt_df["filename"].str.strip().str.replace("A", "", regex=False)

    patient_folders = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))],
        key=lambda x: int(x) if x.isdigit() else -1
    )

    c_preds, h_preds, truths = [], [], []

    for pid in patient_folders:
        folder = os.path.join(data_root, pid)

        gt_row = gt_df[gt_df["patient_id"] == pid]
        if gt_row.empty:
            print(f"  Patient {pid}: no ground truth, skipping.")
            continue

        gt_score = float(gt_row["total"].values[0])

        try:
            series = load_dicom_series(folder)
        except FileNotFoundError:
            print(f"  Patient {pid}: no DICOM files found, skipping.")
            continue

        c_score = total_agatston(series, verbose=False, mode="classical", arch=arch)
        h_score = total_agatston(series, verbose=False, mode="hybrid",    arch=arch)

        c_err = c_score - gt_score
        h_err = h_score - gt_score

        print(f"  {pid:>4} | classical: {c_score:7.1f} | hybrid: {h_score:7.1f} | "
              f"truth: {gt_score:7.1f} | classical err: {c_err:+8.1f} | hybrid err: {h_err:+8.1f}")

        c_preds.append(c_score)
        h_preds.append(h_score)
        truths.append(gt_score)

    print()

    if not truths:
        print("No results to evaluate.")
        return

    c_preds = np.array(c_preds)
    h_preds = np.array(h_preds)
    truths  = np.array(truths)

    c_errs = c_preds - truths
    h_errs = h_preds - truths

    c_mae  = float(np.mean(np.abs(c_errs)))
    c_rmse = float(np.sqrt(np.mean(c_errs ** 2)))
    c_r, _ = pearsonr(c_preds, truths) if len(truths) >= 3 else (float("nan"), float("nan"))

    h_mae  = float(np.mean(np.abs(h_errs)))
    h_rmse = float(np.sqrt(np.mean(h_errs ** 2)))
    h_r, _ = pearsonr(h_preds, truths) if len(truths) >= 3 else (float("nan"), float("nan"))

    mae_reduction     = c_mae - h_mae
    mae_reduction_pct = (mae_reduction / c_mae * 100) if c_mae > 0 else 0.0

    print("=" * 55)
    print("CLASSICAL:")
    print(f"  MAE: {c_mae:.1f} | RMSE: {c_rmse:.1f} | Pearson r: {c_r:.3f}")
    print()
    print(f"HYBRID (CNN={arch}):")
    print(f"  MAE: {h_mae:.1f} | RMSE: {h_rmse:.1f} | Pearson r: {h_r:.3f}")
    print()
    print("IMPROVEMENT:")
    print(f"  MAE reduction: {mae_reduction:.1f} points ({mae_reduction_pct:.0f}%)")
    print("=" * 55)


if __name__ == "__main__":
    import argparse
    from classical.cli import pick_folder
    from tkinter import filedialog
    import tkinter as tk

    parser = argparse.ArgumentParser(description="Validate CAC scoring pipeline")
    parser.add_argument("--mode", choices=["classical", "hybrid", "compare"],
                        default="classical",
                        help=("classical = HU pipeline only  "
                              "hybrid = + CNN filter  "
                              "compare = both side-by-side (default: classical)"))
    parser.add_argument("--arch", choices=["resnet18", "efficientnet", "custom"],
                        default="resnet18",
                        help="CNN architecture for hybrid/compare mode (default: resnet18)")
    args, _ = parser.parse_known_args()

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
    print(f"Mode      : {args.mode}")
    if args.mode in ("hybrid", "compare"):
        print(f"CNN classifier: cnn/checkpoints/best_model_{args.arch}.pt")
    print("-" * 55)

    if args.mode == "compare":
        run_compare(data_root, scores_csv, arch=args.arch)
    else:
        run_validation(data_root, scores_csv, mode=args.mode, arch=args.arch)
