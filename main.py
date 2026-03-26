"""
main.py — top-level entry point for ProjectAnti.

Usage:
    python main.py --mode classical    # Score a single patient (GUI folder picker)
    python main.py --mode validate     # Batch validate against ground truth CSV
    python main.py --mode visualize    # Open the slice viewer
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="ProjectAnti",
        description="Automated CAC Scoring from Chest CT"
    )
    parser.add_argument(
        "--mode",
        choices=["classical", "validate", "visualize"],
        required=True,
        help=(
            "classical  — score a single patient folder (Agatston pipeline)\n"
            "validate   — batch validate against a ground truth CSV\n"
            "visualize  — open the interactive slice viewer"
        ),
    )
    args = parser.parse_args()

    if args.mode == "classical":
        from classical.cli import pick_folder
        from classical.load_ct import load_dicom_series
        from classical.score_patient import total_agatston

        folder = pick_folder("Select the patient DICOM folder")
        series = load_dicom_series(folder)
        print(f"Loaded {len(series)} slices.\n")
        total = total_agatston(series)
        print(f"\nTotal Agatston score: {total:.1f}")

    elif args.mode == "validate":
        from classical.validate import run_validation
        from classical.cli import pick_folder
        from tkinter import filedialog
        import tkinter as tk
        import os

        data_root = pick_folder("Select data root folder (contains patient subfolders)")
        scores_csv = os.path.join(data_root, "scores.csv")

        if not os.path.isfile(scores_csv):
            print("scores.csv not found automatically. Select it manually.")
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
                return

        print(f"\nData root : {data_root}")
        print(f"Scores    : {scores_csv}")
        print("-" * 55)
        run_validation(data_root, scores_csv)

    elif args.mode == "visualize":
        from classical.cli import pick_folder
        from classical.load_ct import load_dicom_series
        from classical.visualize import SliceViewer

        folder = pick_folder("Select the patient DICOM folder")
        series = load_dicom_series(folder)
        print(f"Loaded {len(series)} slices.")
        print("Use \u2190 and \u2192 arrow keys to scroll.")
        SliceViewer(series)


if __name__ == "__main__":
    main()
