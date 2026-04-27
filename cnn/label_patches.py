import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import pandas as pd


# Loads scores.csv and returns a dict mapping patient_id -> total Agatston ground truth
def _load_ground_truth(scores_csv_path: str) -> dict:
    df = pd.read_csv(scores_csv_path)
    df.columns = df.columns.str.strip()
    df["patient_id"] = df["filename"].str.strip().str.rstrip("A")  # "109A" → "109"
    return dict(zip(df["patient_id"], df["total"].astype(float)))


# Run the classical pipeline on one patient folder and return total Agatston score.
# The score is computed silently (verbose=False) so the output stays clean.
def _classical_score(patient_folder: str) -> float:
    from classical.load_ct import load_dicom_series
    from classical.score_patient import total_agatston
    try:
        series = load_dicom_series(patient_folder)
        return total_agatston(series, verbose=False, mode="classical")
    except Exception as e:
        print(f"    [WARN] Could not score {patient_folder}: {e}")
        return 0.0


# Old heuristic as fallback for Rule 5
def _heuristic_label(area: float, peak: float) -> int:
    if area > 15 and peak > 400:  # large + bright → likely aortic wall or rib
        return 0
    return 1


# Assign labels to a list of patches using a 5-rule ground-truth-aware strategy.
#
# Rules (first matching wins per patient):
#   1. Zero-calcium patients (gt == 0 or < 2)  → ALL blobs labelled 0 (FP examples)
#   2. Massively over-scored (pred > 2.5×gt AND gt > 10)
#      → large/bright blobs (area>10 OR peak>450) → 0, smaller blobs → 1
#   3. Well-scored (0.5 < pred/gt < 2.0)  → ALL blobs labelled 1 (confirmed CAC)
#   4. Under-scored (pred < 0.5×gt AND gt > 50)  → ALL blobs labelled 1 (real CAC)
#   5. Fallback heuristic: area > 15 AND peak > 400 → 0, else 1
def label_patches(patch_list: list, scores_csv_path: str, data_root: str) -> list:
    gt_scores = _load_ground_truth(scores_csv_path)

    # Group patches by patient
    by_patient: dict[str, list] = {}
    for patch in patch_list:
        pid = str(patch["patient_id"])
        by_patient.setdefault(pid, []).append(patch)

    print(f"\nRunning classical pipeline per patient to get predictions ...\n")

    labelled = []

    for pid, patches in sorted(by_patient.items(), key=lambda x: x[0]):
        gt = gt_scores.get(pid, None)
        if gt is None:
            print(f"  {pid:>4} | no ground truth in CSV - using fallback heuristic")
            for p in patches:
                p["label"] = _heuristic_label(p["area_mm2"], p["peak_hu"])
                labelled.append(p)
            continue

        folder = os.path.join(data_root, pid)
        pred   = _classical_score(folder)
        ratio  = (pred / gt) if gt > 0 else float("inf")

        # Rule 1 - zero calcium patient: all blobs are FPs
        if gt < 2:
            rule = "1 (zero-calcium → all FP)"
            for p in patches:
                p["label"] = 0
            n_pos, n_neg = 0, len(patches)

        # Rule 2 - massively over-scored: split on size/HU
        elif pred > 2.5 * gt and gt > 10:
            rule = "2 (over-scored → large blobs = FP)"
            n_pos = n_neg = 0
            for p in patches:
                if p["area_mm2"] > 10 or p["peak_hu"] > 450:
                    p["label"] = 0
                    n_neg += 1
                else:
                    p["label"] = 1
                    n_pos += 1

        # Rule 3 - well-scored: all blobs are likely true CAC
        elif gt > 0 and 0.5 < ratio < 2.0:
            rule = "3 (well-scored → all CAC)"
            for p in patches:
                p["label"] = 1
            n_pos, n_neg = len(patches), 0

        # Rule 4 - under-scored with high gt: all blobs are real CAC
        elif pred < 0.5 * gt and gt > 50:
            rule = "4 (under-scored → all CAC)"
            for p in patches:
                p["label"] = 1
            n_pos, n_neg = len(patches), 0

        # Rule 5 - fallback heuristic
        else:
            rule = "5 (fallback heuristic)"
            n_pos = n_neg = 0
            for p in patches:
                p["label"] = _heuristic_label(p["area_mm2"], p["peak_hu"])
                if p["label"] == 1:
                    n_pos += 1
                else:
                    n_neg += 1

        print(f"  {pid:>4} | gt={gt:7.1f} | pred={pred:7.1f} | ratio={ratio:5.2f} | "
              f"rule {rule} | CAC={n_pos} FP={n_neg}")
        labelled.extend(patches)

    n_pos_total = sum(1 for p in labelled if p["label"] == 1)
    n_neg_total = len(labelled) - n_pos_total
    print(f"\nFinal label distribution:")
    print(f"  CAC (1) : {n_pos_total}")
    print(f"  FP  (0) : {n_neg_total}")
    print(f"  Total   : {len(labelled)}")
    return labelled


# Save labelled patches to a compressed .npz file ready for CNN training
def save_labelled_dataset(patch_list: list, output_path: str) -> None:
    if not patch_list:
        print("  [save_labelled_dataset] Nothing to save, list is empty.")
        return

    patches     = np.stack([p["patch"]      for p in patch_list]).astype(np.float32)
    labels      = np.array([p["label"]      for p in patch_list], dtype=np.int32)
    patient_ids = np.array([p["patient_id"] for p in patch_list], dtype=object)
    peak_hus    = np.array([p["peak_hu"]    for p in patch_list], dtype=np.float32)
    area_mm2s   = np.array([p["area_mm2"]   for p in patch_list], dtype=np.float32)

    np.savez_compressed(
        output_path,
        patches     = patches,
        labels      = labels,
        patient_ids = patient_ids,
        peak_hus    = peak_hus,
        area_mm2s   = area_mm2s,
    )
    print(f"  Saved {len(patch_list)} labelled patches to {output_path}.npz")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python cnn/label_patches.py <patches_raw.npz> <scores.csv> <data_root> <output_path>")
        print("Example: python cnn/label_patches.py data/patches_raw.npz data/scores.csv data/raw data/patches_labelled_smart")
        sys.exit(1)

    patches_npz = sys.argv[1]
    scores_csv  = sys.argv[2]
    data_root   = sys.argv[3]
    output_path = sys.argv[4]

    print(f"Patches    : {patches_npz}")
    print(f"Scores CSV : {scores_csv}")
    print(f"Data root  : {data_root}")
    print(f"Output     : {output_path}.npz")

    data = np.load(patches_npz, allow_pickle=True)

    patch_list = []
    for i in range(len(data["patches"])):
        patch_list.append({
            "patch":      data["patches"][i],
            "patient_id": str(data["patient_ids"][i]),
            "peak_hu":    float(data["peak_hus"][i]),
            "area_mm2":   float(data["area_mm2s"][i]),
        })

    print(f"\nLoaded {len(patch_list)} raw patches from {len(set(p['patient_id'] for p in patch_list))} patients.\n")

    patch_list = label_patches(patch_list, scores_csv, data_root)
    save_labelled_dataset(patch_list, output_path)
