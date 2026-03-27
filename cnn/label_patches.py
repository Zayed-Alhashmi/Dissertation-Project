import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import pandas as pd


# Loads the ground truth CSV and returns a dict mapping patient_id to total Agatston score.
def _load_ground_truth(scores_csv_path: str) -> dict:
    df = pd.read_csv(scores_csv_path)
    df.columns = df.columns.str.strip()                    # removes trailing spaces from column names
    df["patient_id"] = df["filename"].str.strip().str.rstrip("A")  # "109A" → "109"
    return dict(zip(df["patient_id"], df["total"].astype(float)))


# Decides whether a single patch is true coronary calcium or a false positive using area and peak HU rules.
def _heuristic_label(patch: dict) -> int:
    area   = patch["area_mm2"]
    peak   = patch["peak_hu"]

    if area > 15 and peak > 400:  # high HU + large area is characteristic of the aortic wall or ribs
        return 0
    if area > 20:                 # anything this large is very unlikely to be a coronary lesion
        return 0
    return 1


# Assigns a 0/1 label to every patch in the list using a heuristic and returns the updated list.
def label_patches(patch_list: list[dict], scores_csv_path: str, strategy: str = "heuristic") -> list[dict]:
    gt_scores = _load_ground_truth(scores_csv_path)  # loaded but available for future score-aware strategies

    for patch in patch_list:
        if strategy == "heuristic":
            patch["label"] = _heuristic_label(patch)
        else:
            raise ValueError(f"Unknown labelling strategy: {strategy}")

    n_pos = sum(1 for p in patch_list if p["label"] == 1)
    n_neg = len(patch_list) - n_pos
    print(f"  Label distribution: {n_pos} CAC (1)  |  {n_neg} false positive (0)  |  total {len(patch_list)}")

    return patch_list


# Saves the labelled patch list to a compressed .npz file ready for CNN training.
def save_labelled_dataset(patch_list: list[dict], output_path: str) -> None:
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
    if len(sys.argv) < 3:
        print("Usage: python -m cnn.label_patches <patches.npz> <scores.csv> [output_path]")
        sys.exit(1)

    patches_npz = sys.argv[1]
    scores_csv  = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "results/labelled_patches"

    data = np.load(patches_npz, allow_pickle=True)

    patch_list = []
    for i in range(len(data["patches"])):
        patch_list.append({
            "patch":      data["patches"][i],
            "patient_id": str(data["patient_ids"][i]),
            "peak_hu":    float(data["peak_hus"][i]),
            "area_mm2":   float(data["area_mm2s"][i]),
        })

    patch_list = label_patches(patch_list, scores_csv)
    save_labelled_dataset(patch_list, output_path)
