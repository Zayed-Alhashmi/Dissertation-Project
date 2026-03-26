import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from load_ct import load_dicom_series
from cli import pick_folder
from utils import (
    calcium_mask,
    lung_guided_roi_mask,
    is_heart_level_slice,
    connected_lesions,
    filter_lesions_by_area,
    filter_elongated_lesions,
    remove_bone_like_components,
)
from scoring import agatston_slice_score


# Run the full CAC pipeline on a single HU slice and return (mask, score)
def process_slice(hu, spacing):
    if not is_heart_level_slice(hu):  # skip if no bilateral lungs detected
        return None, None

    mask = calcium_mask(hu, threshold=130.0)  # pixels >= 130 HU

    roi  = lung_guided_roi_mask(hu)  # mediastinum band, falls back to fixed ellipse
    mask = mask & roi  # restrict candidates to heart ROI

    labels, _ = connected_lesions(mask)
    mask, _   = filter_lesions_by_area(labels, spacing,
                                       min_area_mm2=1.0, max_area_mm2=35.0)  # 1-35 mm²

    labels, _ = connected_lesions(mask.astype(bool))
    mask      = filter_elongated_lesions(labels, spacing,
                                         max_eccentricity=0.97,
                                         min_solidity=0.30)  # drop rib like arcs

    mask  = remove_bone_like_components(hu, mask.astype(bool), spacing,
                                        peak_hu_thr=550.0, area_thr_mm2=15.0)  # drop cortical bone

    score = agatston_slice_score(hu, mask, spacing, min_area_mm2=1.0)

    return mask, score


# Sum Agatston scores across slices in the middle 70 % of the series
def total_agatston(series, verbose: bool = True):
    total   = 0.0
    skipped = 0
    n       = len(series)
    start   = int(0.20 * n)  # skip top 20 % (above heart)
    end     = int(0.90 * n)  # skip bottom 10 % (below heart)

    for i in range(start, end):
        path, hu, spacing = series[i]
        mask, score = process_slice(hu, spacing)

        if mask is None:  # not at heart level, skip
            skipped += 1
            continue

        total += score
        if verbose and score > 0:  # log every slice with any detection (just for debug)
            print(f"  {os.path.basename(path)} -> slice score: {score:.1f}"
                  f"  (pixels: {int(mask.sum())})")

    if verbose:
        print(f"  [auto-skipped {skipped} slice(s) below heart level threshold]")
    return total


if __name__ == "__main__":
    folder = pick_folder("Select the patient DICOM folder")

    print("Running score_patient.py")
    print(f"Folder: {folder}")

    series = load_dicom_series(folder)
    print(f"Loaded {len(series)} slices.\n")

    total = total_agatston(series)
    print(f"\nTotal Agatston score: {total:.1f}")