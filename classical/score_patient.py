import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical.load_ct import load_dicom_series
from classical.cli import pick_folder
from classical.utils import (
    calcium_mask,
    lung_guided_roi_mask,
    is_heart_level_slice,
    connected_lesions,
    filter_lesions_by_area,
    filter_elongated_lesions,
    remove_bone_like_components,
    filter_spine_region_blobs,
    filter_lateral_chest_wall_blobs,
    filter_aortic_blobs,
    detect_aorta_circle,
)
from classical.scoring import agatston_slice_score

# Global tracker caching the loaded CNN neural network to avoid reloading it on every slice.
_cnn_classifier = None
_cnn_arch       = None   # tracks which arch the singleton was built for


# Retrieves the global CNN model from memory or loads it from disk if it's the first time running.
def _get_cnn_classifier(arch: str = "resnet18"):
    global _cnn_classifier, _cnn_arch
    if _cnn_classifier is None or _cnn_classifier.arch != arch:
        from cnn.classifier import CACClassifier
        _cnn_classifier = CACClassifier(arch=arch)
        _cnn_arch = arch  # kept in sync for reference
    return _cnn_classifier


# Executes the entire sequential detection algorithm on one slice, chaining all filters and calculating its score.
def process_slice(hu, spacing, mode: str = "classical",
                  slice_idx: int = None, aorta_cache: dict = None,
                  arch: str = "resnet18"):
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

    # Anatomical position filters - steps 8 and 9, applied in all modes
    # These run on regionprops blobs, so we need to label the mask first
    from skimage.measure import regionprops as _regionprops
    _labels, _ = connected_lesions(mask.astype(bool))
    _blobs     = _regionprops(_labels)
    _blobs     = filter_spine_region_blobs(_blobs, hu)                           # reject spine-zone centroids
    _blobs     = filter_lateral_chest_wall_blobs(_blobs, hu)                     # reject lateral chest wall
    _blobs     = filter_aortic_blobs(_blobs, hu, spacing,
                                     slice_idx=slice_idx, aorta_cache=aorta_cache)  # step 10: geometric aorta exclusion
    mask       = _blobs_to_mask(_labels, _blobs, mask.shape)                    # rebuild mask from survivors

    # CNN FILTER - step 11, hybrid mode only
    # Lazy import keeps torch out of the classical code path entirely.
    # Model is loaded once via singleton - not per-slice - to keep inference fast.
    if mode == "hybrid":
        from skimage.measure import regionprops
        labels, _ = connected_lesions(mask.astype(bool))
        blobs     = regionprops(labels)                               # list of regionprops objects
        blobs     = _get_cnn_classifier(arch).filter_blobs(hu, blobs, spacing)  # CNN keeps true CAC only
        mask      = _blobs_to_mask(labels, blobs, mask.shape)        # rebuild boolean mask from survivors

    score = agatston_slice_score(hu, mask, spacing, min_area_mm2=1.0)

    return mask, score


# Merges isolated structure blobs mathematically back into a complete multi-lesion pixel array.
def _blobs_to_mask(labels, blobs, shape):
    import numpy as np
    out = np.zeros(shape, dtype=bool)
    for blob in blobs:
        out |= (labels == blob.label)
    return out


# Computes the final total cardiovascular risk by scanning all patient slices, adding up all valid Agatston scores.
def total_agatston(series, verbose: bool = True, mode: str = "classical",
                   arch: str = "resnet18"):
    total   = 0.0
    skipped = 0
    n       = len(series)
    start   = int(0.20 * n)  # skip top 20 % (above heart)
    end     = int(0.90 * n)  # skip bottom 10 % (below heart)

    # first pass - build aorta cache for all heart-level slices
    aorta_cache = {}
    for i in range(start, end):
        _, hu, spacing = series[i]
        if not is_heart_level_slice(hu):
            continue
        result = detect_aorta_circle(hu, spacing)
        if result is not None:
            aorta_cache[i] = result  # store (cy, cx, radius_mm) keyed by slice index

    # second pass - full pipeline, passing cache for fallback on missed slices
    for i in range(start, end):
        path, hu, spacing = series[i]
        mask, score = process_slice(hu, spacing, mode=mode,
                                    slice_idx=i, aorta_cache=aorta_cache,
                                    arch=arch)

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
    import argparse
    parser = argparse.ArgumentParser(description="CAC scoring pipeline")
    parser.add_argument("--mode", choices=["classical", "hybrid"], default="classical",
                        help="classical = HU thresholding only; hybrid = + CNN filter (default: classical)")
    parser.add_argument("--arch", choices=["resnet18", "efficientnet", "custom"],
                        default="resnet18",
                        help="CNN architecture for hybrid mode (default: resnet18)")
    args = parser.parse_args()

    folder = pick_folder("Select the patient DICOM folder")

    print(f"Running score_patient.py  [mode={args.mode}]")
    print(f"Folder: {folder}")

    series = load_dicom_series(folder)
    print(f"Loaded {len(series)} slices.\n")

    total = total_agatston(series, mode=args.mode, arch=args.arch)
    print(f"\nTotal Agatston score: {total:.1f}")