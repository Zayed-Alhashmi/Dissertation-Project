import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
from skimage.measure import regionprops
from skimage.measure import label as sk_label

from classical.load_ct import load_dicom_series
from classical.utils import (
    is_heart_level_slice,
    calcium_mask,
    lung_guided_roi_mask,
    connected_lesions,
    filter_lesions_by_area,
)


# Cuts a square patch out of the HU image centred on the given row/col, zero-padding if the blob is near the edge.
def _extract_single_patch(hu: np.ndarray, centroid, patch_size: int) -> np.ndarray:
    h, w = hu.shape
    half = patch_size // 2
    r, c = int(round(centroid[0])), int(round(centroid[1]))

    r0, r1 = r - half, r - half + patch_size
    c0, c1 = c - half, c - half + patch_size

    r0c, r1c = max(r0, 0), min(r1, h)  # clamp to valid image bounds
    c0c, c1c = max(c0, 0), min(c1, w)

    patch = np.zeros((patch_size, patch_size), dtype=np.float32)

    dr0 = r0c - r0  # offset inside the patch where the real data starts
    dc0 = c0c - c0
    dr1 = dr0 + (r1c - r0c)
    dc1 = dc0 + (c1c - c0c)

    patch[dr0:dr1, dc0:dc1] = hu[r0c:r1c, c0c:c1c]
    return patch


# Clips HU values to the calcium-relevant range and scales them to [0, 1].
def _normalise(patch: np.ndarray) -> np.ndarray:
    return np.clip(patch, 0, 1000) / 1000.0


# Runs the classical pipeline on a patient folder up to the blob detection step and returns one patch per surviving calcium candidate.
def extract_patches(patient_folder: str, patch_size: int = 64) -> list[dict]:
    patient_id = os.path.basename(os.path.normpath(patient_folder))
    series = load_dicom_series(patient_folder)
    patches = []

    for slice_idx, (_, hu, spacing) in enumerate(series):

        if not is_heart_level_slice(hu):
            continue

        mask = calcium_mask(hu, threshold=130.0)
        roi = lung_guided_roi_mask(hu)
        mask = mask & roi

        labels, _ = connected_lesions(mask)

        mask_filtered, _ = filter_lesions_by_area(
            labels, spacing, min_area_mm2=1.0, max_area_mm2=35.0
        )

        labels_clean = sk_label(mask_filtered.astype(bool), connectivity=2)

        props = regionprops(labels_clean)
        if not props:
            continue

        ry, rx = spacing
        pixel_area_mm2 = ry * rx

        for prop in props:
            centroid = prop.centroid
            bbox = prop.bbox
            area_mm2 = float(prop.area * pixel_area_mm2)

            blob_mask = (labels_clean == prop.label)
            peak_hu = float(np.max(hu[blob_mask]))

            raw_patch = _extract_single_patch(hu, centroid, patch_size)
            norm_patch = _normalise(raw_patch)

            patches.append({
                "patch":      norm_patch,
                "patient_id": patient_id,
                "slice_idx":  slice_idx,
                "centroid":   centroid,
                "bbox":       bbox,
                "peak_hu":    peak_hu,
                "area_mm2":   area_mm2,
            })

    return patches


# Saves the extracted patch list to a compressed .npz file so it can be loaded for training later.
def save_patches(patch_list: list[dict], output_path: str) -> None:
    if not patch_list:
        print("  [save_patches] Nothing to save, list is empty.")
        return

    np.savez_compressed(
        output_path,
        patches     = np.stack([p["patch"]      for p in patch_list]).astype(np.float32),
        patient_ids = np.array([p["patient_id"] for p in patch_list], dtype=object),
        slice_idxs  = np.array([p["slice_idx"]  for p in patch_list], dtype=np.int32),
        centroids   = np.array([p["centroid"]   for p in patch_list], dtype=np.float32),
        bboxes      = np.array([p["bbox"]        for p in patch_list], dtype=np.int32),
        peak_hus    = np.array([p["peak_hu"]    for p in patch_list], dtype=np.float32),
        area_mm2s   = np.array([p["area_mm2"]   for p in patch_list], dtype=np.float32),
    )
    print(f"  Saved {len(patch_list)} patches to {output_path}.npz")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cnn.patch_extractor <patient_dicom_folder> [patch_size]")
        sys.exit(1)

    folder = sys.argv[1]
    patch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    print(f"Patient folder : {folder}")
    print(f"Patch size     : {patch_size}x{patch_size}")

    patch_list = extract_patches(folder, patch_size=patch_size)
    print(f"\nExtracted {len(patch_list)} patches.")

    if patch_list:
        p = patch_list[0]
        print(f"  patient_id : {p['patient_id']}")
        print(f"  slice_idx  : {p['slice_idx']}")
        print(f"  centroid   : ({p['centroid'][0]:.1f}, {p['centroid'][1]:.1f})")
        print(f"  peak_hu    : {p['peak_hu']:.1f}")
        print(f"  area_mm2   : {p['area_mm2']:.2f}")
        print(f"  patch mean : {p['patch'].mean():.4f}")
        print(f"  patch shape: {p['patch'].shape}")
