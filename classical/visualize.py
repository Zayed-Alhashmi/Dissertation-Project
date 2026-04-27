import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from classical.load_ct import load_dicom_series
from classical.cli import pick_folder
from classical.utils import (
    apply_window, calcium_mask, lung_guided_roi_mask, is_heart_level_slice,
    connected_lesions, filter_lesions_by_area, filter_elongated_lesions,
    remove_bone_like_components,
)
from classical.scoring import agatston_slice_score
from classical.score_patient import process_slice, _get_cnn_classifier

# Scroll through CT slices with arrow keys.
# mode="classical" - red overlay for calcium (unchanged behaviour).
# mode="hybrid"    - red = CNN-kept true CAC, blue = CNN-rejected false positives.
class SliceViewer:
    def __init__(self, series, mode: str = "classical"):
        self.series = series
        self.mode   = mode
        self.idx    = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

        # Open the window in full screen - try Windows then Mac API
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state("zoomed")
        except Exception:
            try:
                mng.window.showMaximized()
            except Exception:
                pass

        img, overlay_data, title_str = self._compute(self.idx)

        self.base = self.ax.imshow(img, cmap="gray")
        # RGBA overlay works for both modes: classical uses a single-channel Red map,
        # hybrid uses a full RGBA array with transparent background.
        if self.mode == "hybrid":
            self.overlay = self.ax.imshow(overlay_data)  # overlay_data is RGBA (H,W,4)
        else:
            self.overlay = self.ax.imshow(overlay_data, cmap="Reds", alpha=0.85, vmin=0, vmax=1)
        self.ax.set_title(title_str, color="white", pad=12, fontsize=14)
        self.ax.axis("off")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    # Helpers

    # Boolean mask → float array: True=1.0, False=NaN (used for classical cmap="Reds" overlay)
    def _mask_to_overlay(self, mask):
        return np.where(mask, 1.0, np.nan)

    # Build an RGBA (H, W, 4) overlay from two boolean masks.
    # kept_mask  → red   [1, 0, 0, alpha]
    # rej_mask   → blue  [0, 0.4, 1, alpha]
    def _masks_to_rgba(self, kept_mask, rej_mask, alpha: float = 0.6):
        h, w  = kept_mask.shape
        rgba  = np.zeros((h, w, 4), dtype=np.float32)  # fully transparent by default
        rgba[kept_mask] = [1.0, 0.0,  0.0,  alpha]     # red   - true CAC
        rgba[rej_mask]  = [0.0, 0.4,  1.0,  alpha]     # blue  - CNN-rejected FP
        return rgba

    # Run 7 classical steps and return (labels_array, blob_list, score_of_all_blobs).
    # This mirrors the internal logic of process_slice without the CNN step.
    def _classical_blobs(self, hu, spacing):
        from skimage.measure import regionprops
        if not is_heart_level_slice(hu):
            return None, [], 0.0

        mask = calcium_mask(hu, threshold=130.0)
        roi  = lung_guided_roi_mask(hu)
        mask = mask & roi

        labels, _ = connected_lesions(mask)
        mask, _   = filter_lesions_by_area(labels, spacing, min_area_mm2=1.0, max_area_mm2=35.0)
        labels, _ = connected_lesions(mask.astype(bool))
        mask      = filter_elongated_lesions(labels, spacing, max_eccentricity=0.97, min_solidity=0.30)
        mask      = remove_bone_like_components(hu, mask.astype(bool), spacing,
                                                peak_hu_thr=550.0, area_thr_mm2=15.0)
        labels, _ = connected_lesions(mask.astype(bool))
        blobs     = regionprops(labels)
        score     = agatston_slice_score(hu, mask, spacing, min_area_mm2=1.0)
        return labels, blobs, score

    # Rebuild a boolean mask from a subset of regionprops objects
    def _blobs_to_mask(self, labels, blobs, shape):
        out = np.zeros(shape, dtype=bool)
        for b in blobs:
            out |= (labels == b.label)
        return out

    # Compute display data for the current slice index.
    # Returns (greyscale_img, overlay_data, title_string).
    def _compute(self, idx):
        path, hu, spacing = self.series[idx]
        img      = apply_window(hu, level=50, width=350)
        filename = os.path.basename(path)
        try:
            slice_num = int(filename.split("-")[-1].split(".")[0])
            slice_label = f"Slice {slice_num}"
        except ValueError:
            slice_label = filename

        if self.mode == "hybrid":
            labels, all_blobs, score = self._classical_blobs(hu, spacing)

            if labels is None:  # not at heart level
                empty = np.zeros((*hu.shape, 4), dtype=np.float32)  # transparent RGBA
                return img, empty, f"{slice_label} | Score: 0.0 | CNN kept: 0 | CNN rejected: 0"

            # CNN FILTER - step 8
            kept_blobs = _get_cnn_classifier().filter_blobs(hu, all_blobs, spacing)
            kept_ids   = {b.label for b in kept_blobs}
            rej_blobs  = [b for b in all_blobs if b.label not in kept_ids]  # everything the CNN dropped

            kept_mask = self._blobs_to_mask(labels, kept_blobs, hu.shape)
            rej_mask  = self._blobs_to_mask(labels, rej_blobs,  hu.shape)

            # Recompute score using only CNN-approved blobs
            cnn_score = agatston_slice_score(hu, kept_mask, spacing, min_area_mm2=1.0)
            overlay   = self._masks_to_rgba(kept_mask, rej_mask)
            title     = (f"{slice_label} | Score: {cnn_score:.1f} | "
                         f"CNN kept: {len(kept_blobs)} blobs | CNN rejected: {len(rej_blobs)} blobs")
            return img, overlay, title

        else:  # classical - behaviour unchanged
            mask, score = process_slice(hu, spacing)
            if mask is None:
                mask  = np.zeros(hu.shape, dtype=bool)
                score = 0.0
            overlay = self._mask_to_overlay(mask)
            title   = f"{slice_label} | Slice Agatston: {score:.2f}"
            return img, overlay, title

    # Event handler
    def on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, len(self.series) - 1)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
        else:
            return

        img, overlay_data, title_str = self._compute(self.idx)
        self.base.set_data(img)
        self.overlay.set_data(overlay_data)
        self.ax.set_title(title_str, color="white", pad=12, fontsize=14)
        self.fig.canvas.draw_idle()



# Shows labelled 64x64 patches from a .npz file one at a time with a coloured border indicating the label.
class PatchViewer:
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.patches     = data["patches"]      # (N, 64, 64) float32
        self.labels      = data["labels"]       # (N,) int
        self.patient_ids = data["patient_ids"]  # (N,)
        self.peak_hus    = data["peak_hus"]     # (N,)
        self.area_mm2s   = data["area_mm2s"]    # (N,)
        self.n           = len(self.labels)
        self.idx         = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")

        self.im = self.ax.imshow(self.patches[0], cmap="gray", vmin=0, vmax=1)
        self.ax.axis("off")

        self._draw()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout()
        plt.show()

    # Redraws the current patch, title, and border colour.
    def _draw(self):
        patch = self.patches[self.idx]
        label = int(self.labels[self.idx])

        self.im.set_data(patch)

        label_str = "CAC" if label == 1 else "FALSE POSITIVE"
        border_colour = "red" if label == 1 else "blue"
        pid      = str(self.patient_ids[self.idx])
        peak_hu  = float(self.peak_hus[self.idx])
        area     = float(self.area_mm2s[self.idx])

        title = (f"Patch {self.idx + 1}/{self.n}  |  Patient: {pid}  |  "
                 f"Label: {label_str}  |  Peak HU: {peak_hu:.0f}  |  Area: {area:.2f} mm²")
        self.ax.set_title(title, color="white", fontsize=9, pad=8)

        for spine in self.ax.spines.values():  # coloured border signals the label at a glance
            spine.set_edgecolor(border_colour)
            spine.set_linewidth(5)
            spine.set_visible(True)

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, self.n - 1)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
        else:
            return
        self._draw()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CT slice / patch viewer")
    parser.add_argument("--patches", metavar="NPZ", default=None,
                        help="Path to a labelled patches .npz - opens the patch viewer")
    parser.add_argument("--mode", choices=["classical", "hybrid"], default="classical",
                        help="classical = red calcium overlay (default); hybrid = red kept + blue rejected")
    args, _ = parser.parse_known_args()

    if args.patches:
        print(f"Opening patch viewer: {args.patches}")
        PatchViewer(args.patches)
    else:
        folder = pick_folder("Select the patient DICOM folder")
        series = load_dicom_series(folder)
        print(f"Loaded {len(series)} slices.  [mode={args.mode}]")
        print("Use \u2190 and \u2192 arrow keys to scroll.")
        SliceViewer(series, mode=args.mode)