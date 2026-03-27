import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from classical.load_ct import load_dicom_series
from classical.cli import pick_folder
from classical.utils import apply_window
from classical.score_patient import process_slice

# To manually scroll through the slices, red shows calcium
class SliceViewer:
    def __init__(self, series):
        self.series = series
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

        # open the window in full screen
        # Windows vs Mac, so we try both and silently ignore any error.
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state("zoomed")
        except Exception:
            try:
                mng.window.showMaximized()
            except Exception:
                pass

        img, overlay_data, score = self._compute(self.idx)

        self.base= self.ax.imshow(img, cmap="gray")
        self.overlay = self.ax.imshow(overlay_data, cmap="Reds",
                                      alpha=0.85, vmin=0, vmax=1)
        self.ax.set_title(self._title(score), color="white", pad=12, fontsize=14)
        self.ax.axis("off")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    # Helpers
    # Conver bool mask to float aray where true is 1 and false is Nan
    def _mask_to_overlay(self, mask):
        overlay = np.where(mask, 1.0, np.nan)
        return overlay

    # Run full CAC pipeline on slice idx and return display data
    def _compute(self, idx):
        path, hu, spacing = self.series[idx]
        img = apply_window(hu, level=50, width=350)
        mask, score = process_slice(hu, spacing)

        # non heart level slices return (None, None) and show empty overlay
        if mask is None:
            mask  = np.zeros(hu.shape, dtype=bool)
            score = 0.0

        return img, self._mask_to_overlay(mask), score

    def _title(self, score):
        filename = os.path.basename(self.series[self.idx][0])
        try:
            slice_num = int(filename.split("-")[-1].split(".")[0])
            label     = f"Slice {slice_num} | {filename}"
        except ValueError:
            label = filename
        return f"{label} | Slice Agatston: {score:.2f}"

    # Event handler
    def on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, len(self.series) - 1)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
        else:
            return

        img, overlay_data, score = self._compute(self.idx)
        self.base.set_data(img)
        self.overlay.set_data(overlay_data)
        self.ax.set_title(self._title(score), color="white", pad=12, fontsize=14)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", metavar="NPZ", default=None,
                        help="Path to a labelled patches .npz file — opens the patch viewer")
    args, _ = parser.parse_known_args()

    if args.patches:
        print(f"Opening patch viewer: {args.patches}")
        PatchViewer(args.patches)
    else:
        folder = pick_folder("Select the patient DICOM folder")
        series = load_dicom_series(folder)
        print(f"Loaded {len(series)} slices.")
        print("Use \u2190 and \u2192 arrow keys to scroll.")
        SliceViewer(series)