import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from load_ct import load_dicom_series
from utils import apply_window
from score_patient import process_slice

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


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "100")
    folder = os.path.normpath(folder)

    series = load_dicom_series(folder)
    print(f"Loaded {len(series)} slices.")
    print("Use ← and → arrow keys to scroll.")
    SliceViewer(series)