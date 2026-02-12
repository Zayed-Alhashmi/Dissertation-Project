import matplotlib.pyplot as plt
from load_ct import load_dicom_series
from utils import apply_window, calcium_mask, heart_roi_mask


# Viewer class to scroll through CT slices interactively
class SliceViewer:
    def __init__(self, series):
        self.series = series
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Black background
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")

        # Remove all margins
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

        # Make window full screen (works on most systems)
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')  # Windows
        except:
            try:
                mng.window.showMaximized()  # Mac/Linux
            except:
                pass

        # Draw first slice + overlay
        self.base = self.ax.imshow(
            apply_window(self.series[self.idx][1], level=50, width=350),
            cmap="gray", aspect="auto"
        )

        hu = self.series[self.idx][1]
        roi = heart_roi_mask(hu.shape)
        mask = calcium_mask(hu, threshold=130.0) & roi

        # Create overlay once
        self.overlay = self.ax.imshow(mask, cmap="Reds", alpha=0.25, aspect="auto")

        self.ax.set_title(self._title(), color="white", pad=12, fontsize=14)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.ax.axis("off")
        plt.show()

    # Generate a readable title using the DICOM filename
    def _title(self):
        filename = self.series[self.idx][0].split("/")[-1]
        slice_number = int(filename.split("-")[-1].split(".")[0]) # to match with filenames 
        return f"Slice {slice_number} | {filename}"
    
    # left & right arrow keys to scroll slices
    def on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, len(self.series) - 1)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
        else:
            return

        hu = self.series[self.idx][1]
        img = apply_window(hu, level=50, width=350)
        roi = heart_roi_mask(hu.shape)
        mask = calcium_mask(hu, threshold=130.0) & roi

        # Update both layers
        self.base.set_data(img)
        self.overlay.set_data(mask)

        self.ax.set_title(self._title(), color="white")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    folder = "data/raw/100"
    series = load_dicom_series(folder)

    print(f"Loaded {len(series)} slices.")
    print("Use ← and → arrow keys to scroll.")
    SliceViewer(series)