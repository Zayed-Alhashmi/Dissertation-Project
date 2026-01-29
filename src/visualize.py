import matplotlib.pyplot as plt
from load_ct import load_dicom_series

class SliceViewer:
    def __init__(self, series):
        self.series = series
        self.idx = 0

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.img_plot = self.ax.imshow(
            self.series[self.idx][1], cmap="gray"
        )
        self.ax.set_title(self._title())
        self.ax.axis("off")

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def _title(self):
        filename = self.series[self.idx][0].split("/")[-1]
        slice_number = int(filename.split("-")[-1].split(".")[0]) # to match with filenames 
        return f"Slice {slice_number} | {filename}"

    def on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, len(self.series) - 1)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
        else:
            return

        self.img_plot.set_data(self.series[self.idx][1])
        self.ax.set_title(self._title())
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    folder = "data/raw/patient_10"
    series = load_dicom_series(folder)

    print(f"Loaded {len(series)} slices.")
    print("Use ← and → arrow keys to scroll.")
    SliceViewer(series)