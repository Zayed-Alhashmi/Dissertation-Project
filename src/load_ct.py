import os
import pydicom
import numpy as np
import glob

def load_dicom_series(folder: str):
    """Load all DICOM files in a folder and return a list of (filename, image_array)."""
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .dcm files found in: {folder}")

    series = []
    for fp in files:
        ds = pydicom.dcmread(fp)
        img = ds.pixel_array.astype(np.float32)
        series.append((fp, img))

    return series

if __name__ == "__main__":
    folder = "data/raw/patient_10"
    series = load_dicom_series(folder)

    print("Number of slices:", len(series))
    print("First slice:", series[0][0], "shape:", series[0][1].shape)
    print("Last slice:", series[-1][0], "shape:", series[-1][1].shape)