import os
import glob
import pydicom
import numpy as np


# Reads every .dcm file in the folder, converts raw pixels → HU, and returns a list of (path, HU array, spacing)
def load_dicom_series(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))  # sort so slices are in correct order
    if not files:
        raise FileNotFoundError(f"No .dcm files found in: {folder}")

    series = []
    for fp in files:
        dcm = pydicom.dcmread(fp)

        img       = dcm.pixel_array.astype(np.float32)
        slope     = float(dcm.get("RescaleSlope",     1))   # scanner scale factor
        intercept = float(dcm.get("RescaleIntercept", 0))   # scanner offset
        hu        = img * slope + intercept                  # final HU = pixel × slope + intercept

        spacing = dcm.get("PixelSpacing", [1.0, 1.0])       # real world mm per pixel
        spacing = (float(spacing[0]), float(spacing[1]))

        series.append((fp, hu, spacing))

    return series