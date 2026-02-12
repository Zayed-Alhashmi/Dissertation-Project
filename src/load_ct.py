# Import libraries for file handling, DICOM reading, and numerical processing
import os
import pydicom
import numpy as np

# Load all DICOM slices from a folder and convert them to Hounsfield Units (HU)
def load_dicom_series(folder):
    series = []

    # Get all DICOM files in the folder
    files = sorted([f for f in os.listdir(folder) if f.endswith(".dcm")])

    for f in files:
        path = os.path.join(folder, f)
        dcm = pydicom.dcmread(path)

        # Extract raw pixel values
        img = dcm.pixel_array.astype(np.float32)

        # Convert raw pixels to Hounsfield Units (HU)
        slope = float(dcm.get("RescaleSlope", 1))
        intercept = float(dcm.get("RescaleIntercept", 0))
        hu = img * slope + intercept

        series.append((path, hu))

    return series

# Simple test to check loading and HU conversion
if __name__ == "__main__":
    folder = "data/raw/100"
    series = load_dicom_series(folder)

    print("Number of slices:", len(series))
    print("First slice shape:", series[0][1].shape)
    print("Last slice shape:", series[-1][1].shape)