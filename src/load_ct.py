import os
import pydicom
import numpy as np

def load_dicom(path: str) -> np.ndarray:
    """Load a single DICOM file and return the pixel array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"DICOM not found: {path}")

    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    return img

if __name__ == "__main__":
    test_path = "data/raw/patient_10/IM-0001-0002.dcm"
    img = load_dicom(test_path)

    print("Loaded DICOM slice:", test_path)
    print("Shape:", img.shape)
    print("Min/Max:", img.min(), img.max())