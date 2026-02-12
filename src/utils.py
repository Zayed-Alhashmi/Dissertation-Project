import numpy as np

def calcium_mask(hu_img: np.ndarray, threshold: float = 130.0) -> np.ndarray:
    """Return a boolean mask where HU >= threshold."""
    return hu_img >= threshold

def apply_window(img, level=40, width=400):
    lower = level - width / 2
    upper = level + width / 2

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    img = (img * 255).astype(np.uint8)

    return img

def heart_roi_mask(shape, x0=0.35, x1=0.65, y0=0.32, y1=0.58):
    """
    Returns a boolean mask for a central 'heart area' box.
    Fractions are relative to image width/height.
    """
    h, w = shape
    X0, X1 = int(w * x0), int(w * x1)
    Y0, Y1 = int(h * y0), int(h * y1)

    m = np.zeros((h, w), dtype=bool)
    m[Y0:Y1, X0:X1] = True
    return m