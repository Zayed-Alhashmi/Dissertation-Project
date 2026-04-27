import numpy as np
from classical.utils import connected_lesions


# Converts the highest pixel density (HU) inside a lesion into a standardized weight multiplier.
def agatston_weight(max_hu: float) -> int:
    if max_hu < 130:
        return 0
    elif max_hu < 200:
        return 1  # 130-199 HU
    elif max_hu < 300:
        return 2  # 200-299 HU
    elif max_hu < 400:
        return 3  # 300-399 HU
    else:
        return 4  # 400+ HU


# Calculates the total Agatston score for a single CT slice by gathering isolated calcium lesions and scoring them.
def agatston_slice_score(hu: np.ndarray, mask: np.ndarray,
                         pixel_spacing, min_area_mm2: float = 1.0) -> float:
    ry, rx = pixel_spacing
    # Formula: Pixel Area = Height (ry) * Width (rx)
    pixel_area = ry * rx  # calculates the physical space a single pixel takes up in square millimeters

    labeled, n = connected_lesions(mask)
    score = 0.0

    for label_id in range(1, n + 1):
        lesion   = (labeled == label_id)
        
        # Formula: Total Lesion Area = Number of pixels * Area per pixel
        area_mm2 = float(lesion.sum() * pixel_area)  # convert pixel count into physical lesion area
        
        if area_mm2 < min_area_mm2:  # ignore lesions that are physically too small to be dangerous
            continue

        peak_hu = float(np.max(hu[lesion]))  # find the densest pixel inside this specific lesion
        
        # Formula: Agatston Score = Area (mm^2) * Density Factor (1 to 4)
        score  += area_mm2 * agatston_weight(peak_hu)

    return score