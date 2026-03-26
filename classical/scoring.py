import numpy as np
from classical.utils import connected_lesions


# Map a peak HU value to an Agatston density weight (1-4 scale)
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


# Compute the Agatston score for a single CT slice (area mm2 x density weight per lesion)
def agatston_slice_score(hu: np.ndarray, mask: np.ndarray,
                         pixel_spacing, min_area_mm2: float = 1.0) -> float:
    ry, rx = pixel_spacing
    pixel_area = ry * rx  # mm2 per pixel

    labeled, n = connected_lesions(mask)
    score = 0.0

    for label_id in range(1, n + 1):
        lesion   = (labeled == label_id)
        area_mm2 = float(lesion.sum() * pixel_area)  # lesion area in mm2

        if area_mm2 < min_area_mm2:  # skip sub-threshold lesions
            continue

        peak_hu = float(np.max(hu[lesion]))  # peak HU drives the density weight
        score  += area_mm2 * agatston_weight(peak_hu)

    return score