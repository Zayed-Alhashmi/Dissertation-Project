import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops


# Generates a fixed elliptical mask over the heart region, explicitly cutting out the sternum and spine.
def heart_roi_mask(shape, cx=0.50, cy=0.52, rx=0.20, ry=0.17,sternum_y=0.25, spine_y=0.58, cut_w=0.18):
    h, w = shape
    y, x = np.ogrid[:h, :w]          # pixel coordinate grids

    cx_px = w * cx  # centre x in pixels
    cy_px = h * cy  # centre y in pixels
    rx_px = w * rx  # semi axis width in pixels
    ry_px = h * ry  # semi axis height in pixels

    # Formula: Ellipse Boundary = ( (X - Center_X)/Radius_X )^2 + ( (Y - Center_Y)/Radius_Y )^2 <= 1
    ellipse = ((x - cx_px) / rx_px) ** 2 + ((y - cy_px) / ry_px) ** 2 <= 1.0
    sternum = (y < h * sternum_y) & (np.abs(x - cx_px) < w * cut_w) # top centre strip (sternum)
    spine   = (y > h * spine_y)   & (np.abs(x - cx_px) < w * 0.24)  # bottom centre strip (spine - wider)

    return ellipse & ~sternum & ~spine   # ellipse minus bone exclusions


# Isolates the central heart band by detecting the inner bounds of the left and right lungs.
def lung_guided_roi_mask(hu: np.ndarray,
                         lung_hu_thr: float = -400.0,
                         min_lung_area_px: int = 3000,
                         heart_margin_px: int = 10) -> np.ndarray:
    h, w = hu.shape
    fallback = heart_roi_mask(hu.shape)   # used if lung detection fails
    midline  = w // 2

    x_lo, x_hi = w * 0.10, w * 0.90  # valid centroid x range (inner 80 %)
    y_lo, y_hi = h * 0.10, h * 0.90  # same as x but for y range

    air_mask   = hu < lung_hu_thr  # threshold to isolate air regions
    labeled, n = ndi.label(air_mask)  # label connected air components
    if n == 0:
        return fallback

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    left_lung  = None
    right_lung = None
    left_size  = 0
    right_size = 0

    for label_id in range(1, n + 1):
        if sizes[label_id] < min_lung_area_px: # skip small blobs
            continue
        component = (labeled == label_id)

        if (component[0, :].any()  or component[-1, :].any() or
                component[:, 0].any() or component[:,  -1].any()):
            continue  # reject exterior air (touches border)

        props = regionprops(component.astype(np.uint8))
        if not props:
            continue

        cy_c, cx_c = props[0].centroid
        if not (x_lo < cx_c < x_hi and y_lo < cy_c < y_hi):
            continue  # centroid outside inner body region

        if cx_c < midline:  # assign to left or right lung
            if sizes[label_id] > left_size:
                left_lung = component
                left_size = sizes[label_id]
        else:
            if sizes[label_id] > right_size:
                right_lung = component
                right_size = sizes[label_id]

    if left_lung is None or right_lung is None:
        return fallback

    row_start = h // 4  # use middle 50 % of rows for stability
    row_end   = 3 * h // 4

    left_inner_cols  = []
    right_inner_cols = []

    for row in range(row_start, row_end):
        l_cols = np.where(left_lung[row, :])[0]
        r_cols = np.where(right_lung[row, :])[0]
        if l_cols.size > 0:
            left_inner_cols.append(int(l_cols.max()))   # rightmost edge of left lung
        if r_cols.size > 0:
            right_inner_cols.append(int(r_cols.min()))  # leftmost edge of right lung

    if not left_inner_cols or not right_inner_cols:
        return fallback

    col_left  = int(np.median(left_inner_cols))  + heart_margin_px  # mediastinum left bound
    col_right = int(np.median(right_inner_cols)) - heart_margin_px  # mediastinum right bound

    if col_right - col_left < int(0.20 * w):
        return fallback  # band too narrow, likely bad detection

    x_grid   = np.broadcast_to(np.arange(w)[np.newaxis, :], (h, w))
    col_band = (x_grid >= col_left) & (x_grid <= col_right)  # column mask for mediastinum

    return fallback & col_band  # intersect ellipse with column band


# Validates if a CT slice is at the heart level by checking for the presence of two lungs.
def is_heart_level_slice(hu: np.ndarray,
                         lung_hu_thr: float = -400.0,
                         min_lung_frac: float = 0.07) -> bool:
    h, w = hu.shape
    min_lung_area_px = int(h * w * min_lung_frac)  # scale with image resolution
    air_mask   = hu < lung_hu_thr  # isolate air regions
    labeled, n = ndi.label(air_mask)  # label connected components
    if n == 0:
        return False

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    x_lo, x_hi = w * 0.20, w * 0.80  # valid centroid x range (inner 60 %)
    y_lo, y_hi = h * 0.20, h * 0.80  # same for x but for y range
    midline     = w // 2

    left_ok  = False
    right_ok = False

    for label_id in range(1, n + 1):
        if sizes[label_id] < min_lung_area_px:  # skip small blobs
            continue

        component = (labeled == label_id)

        if (component[0, :].any()  or component[-1, :].any() or
                component[:, 0].any() or component[:,  -1].any()):
            continue  # reject exterior air (touches border)

        props = regionprops(component.astype(np.uint8))
        if not props:
            continue

        cy_c, cx_c = props[0].centroid

        if not (x_lo < cx_c < x_hi and y_lo < cy_c < y_hi):
            continue  # centroid outside inner body region

        if cx_c < midline:  # assign to left or right side
            left_ok  = True
        else:
            right_ok = True

    return left_ok and right_ok  # both sides must be present

# Extracts pixels that meet or exceed the standard Hounsfield Unit threshold for calcification.
def calcium_mask(hu_img: np.ndarray, threshold: float = 130.0) -> np.ndarray:
    return hu_img >= threshold

# Groups adjacent calcium pixels together into distinct isolated blocks (lesions).
def connected_lesions(mask: np.ndarray):
    structure = np.ones((3, 3), dtype=np.uint8)  # 8 connectivity kernel
    labels, num = ndi.label(mask.astype(bool), structure=structure)
    return labels, num


# Removes calcium segments that are physically too small or too large based on their physical surface area.
def filter_lesions_by_area(labels: np.ndarray, spacing,
                           min_area_mm2: float = 1.0,
                           max_area_mm2: float | None = None):
    row_sp, col_sp = spacing
    pixel_area = row_sp * col_sp  # mm² per pixel

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background label

    areas_mm2 = counts * pixel_area  # convert pixel counts to mm²

    keep = areas_mm2 >= min_area_mm2
    if max_area_mm2 is not None:
        keep = keep & (areas_mm2 <= max_area_mm2)  # apply upper bound if given

    return keep[labels], keep


# Eliminates false positives like ribs or arcs by checking if the structure is too long or hollow.
def filter_elongated_lesions(labels: np.ndarray,
                             spacing,
                             max_eccentricity: float = 0.98,
                             min_solidity: float = 0.30,
                             min_pixels: int = 30) -> np.ndarray:
    props = regionprops(labels)
    keep_labels = set()

    for prop in props:
        if prop.area < min_pixels:  # shape metrics unreliable at small scale, keep as-is
            keep_labels.add(prop.label)
            continue
        if prop.eccentricity > max_eccentricity:  # too elongated (rib like)
            continue
        if prop.solidity < min_solidity:  # too hollow (arc like)
            continue
        keep_labels.add(prop.label)

    return (np.isin(labels, list(keep_labels))
            if keep_labels else np.zeros_like(labels, dtype=bool))


# Ignores dense massive structures like cortical bone that exceed expected calcium limits.
def remove_bone_like_components(hu: np.ndarray,
                                mask: np.ndarray,
                                spacing,
                                peak_hu_thr: float = 550.0,
                                area_thr_mm2: float = 15.0) -> np.ndarray:
    labels, n = connected_lesions(mask.astype(bool))
    out = np.zeros_like(mask, dtype=bool)

    ry, rx = spacing
    pixel_area = ry * rx  # mm² per pixel

    for label_id in range(1, n + 1):
        lesion   = (labels == label_id)
        area_mm2 = float(lesion.sum() * pixel_area)  # lesion area in mm²
        if area_mm2 <= 0:
            continue
        peak = float(np.max(hu[lesion]))  # peak HU in this component
        if peak > peak_hu_thr and area_mm2 > area_thr_mm2:  # bone like, skip
            continue
        out |= lesion

    return out


# Prevents spinal bone artifacts from being scored by ignoring the bottom central coordinate zone.
def filter_spine_region_blobs(blobs: list, hu_slice: np.ndarray,
                               spine_row_frac: float = 0.65,
                               spine_col_frac: float = 0.20) -> list:
    h, w    = hu_slice.shape
    mid_col = w / 2.0
    col_tol = w * spine_col_frac  # half-width of the spine exclusion band

    keep = []
    for blob in blobs:
        cy, cx = blob.centroid
        in_spine_rows = cy > h * spine_row_frac           # below the threshold row
        in_spine_cols = abs(cx - mid_col) < col_tol       # near the centre column
        if in_spine_rows and in_spine_cols:
            continue  # spine zone - reject
        keep.append(blob)
    return keep


# Discards artifacts found on the extreme left and right boundaries of the chest cavity.
def filter_lateral_chest_wall_blobs(blobs: list, hu_slice: np.ndarray,
                                     wall_frac: float = 0.10) -> list:
    _, w = hu_slice.shape
    col_lo = w * wall_frac        # left exclusion boundary
    col_hi = w * (1.0 - wall_frac)  # right exclusion boundary

    return [b for b in blobs if col_lo <= b.centroid[1] <= col_hi]


# Finds the ascending aorta by looking for the largest, brightest perfect circle in the chest cavity.
def detect_aorta_circle(hu_slice: np.ndarray, spacing,
                         hu_thr: float = 300.0,
                         min_area_mm2: float = 100.0,
                         max_area_mm2: float = 1200.0,
                         max_eccentricity: float = 0.85,
                         radius_margin: float = 2.00,
                         debug: bool = False):
    from scipy.ndimage import binary_closing
    import math
    ry, rx   = spacing
    pix_area = ry * rx  # mm² per pixel
    h, w     = hu_slice.shape

    bright = hu_slice > hu_thr                              # isolate tissue brighter than threshold
    struct = np.ones((5, 5), dtype=bool)
    bright = binary_closing(bright, structure=struct)       # close gaps in the aortic wall

    labeled, n = ndi.label(bright)
    if n == 0:
        return (None, []) if debug else None

    props  = regionprops(labeled)
    col_lo = w * 0.20  # inner 60 % column band - aorta never near the edge
    col_hi = w * 0.80
    row_lo = h * 0.25  # aorta never in top 25 % (avoids trachea / shoulder bright spots)
    row_hi = h * 0.70  # aorta never in bottom 30 % - spine / lower mediastinum

    candidates = []
    for prop in props:
        area_mm2 = prop.area * pix_area
        if area_mm2 < min_area_mm2 or area_mm2 > max_area_mm2:
            continue  # wrong size - skip
        if prop.eccentricity > max_eccentricity:
            continue  # too elongated - rib or aortic arc
        cy, cx = prop.centroid
        if not (col_lo < cx < col_hi):
            continue  # too far to the side
        if cy < row_lo or cy > row_hi:
            continue  # outside expected aorta row band
        candidates.append({"cy": cy, "cx": cx,
                            "area_mm2": area_mm2, "eccentricity": prop.eccentricity})

    if not candidates:
        return (None, []) if debug else None

    best      = min(candidates, key=lambda c: c["eccentricity"])  # most circular
    # Formula: Radius = Square_Root( Area / Pi )
    radius_mm = math.sqrt(best["area_mm2"] / math.pi) * radius_margin  # add safety margin
    radius_mm = max(radius_mm, 20.0)  # floor: aorta exclusion zone never smaller than 20 mm

    result = (best["cy"], best["cx"], radius_mm)
    return (result, candidates) if debug else result


# Excludes calcium clusters lying inside the aorta since we only score coronary arteries.
def filter_aortic_blobs(blobs: list, hu_slice: np.ndarray, spacing,
                         slice_idx: int = None,
                         aorta_cache: dict = None) -> list:
    import math
    aorta = detect_aorta_circle(hu_slice, spacing)

    # interpolation fallback: use nearest cached detection when direct detection fails
    if aorta is None and aorta_cache and slice_idx is not None:
        cached_keys = sorted(aorta_cache.keys(), key=lambda k: abs(k - slice_idx))
        if cached_keys:
            aorta = aorta_cache[cached_keys[0]]  # nearest neighbour in slice index

    if aorta is None:
        return blobs  # no aorta position available - keep all blobs

    ry, rx            = spacing
    cy, cx, radius_mm = aorta

    keep = []
    for blob in blobs:
        by, bx  = blob.centroid
        dist_mm = math.sqrt(((by - cy) * ry) ** 2 + ((bx - cx) * rx) ** 2)  # pixel → mm
        if dist_mm < radius_mm:
            continue  # inside aorta circle - reject
        keep.append(blob)
    return keep


# Compresses the dynamic range of medical scan data into an 8-bit image for viewing.
def apply_window(img: np.ndarray, level: float = 40,
                 width: float = 400) -> np.ndarray:
    # windows bounds
    lower = level - width / 2
    upper = level + width / 2
    img = np.clip(img, lower, upper)  # clamp to window
    img = (img - lower) / (upper - lower)  # normalise to [0, 1]
    return (img * 255).astype(np.uint8)  # scale to 8-bit