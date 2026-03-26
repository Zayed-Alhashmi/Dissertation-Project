import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops


# Heart ROI  (fixed ellipse — always-reliable baseline)
# Elliptical heart / mediastinum ROI with sternum and spine cutouts.
# All parameters are fractions of image dimensions.
def heart_roi_mask(shape, cx=0.50, cy=0.52, rx=0.20, ry=0.17,sternum_y=0.25, spine_y=0.58, cut_w=0.18):
    h, w = shape
    y, x = np.ogrid[:h, :w]          # pixel coordinate grids

    cx_px = w * cx  # centre x in pixels
    cy_px = h * cy  # centre y in pixels
    rx_px = w * rx  # semi axis width in pixels
    ry_px = h * ry  # semi axis height in pixels

    ellipse = ((x - cx_px) / rx_px) ** 2 + ((y - cy_px) / ry_px) ** 2 <= 1.0  # standard ellipse equation
    sternum = (y < h * sternum_y) & (np.abs(x - cx_px) < w * cut_w) # top centre strip (sternum)
    spine   = (y > h * spine_y)   & (np.abs(x - cx_px) < w * 0.24)  # bottom centre strip (spine — wider)

    return ellipse & ~sternum & ~spine   # ellipse minus bone exclusions


# Lung guided ROI  (clips the fixed ellipse to the mediastinum column band)

# Clips the fixed ellipse to the mediastinum column band derived from detected lung positions.
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


# Heart level slice detection

# Return True if the slice contains two interior lungs one on each side of the midline.
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

# Calcium thresholding

# Return a boolean mask of pixels at or above the HU threshold
def calcium_mask(hu_img: np.ndarray, threshold: float = 130.0) -> np.ndarray:
    return hu_img >= threshold

# Label connected components in mask using 8 connectivity
def connected_lesions(mask: np.ndarray):
    structure = np.ones((3, 3), dtype=np.uint8)  # 8 connectivity kernel
    labels, num = ndi.label(mask.astype(bool), structure=structure)
    return labels, num


# Lesion area filtering

# Keep only labelled components whose area (mm²) falls within [min, max]
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


# Shape filter  (rejects rib like arcs)

# Remove rib like (high eccentricity) or arc like (low solidity) components.
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


# Bone like component rejection

# Remove components that look like cortical bone (high peak HU and large area)
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


# Display windowing

# Clip and normalise HU values to an 8-bit display window
def apply_window(img: np.ndarray, level: float = 40,
                 width: float = 400) -> np.ndarray:
    # windows bounds
    lower = level - width / 2
    upper = level + width / 2
    img = np.clip(img, lower, upper)  # clamp to window
    img = (img - lower) / (upper - lower)  # normalise to [0, 1]
    return (img * 255).astype(np.uint8)  # scale to 8-bit