import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

CROP_SIZE = 64

LBP_SCALES = [
    (1,  8),  
    (3, 24),  
]
LBP_N_BINS = [n + 2 for _, n in LBP_SCALES] 

HSV_N_BINS = 16

HOG_ORIENTATIONS = 8
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_N_FEATURES = len(hog(
    np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8),
    orientations=HOG_ORIENTATIONS,
    pixels_per_cell=HOG_PIXELS_PER_CELL,
    cells_per_block=HOG_CELLS_PER_BLOCK,
))

N_FEATURES = 1 + (6 + 3 * HSV_N_BINS) + sum(LBP_N_BINS) + HOG_N_FEATURES

def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    gray = 0.114 * B + 0.587 * G + 0.299 * R

    return np.clip(gray, 0, 255).astype(np.uint8)

def extract_coin_crop(image: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray | None:
    """Extracts a centered square crop of the image, resized to CROP_SIZE x CROP_SIZE.

    Args:
        image (np.ndarray): Original image (BGR)
        cx (float): x-coordinate of the crop's center
        cy (float): y-coordinate of the crop's center
        r (float): Radius of the crop

    Returns:
        np.ndarray | None: Resized crop, or None if outside the image
    """
    h, w = image.shape[:2]
    cx, cy, r = int(cx), int(cy), int(r)
    x1, y1 = max(cx - r, 0), max(cy - r, 0)
    x2, y2 = min(cx + r, w), min(cy + r, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def extract_coin_mask(size: int = CROP_SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    cv2.circle(mask, (center, center), center, 255, -1)
    return mask


def extract_size_feature(r: float, image_h: int = 0, image_w: int = 0) -> np.ndarray:
    if image_h > 0 and image_w > 0:
        diagonal = np.sqrt(image_h ** 2 + image_w ** 2)
        return np.array([r / diagonal], dtype=np.float32)
    return np.array([2 * r], dtype=np.float32)


def extract_color_features(crop: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Calculates the mean and standard deviation of the HSV values and an HSV histogram for the pixels in the image.

    Args:
        crop (np.ndarray): Cropped portion of the image (BGR)
        mask (np.ndarray | None): Binary mask to exclude the background. Generated if None.

    Returns:
        np.ndarray: 6 (mean+std) + 3*HSV_N_BINS (histograms) = 54 values
    """
    if mask is None:
        mask = extract_coin_mask(crop.shape[0])

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]

    mean_std = np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])

    ranges = [(0, 180), (0, 256), (0, 256)] 
    hists = []
    for ch, (lo, hi) in enumerate(ranges):
        hist, _ = np.histogram(pixels[:, ch], bins=HSV_N_BINS, range=(lo, hi), density=True)
        hists.append(hist)

    return np.concatenate([mean_std, *hists]).astype(np.float32)


def extract_lbp_features(crop: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Calculates multiscale LBP histograms to describe the texture of the object.

    Args:
        crop (np.ndarray): Crop of the object (BGR)
        mask (np.ndarray | None): Binary mask to exclude the background. Generated if None.

    Returns:
        np.ndarray: Concatenation of normalized LBP histograms = sum(LBP_N_BINS) values
    """
    if mask is None:
        mask = extract_coin_mask(crop.shape[0])

    gray = bgr_to_gray(crop)
    hists = []

    for (radius, n_points), n_bins in zip(LBP_SCALES, LBP_N_BINS):
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp[mask > 0], bins=n_bins, range=(0, n_bins), density=True)
        hists.append(hist)

    return np.concatenate(hists).astype(np.float32)


def extract_hog_features(crop: np.ndarray) -> np.ndarray:
    """Calculates an HOG (Histogram of Oriented Gradients) descriptor for the object."""
    gray = bgr_to_gray(crop)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    ).astype(np.float32)


def extract_features(image: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray | None:
    h, w = image.shape[:2]
    crop = extract_coin_crop(image, cx, cy, r)
    if crop is None:
        return None
    mask = extract_coin_mask(crop.shape[0])
    return np.concatenate([
        extract_size_feature(r, h, w),
        extract_color_features(crop, mask),
        extract_lbp_features(crop, mask),
        extract_hog_features(crop),
    ])


def extract_features_batch(image: np.ndarray, circles: np.ndarray) -> tuple[np.ndarray, list[int]]:
    features, valid_indices = [], []
    for idx, (cx, cy, r) in enumerate(circles):
        feat = extract_features(image, cx, cy, r)
        if feat is not None:
            features.append(feat)
            valid_indices.append(idx)
    if not features:
        return np.empty((0, N_FEATURES), dtype=np.float32), []
    return np.stack(features), valid_indices