import numpy as np
from skimage import color
from scipy import ndimage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NORMALIZATION, TISSUE_DETECTION


# =============================================================================
# Tissue Detection
# =============================================================================

def detect_tissue(
    img: np.ndarray, l_threshold: float = TISSUE_DETECTION["l_threshold"],
    color_threshold: float = TISSUE_DETECTION["color_threshold"],
    variance_threshold: float = TISSUE_DETECTION["variance_threshold"],
    min_criteria: int = TISSUE_DETECTION["min_criteria"],
) -> tuple:
    """
    Detect tissue in a patch using LAB color space.

    Combines brightness, color saturation, and local texture criteria.
    Handles lightly-stained hematoxylin that optical density methods miss.

    Parameters
    ----------
    img : ndarray (H, W, 3), uint8 RGB
    l_threshold : float
        Max L* value to count as tissue (0-100).
    color_threshold : float
        Min color magnitude sqrt(a^2 + b^2).
    variance_threshold : float
        Min local grayscale variance.
    min_criteria : int
        Must pass at least this many of the 3 criteria.

    Returns
    -------
    tissue_mask : ndarray (H, W), bool
    tissue_fraction : float
    """
    img_lab = color.rgb2lab(img.astype(np.float32) / 255.0)
    h, w = img_lab.shape[:2]

    # Criterion 1: not too bright
    l_mask = img_lab[:, :, 0] < l_threshold

    # Criterion 2: has color (excludes pure gray/white)
    color_mag = np.sqrt(img_lab[:, :, 1] ** 2 + img_lab[:, :, 2] ** 2)
    color_mask = color_mag > color_threshold

    # Criterion 3: local texture
    gray = np.mean(img, axis=2).astype(float)
    local_mean = ndimage.uniform_filter(gray, size=5)
    local_sqr_mean = ndimage.uniform_filter(gray ** 2, size=5)
    local_var = local_sqr_mean - local_mean ** 2
    var_mask = local_var > variance_threshold

    # Combine: must pass at least min_criteria of 3
    score = l_mask.astype(int) + color_mask.astype(int) + var_mask.astype(int)
    tissue_mask = score >= min_criteria
    tissue_fraction = np.sum(tissue_mask) / (h * w)

    return tissue_mask, tissue_fraction


# =============================================================================
# LAB Statistics
# =============================================================================

def compute_lab_stats(img: np.ndarray, tissue_mask: np.ndarray = None) -> dict:
    """
    Compute LAB channel statistics, optionally masked to tissue pixels.

    Parameters
    ----------
    img : ndarray (H, W, 3), uint8 RGB
    tissue_mask : ndarray (H, W), bool, optional

    Returns
    -------
    dict with keys L_mean, L_std, a_mean, a_std, b_mean, b_std
    """
    img_lab = color.rgb2lab(img.astype(np.float32) / 255.0)

    if tissue_mask is not None and np.sum(tissue_mask) > 1000:
        pixels = img_lab[tissue_mask]
    else:
        pixels = img_lab.reshape(-1, 3)

    return {
        "L_mean": float(np.mean(pixels[:, 0])),
        "L_std": float(np.std(pixels[:, 0])),
        "a_mean": float(np.mean(pixels[:, 1])),
        "a_std": float(np.std(pixels[:, 1])),
        "b_mean": float(np.mean(pixels[:, 2])),
        "b_std": float(np.std(pixels[:, 2])),
    }


def classify_stain(stats: dict) -> str:
    """
    Classify dominant stain from LAB statistics.

    In LAB space:
      - Hematoxylin (blue-purple): negative b* values
      - Eosin (pink): positive a* values

    Returns
    -------
    str : 'hematoxylin', 'eosin', or 'mixed'
    """
    if stats["b_mean"] < -2:
        return "hematoxylin"
    elif stats["a_mean"] > 5:
        return "eosin"
    return "mixed"


# =============================================================================
# Modified Reinhard Normalization
# =============================================================================

def normalize_patch(
    patch: np.ndarray,
    source_stats: dict,
    ref_stats: dict = None,
    ab_strength: float = None,
) -> np.ndarray:
    """
    Modified Reinhard normalization for LH&E tissue.

    L channel: full proportional normalization with contrast preservation.
    a/b channels: partial normalization (ab_strength controls blending).

    Parameters
    ----------
    patch : ndarray (H, W, 3), uint8 RGB
    source_stats : dict
        LAB stats of this specific patch (from compute_lab_stats).
    ref_stats : dict, optional
        Reference LAB stats. Defaults to config.NORMALIZATION values.
    ab_strength : float, optional
        Blending weight for a/b channels. Defaults to config value.

    Returns
    -------
    ndarray (H, W, 3), uint8 RGB
    """
    if ref_stats is None:
        ref_stats = {
            "L_mean": NORMALIZATION["ref_L_mean"],
            "L_std": NORMALIZATION["ref_L_std"],
            "a_mean": NORMALIZATION["ref_a_mean"],
            "a_std": NORMALIZATION["ref_a_std"],
            "b_mean": NORMALIZATION["ref_b_mean"],
            "b_std": NORMALIZATION["ref_b_std"],
        }
    if ab_strength is None:
        ab_strength = NORMALIZATION["ab_strength"]

    img_lab = color.rgb2lab(patch.astype(np.float32) / 255.0)

    # --- L channel: proportional mapping with contrast preservation ---
    src_L_std = max(source_stats["L_std"], 0.001)
    ref_L_std = max(ref_stats["L_std"], 0.001)

    std_ratio = ref_L_std / src_L_std
    # If source already has more contrast than reference, don't reduce it
    if std_ratio < 1.0:
        std_ratio = max(std_ratio, 1.0)

    img_lab[:, :, 0] = ref_stats["L_mean"] + (
        img_lab[:, :, 0] - source_stats["L_mean"]
    ) * std_ratio

    # --- a/b channels: partial normalization ---
    for ch, name in [(1, "a"), (2, "b")]:
        src_mean = source_stats[f"{name}_mean"]
        src_std = max(source_stats[f"{name}_std"], 0.001)
        ref_mean = ref_stats[f"{name}_mean"]
        ref_std = max(ref_stats[f"{name}_std"], 0.001)

        full_norm = (img_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean
        img_lab[:, :, ch] += ab_strength * (full_norm - img_lab[:, :, ch])

    # Clip to valid LAB ranges
    img_lab[:, :, 0] = np.clip(img_lab[:, :, 0], 0, 100)
    img_lab[:, :, 1] = np.clip(img_lab[:, :, 1], -127, 127)
    img_lab[:, :, 2] = np.clip(img_lab[:, :, 2], -127, 127)

    # Convert back to RGB uint8
    img_rgb = color.lab2rgb(img_lab)
    return (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)


def normalize_image(img: np.ndarray, ref_stats: dict = None, ab_strength: float = None) -> np.ndarray:
    """
    Convenience function: detect tissue, compute stats, normalize.

    Parameters
    ----------
    img : ndarray (H, W, 3), uint8 RGB
    ref_stats : dict, optional
    ab_strength : float, optional

    Returns
    -------
    ndarray (H, W, 3), uint8 RGB
    """
    tissue_mask, tissue_frac = detect_tissue(img)
    source_stats = compute_lab_stats(img, tissue_mask)
    return normalize_patch(img, source_stats, ref_stats, ab_strength)