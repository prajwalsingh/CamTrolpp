"""Low-frequency appearance anchoring.

Colours drift a little with every diffusion pass. Matching the chroma of a
frame to the input image (a* and b* histograms in CIELAB, lightness kept)
removes that drift without touching structure.
"""

import cv2
import numpy as np


def _match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map the 8-bit channel `source` so its histogram follows `reference`."""
    cdf_src = np.histogram(source.flatten(), 256, [0, 256])[0].cumsum()
    cdf_ref = np.histogram(reference.flatten(), 256, [0, 256])[0].cumsum()
    if cdf_src[-1] == 0 or cdf_ref[-1] == 0:
        return source
    lut = np.interp(cdf_src / cdf_src[-1], cdf_ref / cdf_ref[-1], np.arange(256))
    return cv2.LUT(source, np.clip(lut, 0, 255).astype(np.uint8))


def match_lab(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the a* and b* channels of an RGB uint8 image to a reference image."""
    src = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    if src.shape[:2] != ref.shape[:2]:
        ref = cv2.resize(ref, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
    l_src, a_src, b_src = cv2.split(src)
    _, a_ref, b_ref = cv2.split(ref)
    merged = cv2.merge([l_src, _match_histogram(a_src, a_ref), _match_histogram(b_src, b_ref)])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
