"""
Dominant vehicle colour from a cropped bounding-box image, via k-means on pixel colours.
The aerial top-down view mostly shows the vehicle roof - the least-occluded, most
colour-uniform part of a vehicle visible from this angle - see LEVEL2_PLAN.md.
"""
import cv2
import numpy as np


def dominant_colour(crop_bgr: np.ndarray, k: int = 3, sample_max: int = 2000) -> tuple[int, int, int]:
    """
    crop_bgr: HxWx3 uint8 BGR image (a vehicle's cropped bounding box).
    Returns (b, g, r) of the largest k-means cluster's centre colour.
    """
    pixels = crop_bgr.reshape(-1, 3).astype(np.float32)
    if len(pixels) > sample_max:
        idx = np.random.default_rng(0).choice(len(pixels), sample_max, replace=False)
        pixels = pixels[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten())
    dominant_idx = counts.argmax()
    b, g, r = centers[dominant_idx]
    return int(b), int(g), int(r)


if __name__ == "__main__":
    # Self-test: a synthetic crop that's 70% solid red, 30% noise - confirms the dominant
    # cluster correctly recovers the majority colour despite noise, without needing a real
    # vehicle crop.
    rng = np.random.default_rng(0)
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    crop[:, :] = (20, 20, 200)  # BGR: strong red
    noise_mask = rng.random((50, 50)) < 0.3
    crop[noise_mask] = rng.integers(0, 255, size=(noise_mask.sum(), 3), dtype=np.uint8)

    b, g, r = dominant_colour(crop)
    print(f"Self-test: recovered dominant colour BGR=({b},{g},{r})")
    assert r > 150 and r > b and r > g, "expected red to dominate a 70%-red synthetic crop"
    print("Self-test OK.")
