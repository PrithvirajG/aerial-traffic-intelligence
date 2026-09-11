"""
Geo-referencing: pixel -> real-world ground-plane coordinates (metres), via a fixed
homography per source video. One fixed homography per source is valid because both drone
sources are stationary hovers for their full duration (confirmed via telemetry - GPS/pitch
unchanged across the clip, see PLAN.md) - no per-frame reprojection needed.

Calibration is NOT yet filled in below. CALIBRATION_POINTS per source video needs >= 4
pixel<->real-world (metres, arbitrary local ground-plane origin) point correspondences, picked
from a real frame using known reference distances in the scene (e.g. standard lane width
~3.5m, a crosswalk's known width) - see LEVEL2_PLAN.md for the pragmatic-vs-rigorous
calibration tradeoff. Do not trust any homography built from this file until real points are
supplied - GroundPlaneHomography raises rather than silently using placeholder data.
"""
import cv2
import numpy as np

CALIBRATION_POINTS = {
    # source_video: [(pixel_x, pixel_y, real_x_m, real_y_m), ...] - need >= 4 points, not
    # collinear, from a real frame. TBD.
    "multiroad": None,
    "intersection": None,
}


class GroundPlaneHomography:
    def __init__(self, source_video: str):
        points = CALIBRATION_POINTS.get(source_video)
        if not points or len(points) < 4:
            raise ValueError(
                f"No calibration points set for {source_video!r} yet - fill in "
                f"CALIBRATION_POINTS in pipeline/geo_reference.py (see LEVEL2_PLAN.md's "
                f"calibration section) before using this class."
            )
        pixel_pts = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
        real_pts = np.array([(p[2], p[3]) for p in points], dtype=np.float32)
        self.H, _ = cv2.findHomography(pixel_pts, real_pts)

    def pixel_to_ground(self, x: float, y: float) -> tuple[float, float]:
        """Projects a pixel point to real-world ground-plane (x_m, y_m)."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])


def ground_contact_point(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Bottom-centre of a bounding box - the vehicle's ground-contact point, not its
    centroid. Using the centroid introduces systematic position error at oblique camera
    angles, since it sits partway up the vehicle's height rather than on the ground plane -
    see LEVEL2_PLAN.md."""
    return (x1 + x2) / 2, y2


if __name__ == "__main__":
    # Self-test of what doesn't need real calibration data yet.
    assert ground_contact_point(10, 20, 30, 60) == (20.0, 60.0)
    try:
        GroundPlaneHomography("multiroad")
        raise AssertionError("expected ValueError for missing calibration")
    except ValueError:
        pass
    print("Self-test OK (ground_contact_point + fail-fast on missing calibration).")
