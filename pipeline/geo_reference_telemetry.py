"""
Telemetry-driven geo-referencing: pixel -> real-world ground-plane metres, using the drone's
OWN .srt telemetry, with no manual calibration points needed.

This supersedes the manual-correspondence approach in geo_reference.py for these two sources.
The real telemetry carries every field a pinhole camera model needs:

    [focal_len: 24.00] [rel_alt: 70.469] [gb_pitch: -18.5] [gb_yaw: -83.4] [gb_roll: 0.0]

so the mapping is fully determined by measured data rather than by clicking reference points
on a frame and guessing real distances. Speeds come out in m/s instead of px/s.

Method: build camera intrinsics from focal length + frame size, orient the camera with the
gimbal angles, cast the ray through each pixel, and intersect it with the flat ground plane
at `rel_alt` below the camera.

World frame is East / North / Up (metres), origin on the ground directly beneath the drone.

Assumptions - each one is a real limitation, not a formality:
  * **Flat ground.** Fine for a road scene; wrong on a slope, flyover, or multi-level junction.
  * **`rel_alt` is height above the ROAD.** It is actually height above the drone's takeoff
    point. Any difference in elevation between takeoff spot and road surface scales every
    distance proportionally. `SCALE_CORRECTION` below exists to absorb this after validation.
  * **`focal_len` is a 35mm-equivalent focal length** (the DJI convention), so pixel focal
    length = frame_width * focal_len / 36. If it were a physical focal length instead, all
    distances would be off by a constant factor - which is exactly what validate() checks by
    measuring known-size objects.
  * **Accuracy degrades with distance from the camera**, badly at shallow pitch: at
    `multiroad`'s -18.5° the far end of the road is highly oblique, so a pixel of detection
    jitter there covers far more ground than one nearby. Treat far-field speeds as coarser.
"""
import math

import numpy as np

FULL_FRAME_WIDTH_MM = 36.0  # the 35mm-equivalent reference that DJI's focal_len is quoted against

# Per-source multiplier applied to projected distances, to absorb the takeoff-vs-road
# elevation difference described above. 1.0 = trust telemetry as-is. Set only from a real
# measurement (see validate_scale.py), never guessed.
SCALE_CORRECTION = {
    "multiroad": 1.0,
    "intersection": 1.0,
}


def _rot_x(deg):
    r = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])


def _rot_y(deg):
    r = math.radians(deg)
    return np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])


def _rot_z(deg):
    r = math.radians(deg)
    return np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])


class TelemetryGroundProjector:
    """Projects image pixels onto the ground plane using one telemetry sample.

    Both sources are stationary hovers, so a single sample describes the whole clip and no
    per-frame reprojection is needed (confirmed via telemetry - see PLAN.md).
    """

    def __init__(self, frame_width, frame_height, focal_len_35mm, rel_alt_m,
                 gimbal_pitch_deg, gimbal_yaw_deg=0.0, gimbal_roll_deg=0.0, scale=1.0):
        self.w, self.h = frame_width, frame_height
        self.height_m = rel_alt_m * scale
        self.scale = scale
        # Pixel focal length from the 35mm-equivalent value. Same for x and y (square pixels).
        self.f_px = frame_width * focal_len_35mm / FULL_FRAME_WIDTH_MM
        self.cx, self.cy = frame_width / 2.0, frame_height / 2.0

        # Base orientation: camera level, looking North. Camera axes are the usual CV
        # convention - x right, y DOWN, z forward along the optical axis - so as world
        # (East, North, Up) columns: x_c=East, y_c=Down, z_c=North.
        r_base = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, -1.0, 0.0]])
        # Pitch about the camera's right axis (East): -90 deg points the optical axis straight
        # down (nadir), 0 deg leaves it horizontal. Yaw is a compass heading (clockwise from
        # North), hence the negative sign for a right-handed rotation about Up.
        self.R = _rot_z(-gimbal_yaw_deg) @ _rot_x(gimbal_pitch_deg) @ _rot_y(gimbal_roll_deg) @ r_base

    def pixel_to_ground(self, u, v):
        """Returns (east_m, north_m) on the ground for image pixel (u, v), or None if the ray
        points at or above the horizon (no ground intersection - e.g. sky/far buildings)."""
        d_cam = np.array([(u - self.cx) / self.f_px, (v - self.cy) / self.f_px, 1.0])
        d_world = self.R @ d_cam
        if d_world[2] >= -1e-9:      # not pointing downward: no ground hit
            return None
        t = self.height_m / -d_world[2]
        return float(t * d_world[0]), float(t * d_world[1])

    def pixels_to_ground(self, us, vs):
        """Vectorized form. Returns (east_m, north_m) float arrays with NaN where a ray misses
        the ground - per-row Python calls are far too slow for whole-CSV enrichment."""
        us = np.asarray(us, dtype=float)
        vs = np.asarray(vs, dtype=float)
        d_cam = np.stack([(us - self.cx) / self.f_px, (vs - self.cy) / self.f_px, np.ones_like(us)])
        d_world = self.R @ d_cam
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d_world[2] < -1e-9, self.height_m / -d_world[2], np.nan)
        return t * d_world[0], t * d_world[1]

    def metres_per_pixel_at(self, u, v):
        """Local ground scale at a pixel - useful for sanity-checking how much a pixel of
        detection jitter is worth at that point in the frame (it varies hugely at low pitch)."""
        p0 = self.pixel_to_ground(u, v)
        p1 = self.pixel_to_ground(u + 1, v)
        if p0 is None or p1 is None:
            return None
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def from_telemetry_frame(tf, frame_width, frame_height, source_video=None):
    """Builds a projector from one parsed TelemetryFrame (pipeline/srt_parser.py)."""
    focal = _focal_from_raw(tf.raw_text)
    if focal is None:
        raise ValueError("No focal_len in telemetry - cannot build a camera model")
    if tf.rel_altitude is None:
        raise ValueError("No rel_alt in telemetry - cannot build a camera model")
    if tf.gimbal_pitch is None:
        raise ValueError("No gb_pitch in telemetry - cannot build a camera model")
    return TelemetryGroundProjector(
        frame_width, frame_height, focal, tf.rel_altitude,
        tf.gimbal_pitch, tf.gimbal_yaw or 0.0, tf.gimbal_roll or 0.0,
        scale=SCALE_CORRECTION.get(source_video, 1.0),
    )


def _focal_from_raw(raw_text):
    import re
    m = re.search(r"focal_len\s*:?\s*([\d.]+)", raw_text, re.IGNORECASE)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    # Self-test 1: exact nadir. Straight down from 100m with a 90 deg-ish FOV - the ground
    # point under the principal point must be directly beneath the drone, and the geometry
    # must be symmetric about it. Checks the rotation convention, not just that it runs.
    p = TelemetryGroundProjector(4000, 3000, 24.0, 100.0, -90.0)
    centre = p.pixel_to_ground(2000, 1500)
    assert centre is not None and abs(centre[0]) < 1e-6 and abs(centre[1]) < 1e-6, centre
    right = p.pixel_to_ground(3000, 1500)
    left = p.pixel_to_ground(1000, 1500)
    assert abs(right[0] + left[0]) < 1e-6, (left, right)   # symmetric E/W
    assert abs(right[1]) < 1e-6 and abs(left[1]) < 1e-6    # no N/S drift along an image row
    # At nadir, f_px = 4000*24/36 = 2666.7, so 1000px off-axis => 100m * 1000/2666.7 = 37.5m.
    assert abs(right[0] - 37.5) < 1e-3, right

    # Self-test 2: horizon handling. Level camera - the upper half of the frame is sky and
    # must return None rather than a bogus huge distance.
    lvl = TelemetryGroundProjector(4000, 3000, 24.0, 100.0, 0.0)
    assert lvl.pixel_to_ground(2000, 100) is None
    assert lvl.pixel_to_ground(2000, 2900) is not None

    # Self-test 3: oblique pitch pushes the ground point away from the drone, and scale per
    # pixel grows with distance - the accuracy caveat in the docstring, made measurable.
    ob = TelemetryGroundProjector(3840, 2160, 24.0, 70.0, -18.5)
    near = ob.metres_per_pixel_at(1920, 2000)
    far = ob.metres_per_pixel_at(1920, 1200)
    # Measured 1.82x over this 800px span at -18.5 deg. Asserting monotonic growth with a
    # margin, not a specific ratio - the exact figure is geometry, not a spec.
    assert near is not None and far is not None and far > near * 1.5, (near, far)

    # Self-test 4: vectorized path agrees with the scalar one.
    e, n = ob.pixels_to_ground([1920, 1000], [2000, 1800])
    for i, (u, v) in enumerate([(1920, 2000), (1000, 1800)]):
        s = ob.pixel_to_ground(u, v)
        assert abs(e[i] - s[0]) < 1e-9 and abs(n[i] - s[1]) < 1e-9

    print("Self-test OK - nadir symmetry, horizon rejection, oblique scale growth, vectorized match.")
    print(f"  multiroad-like (-18.5 deg, 70m): {near:.3f} m/px near-field, {far:.3f} m/px far-field")
