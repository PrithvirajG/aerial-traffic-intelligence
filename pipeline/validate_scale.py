"""
Empirically validates telemetry-driven geo-referencing against known real-world vehicle sizes.

This is the check that decides whether the camera model is trustworthy, and specifically
whether the two assumptions that scale ALL distances linearly are right:
  1. `focal_len` in the .srt is a 35mm-equivalent focal length (DJI convention), and
  2. `rel_alt` (height above takeoff) is close enough to height above the road.

If both hold, detected cars must measure roughly 3.5-5.5 m long and 1.6-2.2 m wide, and buses
must come out clearly longer than cars. If everything is off by a constant factor, the factor
itself tells us what to put in geo_reference_telemetry.SCALE_CORRECTION.

Method: for each track's single largest (clearest) detection, project the bbox's two bottom
corners to the ground and measure the distance between them. At an oblique angle a bbox's
bottom edge tracks the vehicle's ground footprint far better than its full height does, so
this measures footprint width rather than being inflated by vehicle height.

Usage: python pipeline/validate_scale.py
"""
import glob
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.geo_reference_telemetry import from_telemetry_frame
from pipeline.srt_parser import parse_srt_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRT = {
    "multiroad": os.path.join(REPO, "videos", "clips", "compressed", "multiroad_full.srt"),
    "intersection": os.path.join(REPO, "videos", "clips", "compressed", "intersection_full.srt"),
}
# Real-world footprint width (across the vehicle) for a rough plausibility band, in metres.
EXPECTED = {"car": (1.5, 2.3), "bus": (2.3, 3.0), "motorcycle": (0.5, 1.2), "pedestrian": (0.3, 0.9)}


def projector_for(source, frame_w, frame_h):
    frames = parse_srt_file(SRT[source])
    if not frames:
        raise SystemExit(f"No telemetry parsed from {SRT[source]}")
    tf = frames[len(frames) // 2]      # mid-clip sample; stationary hover so any frame works
    proj = from_telemetry_frame(tf, frame_w, frame_h, source_video=source)
    return proj, tf


def main():
    for model_dir in sorted(glob.glob(os.path.join(REPO, "videos", "samples", "*", "10s", "*botsort"))):
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        source = meta["source_video"]
        fw, fh = meta["frame_width"], meta["frame_height"]
        proj, tf = projector_for(source, fw, fh)

        print(f"\n=== {source} ===")
        print(f"  telemetry: focal_len(35mm-eq) from srt, rel_alt={tf.rel_altitude} m, "
              f"pitch={tf.gimbal_pitch}, yaw={tf.gimbal_yaw}")
        print(f"  f_px={proj.f_px:.1f}, camera height used={proj.height_m:.1f} m")

        df = pd.read_csv(os.path.join(model_dir, "detections.csv"))
        df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
        best = df.loc[df.groupby("track_id")["area"].idxmax()]

        rows = []
        for r in best.itertuples():
            # Bottom-left and bottom-right corners: the vehicle's ground footprint edge.
            pl = proj.pixel_to_ground(r.x1, r.y2)
            pr = proj.pixel_to_ground(r.x2, r.y2)
            if pl is None or pr is None:
                continue
            width_m = math.hypot(pr[0] - pl[0], pr[1] - pl[1])
            rows.append({"cls": r.taxonomy_class, "width_m": width_m, "px": r.x2 - r.x1})
        out = pd.DataFrame(rows)
        if out.empty:
            print("  no projectable detections (all rays missed the ground)")
            continue

        print(f"  {len(out)} tracks projected onto the ground plane")
        for cls, grp in out.groupby("cls"):
            med = grp["width_m"].median()
            band = EXPECTED.get(cls)
            verdict = ""
            if band:
                lo, hi = band
                verdict = "  <-- PLAUSIBLE" if lo <= med <= hi else f"  <-- off (expect {lo}-{hi} m)"
            print(f"    {cls:16s} n={len(grp):4d}  median footprint width {med:6.2f} m"
                  f"  (median {grp['px'].median():.0f} px){verdict}")

        cars = out[out["cls"] == "car"]["width_m"]
        if len(cars) > 10:
            implied = 1.85 / cars.median()   # a typical car is ~1.85 m wide
            print(f"  => implied SCALE_CORRECTION for '{source}' if cars should be 1.85 m wide: "
                  f"{implied:.3f}")


if __name__ == "__main__":
    main()
