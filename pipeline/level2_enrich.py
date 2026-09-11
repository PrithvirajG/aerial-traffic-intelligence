"""
Level 2 local enrichment: adds kinematics (speed/accel) to detections.csv and builds a
per-track tracks_summary.csv (dominant colour + kinematics + timing), from a Kaggle tracking
run's output plus the exact source clip it was run against. Pure post-processing - no GPU, no
Kaggle - see pipeline/colour.py, pipeline/kinematics.py (both already self-tested).

**Real-world units when telemetry is available.** Each detection's ground-contact point is
projected onto the ground plane via the drone's own .srt telemetry (focal length, altitude,
gimbal angles - see pipeline/geo_reference_telemetry.py), so speed comes out in m/s and km/h,
not px/s. Pixel-space columns are still written alongside, both as a fallback when no
telemetry is found and because they're what the earlier runs were validated against.

Positions are also written as east_m/north_m (metres from the point directly beneath the
drone), which is what the Level 3 aggregate-insight stage consumes - lane volumes, queue
lengths and flow/density all need real distances, not pixels.

Usage: python pipeline/level2_enrich.py <source_clip> <model_dir> [--source-name multiroad]
  <model_dir> must contain detections.csv + meta.json (as written by the Kaggle notebooks).
  Writes detections.csv in place (adding position/kinematics columns) and tracks_summary.csv.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.colour import dominant_colour
from pipeline.geo_reference import ground_contact_point
from pipeline.geo_reference_telemetry import from_telemetry_frame
from pipeline.kinematics import compute_kinematics
from pipeline.srt_parser import parse_srt_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_telemetry(source_name):
    """Locates the .srt sidecar for a source. Returns a path or None - a missing telemetry
    file degrades to pixel-space output rather than failing the whole enrichment."""
    for pattern in (
        os.path.join(REPO_ROOT, "videos", "clips", "compressed", f"{source_name}_full.srt"),
        os.path.join(REPO_ROOT, "videos", "clips", "compressed", f"{source_name}*.srt"),
        os.path.join(REPO_ROOT, "videos", f"*{source_name}*.srt"),
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def build_projector(source_name, model_dir):
    """Builds a ground-plane projector from telemetry, or returns None to stay pixel-space.

    Uses a mid-clip telemetry sample: both sources are stationary hovers, so one sample
    describes the whole clip (confirmed via telemetry - see PLAN.md).
    """
    srt_path = find_telemetry(source_name)
    if not srt_path:
        return None, "no .srt telemetry found"
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None, "no meta.json (need frame dimensions)"
    with open(meta_path) as f:
        meta = json.load(f)
    fw, fh = meta.get("frame_width"), meta.get("frame_height")
    if not fw or not fh:
        return None, "meta.json lacks frame_width/frame_height"
    frames = parse_srt_file(srt_path)
    if not frames:
        return None, f"parsed no telemetry frames from {os.path.basename(srt_path)}"
    try:
        proj = from_telemetry_frame(frames[len(frames) // 2], fw, fh, source_video=source_name)
    except ValueError as e:
        return None, str(e)
    return proj, os.path.basename(srt_path)


def enrich(source_clip: str, model_dir: str, source_name: str = None):
    if source_name is None:
        # Infer from the clip filename (e.g. multiroad_10s.MP4 -> multiroad).
        source_name = os.path.basename(source_clip).split("_")[0]

    csv_path = f"{model_dir}/detections.csv"
    df = pd.read_csv(csv_path)
    # Ground-contact point (bbox bottom-centre), vectorized - the project-wide convention;
    # see pipeline/geo_reference.ground_contact_point for why not the centroid.
    df["gx"] = (df["x1"] + df["x2"]) / 2
    df["gy"] = df["y2"]
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

    proj, telemetry_note = build_projector(source_name, model_dir)
    if proj is not None:
        east, north = proj.pixels_to_ground(df["gx"].to_numpy(), df["gy"].to_numpy())
        df["east_m"], df["north_m"] = east, north
        n_missed = int(np.isnan(east).sum())
        print(f"  geo-referenced via {telemetry_note}: camera {proj.height_m:.1f} m up, "
              f"f_px={proj.f_px:.0f}"
              + (f" ({n_missed} detections' rays missed the ground plane)" if n_missed else ""))
    else:
        df["east_m"], df["north_m"] = np.nan, np.nan
        print(f"  ! pixel-space only ({telemetry_note}) - speeds will be px/s, not m/s")

    # Kinematics per track. Run twice: once in metres (the real answer) and once in pixels
    # (kept for continuity with the earlier validated runs, and as the only output when no
    # telemetry exists). Savgol-smoothed - see pipeline/kinematics.py for why a single-window
    # diff isn't trustworthy for acceleration.
    for col in ("speed_px_s", "accel_px_s2", "speed_mps", "accel_mps2", "speed_kmh"):
        df[col] = np.nan
    for track_id, group in df.groupby("track_id"):
        idx = group.sort_values("time_s").index
        t = df.loc[idx, "time_s"].to_numpy()

        speed, accel = compute_kinematics(
            t, df.loc[idx, "gx"].to_numpy(), df.loc[idx, "gy"].to_numpy())
        if speed is not None:
            df.loc[idx, "speed_px_s"] = speed
            df.loc[idx, "accel_px_s2"] = accel

        if proj is not None:
            e = df.loc[idx, "east_m"].to_numpy()
            n = df.loc[idx, "north_m"].to_numpy()
            if not np.isnan(e).any():
                speed_m, accel_m = compute_kinematics(t, e, n)
                if speed_m is not None:
                    df.loc[idx, "speed_mps"] = speed_m
                    df.loc[idx, "accel_mps2"] = accel_m
                    df.loc[idx, "speed_kmh"] = speed_m * 3.6

    # One representative frame per track (largest bbox = clearest crop) for colour - a single
    # sequential video pass rather than N random seeks, since a short seek-heavy pass on a
    # compressed clip risks landing on the wrong nearby keyframe.
    best_row_per_track = df.loc[df.groupby("track_id")["area"].idxmax()]
    frames_needed = {}  # frame_idx -> [(track_id, x1, y1, x2, y2), ...]
    for _, row in best_row_per_track.iterrows():
        frames_needed.setdefault(int(row["frame"]), []).append(
            (int(row["track_id"]), row["x1"], row["y1"], row["x2"], row["y2"])
        )

    colours = {}
    cap = cv2.VideoCapture(source_clip)
    frame_idx = 0
    while frames_needed and frame_idx <= max(frames_needed):
        ok, frame = cap.read()
        if not ok:
            break
        for track_id, x1, y1, x2, y2 in frames_needed.get(frame_idx, []):
            crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if crop.size == 0:
                continue
            b, g, r = dominant_colour(crop)
            colours[track_id] = f"#{r:02x}{g:02x}{b:02x}"
        frame_idx += 1
    cap.release()

    df.drop(columns=["gx", "gy", "area"]).to_csv(csv_path, index=False)

    def stat(group, col, fn):
        return round(float(fn(group[col])), 2) if group[col].notna().any() else None

    summary_rows = []
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("time_s")
        row = {
            "track_id": track_id,
            "taxonomy_class": group["taxonomy_class"].mode().iat[0],
            "dominant_color_hex": colours.get(track_id, "#898781"),
            "avg_speed_px_s": stat(group, "speed_px_s", np.nanmean),
            "max_speed_px_s": stat(group, "speed_px_s", np.nanmax),
            "avg_speed_kmh": stat(group, "speed_kmh", np.nanmean),
            "max_speed_kmh": stat(group, "speed_kmh", np.nanmax),
            "avg_accel_mps2": stat(group, "accel_mps2", np.nanmean),
            "max_accel_mps2": stat(group, "accel_mps2", np.nanmax),
            "first_seen_s": float(group["time_s"].iat[0]),
            "last_seen_s": float(group["time_s"].iat[-1]),
            "duration_s": round(float(group["time_s"].iat[-1] - group["time_s"].iat[0]), 2),
            "n_detections": len(group),
        }
        # Straight-line ground distance travelled - the basis for Level 3's O-D and lane work.
        if group["east_m"].notna().any():
            e, n = group["east_m"].to_numpy(), group["north_m"].to_numpy()
            row["start_east_m"], row["start_north_m"] = round(float(e[0]), 2), round(float(n[0]), 2)
            row["end_east_m"], row["end_north_m"] = round(float(e[-1]), 2), round(float(n[-1]), 2)
            row["net_displacement_m"] = round(float(np.hypot(e[-1] - e[0], n[-1] - n[0])), 2)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(f"{model_dir}/tracks_summary.csv", index=False)
    n_real = sum(1 for r in summary_rows if r.get("avg_speed_kmh") is not None)
    print(f"  {len(df)} rows, {len(summary_rows)} tracks, {len(colours)} with colour, "
          f"{n_real} with real-world speed (km/h)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_clip", help="the exact clip the detections were computed from")
    parser.add_argument("model_dir", help="folder holding detections.csv + meta.json")
    parser.add_argument("--source-name", default=None,
                        help="scene name for telemetry lookup (default: inferred from clip filename)")
    args = parser.parse_args()
    print(f"{args.model_dir}:")
    enrich(args.source_clip, args.model_dir, args.source_name)
