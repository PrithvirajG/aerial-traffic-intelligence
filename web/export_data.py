"""
Exports every processed combo under videos/samples/ into compact JSON the JS dashboard
(web/index.html) loads directly - plus copies the matching plain source clip so the browser
can play it with a canvas overlay on top.

Frame-indexed, not row-indexed: the dashboard needs "what's visible at frame N" on every
animation tick, so paying once here for an O(1) per-frame lookup beats filtering 60k rows in
the browser 30 times a second.

Coordinates stay in source-video pixel space; the dashboard scales them to whatever size the
video element renders at. Per-track summary values (colour, avg/max speed, avg/max accel)
come from tracks_summary.csv (pipeline/level2_enrich.py) and drive the Level 2 filters.

Usage: python web/export_data.py
Writes: web/data/<source>__<duration>__<model>.json, web/data/index.json, web/clips/*.MP4
"""
import glob
import json
import math
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_ROOT = os.path.join(REPO_ROOT, "videos", "samples")
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WEB_DIR, "data")
CLIPS_DIR = os.path.join(WEB_DIR, "clips")

CLASS_ORDER = ["car", "LGV_candidate", "HGV_candidate", "bus", "motorcycle", "pedestrian", "other"]

# 8-way compass, clockwise from North - index order matters, the dashboard's cyclic direction
# palette maps hue to this index so that opposite headings land opposite on the colour wheel.
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
TURN_TYPES = ["through", "left", "right", "u-turn"]
MIN_TRAVEL_M = 3.0     # below this a track hasn't really gone anywhere: no direction assigned
MIN_TRAVEL_PX = 25.0   # pixel-space equivalent, for combos without telemetry


def compass_index(dx, dy):
    """Bucket a displacement into one of 8 compass sectors. dx=east, dy=north."""
    ang = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0   # 0 = North, clockwise
    return int(((ang + 22.5) % 360.0) // 45.0)


def turn_index(entry, exit_):
    """through / left / right / u-turn from entry and exit heading vectors."""
    n1 = math.hypot(*entry)
    n2 = math.hypot(*exit_)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    e = (entry[0] / n1, entry[1] / n1)
    x = (exit_[0] / n2, exit_[1] / n2)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, e[0] * x[0] + e[1] * x[1]))))
    if ang < 30:
        return 0
    if ang > 150:
        return 3
    return 1 if (e[0] * x[1] - e[1] * x[0]) > 0 else 2


def movement_of(group, use_geo):
    """Returns (compass_index, turn_index) for one track's path.

    Three distinct outcomes, kept distinct on purpose:
      * (None, None)  - too few samples to say anything; guessing from noise would be worse.
      * (-1, -1)      - tracked long enough, but genuinely didn't go anywhere: STATIONARY.
                        This is a real observation, not missing data - on the congested
                        intersection clip it's 72% of tracks, i.e. the queue itself - so it
                        gets its own bucket rather than being dropped from the view.
      * (0-7, 0-3)    - a real heading and turn type.
    """
    if use_geo:
        xs, ys = group["east_m"].to_numpy(), group["north_m"].to_numpy()
        min_travel = MIN_TRAVEL_M
    else:
        # Image-space fallback: y is flipped so "up in frame" reads as the positive axis, but
        # this is NOT a compass bearing - the payload flags it so the UI can say so.
        xs = ((group["x1"] + group["x2"]) / 2).to_numpy()
        ys = (-group["y2"]).to_numpy()
        min_travel = MIN_TRAVEL_PX
    ok = ~(pd.isna(xs) | pd.isna(ys))
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 6:
        return None, None
    net = (xs[-1] - xs[0], ys[-1] - ys[0])
    if math.hypot(*net) < min_travel:
        return -1, -1      # stationary - see docstring

    k = min(4, len(xs) - 1)
    entry = (xs[k] - xs[0], ys[k] - ys[0])
    exit_ = (xs[-1] - xs[-1 - k], ys[-1] - ys[-1 - k])
    return compass_index(*net), turn_index(entry, exit_)

# Where to find the plain (un-annotated) source clip for a given source/duration. The
# compressed set is what the Kaggle runs actually processed, so boxes line up exactly.
CLIP_CANDIDATES = [
    os.path.join(REPO_ROOT, "videos", "clips", "compressed", "{source}_{duration}.MP4"),
    os.path.join(REPO_ROOT, "videos", "clips", "{source}_{duration}.MP4"),
]


def find_clip(source, duration):
    for pattern in CLIP_CANDIDATES:
        path = pattern.format(source=source, duration=duration)
        if os.path.exists(path):
            return path
    return None


def prepare_web_clip(src_path, dst_path):
    """Transcodes a source clip to something a browser can actually decode.

    The source clips are H.264 **High 10** profile (10-bit, yuv420p10le) - confirmed via
    ffprobe. No browser decodes 10-bit H.264; Chrome silently fails to play it. So a plain
    file copy here produces a dashboard with a permanently black video panel. Forcing 8-bit
    `yuv420p` is mandatory, not an optimization.

    Also downscaled to 1080p: 4K decode in a browser tab is needlessly heavy, and overlay
    alignment is unaffected because the overlay maps *source* pixel coordinates onto the
    rendered element by aspect ratio (see projector() in app.js) - resolution never enters it.
    """
    if os.path.exists(dst_path) and os.path.getmtime(dst_path) >= os.path.getmtime(src_path):
        return True
    cmd = (
        f'ffmpeg -y -loglevel error -i "{src_path}" '
        f'-vf scale=1920:-2 -c:v libx264 -profile:v high -pix_fmt yuv420p '
        f'-crf 23 -preset veryfast -an -movflags +faststart "{dst_path}"'
    )
    return os.system(cmd) == 0 and os.path.exists(dst_path)


def export_combo(model_dir):
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    source = meta.get("source_video", "unknown")
    duration = f"{meta.get('duration_s', '?')}s"
    model = meta["model"]
    key = f"{source}__{duration}__{model}"

    df = pd.read_csv(os.path.join(model_dir, "detections.csv"))
    class_idx = {c: i for i, c in enumerate(CLASS_ORDER)}

    # frames: {frame_number: [[track_id, x1, y1, x2, y2], ...]} - ints only, no per-row keys,
    # which keeps the payload a fraction of what a list-of-objects export would be.
    frames = {}
    for frame, group in df.groupby("frame"):
        frames[int(frame)] = [
            [int(r.track_id), int(r.x1), int(r.y1), int(r.x2), int(r.y2)]
            for r in group.itertuples()
        ]

    # Per-track direction of travel. Uses real ground coordinates when telemetry-derived
    # columns exist (so a bucket really means a compass bearing); otherwise falls back to
    # image space, which is still a consistent direction but NOT a bearing - flagged in the
    # payload as dirFrame so the UI labels it honestly instead of implying north.
    use_geo = "east_m" in df.columns and df["east_m"].notna().any()
    movements = {}
    for track_id, group in df.groupby("track_id"):
        movements[int(track_id)] = movement_of(group.sort_values("time_s"), use_geo)

    summary_path = os.path.join(model_dir, "tracks_summary.csv")
    tracks = {}
    if os.path.exists(summary_path):
        sdf = pd.read_csv(summary_path)
        # Per-track accel isn't in tracks_summary.csv, so derive it here from the enriched
        # detections rather than re-running enrichment.
        accel = df.groupby("track_id")["accel_px_s2"].agg(["mean", "max"]) \
            if "accel_px_s2" in df.columns else None
        for r in sdf.itertuples():
            entry = {
                "c": class_idx.get(r.taxonomy_class, class_idx["other"]),
                "col": r.dominant_color_hex,
                "sp": None if pd.isna(r.avg_speed_px_s) else round(float(r.avg_speed_px_s), 1),
                "spx": None if pd.isna(r.max_speed_px_s) else round(float(r.max_speed_px_s), 1),
                "dur": float(r.duration_s),
                "n": int(r.n_detections),
            }
            # Real-world speed, present once level2_enrich has telemetry to work with.
            for src_col, key_out in (("avg_speed_kmh", "kmh"), ("max_speed_kmh", "kmhx")):
                v = getattr(r, src_col, None)
                entry[key_out] = None if v is None or pd.isna(v) else round(float(v), 1)
            if accel is not None and r.track_id in accel.index:
                a = accel.loc[r.track_id]
                entry["ac"] = None if pd.isna(a["mean"]) else round(float(a["mean"]), 1)
                entry["acx"] = None if pd.isna(a["max"]) else round(float(a["max"]), 1)
            entry["dir"], entry["turn"] = movements.get(int(r.track_id), (None, None))
            tracks[str(int(r.track_id))] = entry
    else:
        # No Level 2 enrichment for this combo - class only, so class filters still work and
        # the Level 2 filters simply have nothing to filter on (reported in the UI).
        for track_id, group in df.groupby("track_id"):
            d, t = movements.get(int(track_id), (None, None))
            tracks[str(int(track_id))] = {
                "c": class_idx.get(group["taxonomy_class"].mode().iat[0], class_idx["other"]),
                "col": "#898781", "sp": None, "spx": None,
                "dur": float(group["time_s"].max() - group["time_s"].min()),
                "n": len(group), "dir": d, "turn": t,
            }

    clip_src = find_clip(source, duration)
    clip_name = None
    if clip_src:
        clip_name = f"{source}_{duration}_web.mp4"
        if not prepare_web_clip(clip_src, os.path.join(CLIPS_DIR, clip_name)):
            print(f"    ! transcode failed for {clip_src}")
            clip_name = None

    # Level 3 aggregate insight, if pipeline/level3_aggregate.py has been run for this combo.
    agg_path = os.path.join(model_dir, "aggregate_insight.json")
    aggregate = None
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            aggregate = json.load(f)

    # Level 4 map layers (road network + map-matched trajectories, desire lines, queues),
    # if pipeline/map_export.py has been run. Pre-projected to local ENU metres so the
    # dashboard can draw real road geometry as vectors with no tile server.
    map_path = os.path.join(model_dir, "map_layers.json")
    map_layers = None
    if os.path.exists(map_path):
        with open(map_path) as f:
            map_layers = json.load(f)

    # Frame-indexed ground positions, so the map panel can animate in lockstep with the video
    # exactly like the flow panel does. Stored as decimetres (ints) rather than floats: 0.1 m
    # precision is far finer than the ~1.3 m registration residual, and it roughly halves the
    # payload versus full-precision floats.
    map_frames = {}
    mm_path = os.path.join(model_dir, "map_matched.csv")
    if os.path.exists(mm_path):
        mm = pd.read_csv(mm_path)
        mm = mm[mm["map_east_m"].notna() & mm["map_north_m"].notna()]
        for frame, grp in mm.groupby("frame"):
            map_frames[int(frame)] = [
                [int(r.track_id), int(round(r.map_east_m * 10)), int(round(r.map_north_m * 10)),
                 int(r.lane) if pd.notna(r.lane) else 0]
                for r in grp.itertuples()
            ]

    payload = {
        "key": key, "source": source, "duration": duration, "model": model,
        "meta": meta, "classOrder": CLASS_ORDER, "clip": clip_name,
        "frameWidth": meta.get("frame_width"), "frameHeight": meta.get("frame_height"),
        "fps": meta.get("fps"), "frames": frames, "tracks": tracks,
        "aggregate": aggregate, "mapLayers": map_layers,
        "mapFrames": map_frames or None,
        "compass": COMPASS, "turnTypes": TURN_TYPES,
        "dirFrame": "compass" if use_geo else "image",
    }
    out_path = os.path.join(DATA_DIR, f"{key}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  {key}: {len(df)} rows, {len(tracks)} tracks, {len(frames)} frames "
          f"-> {size_mb:.1f} MB" + ("" if clip_name else "  [no source clip found!]"))
    return {
        "key": key, "source": source, "duration": duration, "model": model,
        "tracker": meta.get("tracker"), "uniqueTracks": len(tracks),
        "hasLevel2": os.path.exists(summary_path), "clip": clip_name,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    meta_paths = sorted(glob.glob(os.path.join(SAMPLES_ROOT, "*", "*", "*", "meta.json")))
    if not meta_paths:
        raise SystemExit(f"No processed combos found under {SAMPLES_ROOT}")

    print(f"Exporting {len(meta_paths)} combo(s):")
    index = [export_combo(os.path.dirname(p)) for p in meta_paths]
    with open(os.path.join(DATA_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote {DATA_DIR}/index.json — start the dashboard with: python web/serve.py")


if __name__ == "__main__":
    main()
