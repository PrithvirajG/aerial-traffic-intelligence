"""
Draws per-track trajectory trails onto a clean source clip, using detections already
computed by a Kaggle tracking run (kaggle/tracked_demo/) - pure local post-processing, no
model inference, no Kaggle involved. This is only meaningful once tracking is stable (see
LEVEL1_DETECTION_TRACKING.md's BoT-SORT results) - drawn against the earlier ByteTrack@0.15
output, trails would mostly show thousands of few-frame fragments, not real paths.

Trail point = bounding-box bottom-centre (ground-contact point), not the box centroid - same
convention PLAN.md specifies for the upcoming geo-referencing step, so trajectory data stays
consistent between this visualization and the real kinematics work later.

Usage: python pipeline/trajectory_trails.py <source_video> <detections_csv> <output_video>
"""
import argparse
import colorsys
import csv
from collections import defaultdict

import cv2

TRAIL_LENGTH_FRAMES = 45  # ~1.5s at 30fps - long enough to see real motion, short enough
                          # that trails don't turn into an unreadable tangle in dense traffic
POINT_RADIUS = 3
LINE_THICKNESS = 2


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR color per track ID (golden-ratio hue spacing)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def load_detections(csv_path: str) -> dict[int, list[tuple[int, int, float, float]]]:
    """Returns {track_id: [(frame, class_label_unused, cx, cy_bottom), ...]} sorted by frame."""
    by_track = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            cx = (x1 + x2) / 2
            cy_bottom = y2  # ground-contact point, not centroid - see module docstring
            by_track[track_id].append((frame, cx, cy_bottom))
    for track_id in by_track:
        by_track[track_id].sort(key=lambda r: r[0])
    return by_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_video")
    parser.add_argument("detections_csv")
    parser.add_argument("output_video")
    parser.add_argument("--trail-length", type=int, default=TRAIL_LENGTH_FRAMES)
    args = parser.parse_args()

    by_track = load_detections(args.detections_csv)

    # Index points by frame for fast per-frame lookup, keeping track_id attached.
    points_by_frame = defaultdict(list)
    for track_id, points in by_track.items():
        for frame, cx, cy in points:
            points_by_frame[frame].append((track_id, cx, cy))

    cap = cv2.VideoCapture(args.source_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Running history per track, capped to the trail window - avoids re-scanning all
    # detections every frame for a clip of any real length.
    history = defaultdict(list)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for track_id, cx, cy in points_by_frame.get(frame_idx, []):
            history[track_id].append((cx, cy))
            if len(history[track_id]) > args.trail_length:
                history[track_id].pop(0)

        for track_id, pts in history.items():
            if len(pts) < 2:
                continue
            color = track_color(track_id)
            for i in range(1, len(pts)):
                # Fade older segments toward the background rather than full-brightness -
                # keeps recent motion visually dominant in dense-traffic frames.
                alpha = i / len(pts)
                faded = tuple(int(c * alpha) for c in color)
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, faded, LINE_THICKNESS)
            cx, cy = pts[-1]
            cv2.circle(frame, (int(cx), int(cy)), POINT_RADIUS, color, -1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{n_frames}")

    cap.release()
    writer.release()
    print(f"Done: {frame_idx} frames, {len(by_track)} tracks -> {args.output_video}")


if __name__ == "__main__":
    main()
