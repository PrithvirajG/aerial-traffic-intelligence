"""
Refines the camera model itself against the road network, instead of rigidly shifting its
output.

Why a rigid transform isn't enough. On the near-nadir `intersection` scene (-63.1 deg) a pure
translation registers trajectories to the carriageways beautifully (median residual 1.11 m).
On the shallow `multiroad` scene (-18.5 deg) the best rigid fit still leaves 32 m of error,
and the residual grows monotonically with range - 6 m near-field, 95 m beyond 450 m. That
signature is a *projection* error, not a placement error, and no translation can fix it.

The cause is geometric and measurable: at -18.5 deg the optical axis meets the ground 211 m
out, where d(range)/d(pitch) is 12.2 m per degree. Gimbal pitch is reported to 0.1 deg but its
absolute accuracy is not that good, and the error is amplified enormously downrange. Altitude
enters differently - range scales linearly with height - so with observations spread over a
wide range band the two are separable rather than degenerate.

So this module fits the physical parameters (pitch, yaw, altitude scale) plus a residual GPS
translation, re-projecting from the original pixels each time, and reports the range band over
which the result is actually trustworthy for lane-level work.

TWO KNOWN LIMITATIONS, both measured, neither yet fixed - do not read the reported residual as
the whole truth:

1. The objective is blind to cross-road error. It is the median distance to the NEAREST drivable
   link, and a divided road is two parallel OSM ways ~15 m apart. Shifting every point sideways
   by a lane simply re-assigns them to the other carriageway, so the cost barely moves: the cost
   surface is almost flat across the road and a lateral or yaw error survives a fit that looks
   converged. `min_dist_to_segments_directed` measures the distance to the carriageway a vehicle
   is legally allowed to be on given its heading, and is now reported alongside as a diagnostic.
   Scoring the SEARCH with it was tried and measured head to head on identical inputs; it did not
   win (plain 3.76 m vs directional 4.48 m median on multiroad/10s) and both runs converged to
   the edge of the parameter grid, so the fit still uses the plain objective.

2. The stored multiroad refit does not reproduce on current inputs. projection_refit.json records
   median 1.44 m over 684 points; the same code on today's detections.csv fits 2318 points and
   reaches ~4.2 m, and widening the parameter grid does not recover it. The most likely reading
   is that 1.44 m was measured over a smaller, easier subset (a well-behaved mid-range corridor)
   and therefore OVERSTATES accuracy - not that the projection got worse. Independent checks that
   do not depend on this objective still hold: pedestrians measure 5.8 km/h, and vehicle points
   sit a median 2.29 m from the nearest carriageway centreline (a lane is 3.5 m wide).

Usage: python pipeline/refine_projection.py <model_dir> [--source-name multiroad]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.geo_reference_telemetry import TelemetryGroundProjector
from pipeline.register_to_map import (min_dist_to_segments, min_dist_to_segments_directed,
                                      segment_arrays, segment_arrays_directed)
from pipeline.road_network import load_network
from pipeline.srt_parser import parse_srt_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIT_POINTS = 2500
LANE_W = 3.5


def telemetry_for(source_name):
    for pat in (f"videos/clips/compressed/{source_name}_full.srt",
                f"videos/clips/compressed/{source_name}*.srt",
                f"videos/*{source_name}*.srt"):
        hits = sorted(glob.glob(os.path.join(REPO_ROOT, pat)))
        if hits:
            frames = parse_srt_file(hits[0])
            if frames:
                return frames[len(frames) // 2], os.path.basename(hits[0])
    raise SystemExit(f"no telemetry found for {source_name}")


def project(px, py, tf, fw, fh, d_pitch, d_yaw, alt_scale, dx, dy):
    """Re-project pixels with perturbed camera parameters, then apply a residual shift."""
    proj = TelemetryGroundProjector(
        fw, fh, 24.0, tf.rel_altitude * alt_scale,
        tf.gimbal_pitch + d_pitch, (tf.gimbal_yaw or 0.0) + d_yaw, tf.gimbal_roll or 0.0,
    )
    e, n = proj.pixels_to_ground(px, py)
    return e + dx, n + dy


def headings(e, n, prev_i, next_i, valid):
    """Ground heading at each sample, from its neighbours in the SAME track.

    Recomputed for every candidate parameter set, because the whole point is to score a
    candidate on where its projected vehicles are going, not just where they are.
    """
    he = np.zeros_like(e)
    hn = np.zeros_like(n)
    he[valid] = e[next_i[valid]] - e[prev_i[valid]]
    hn[valid] = n[next_i[valid]] - n[prev_i[valid]]
    return he, hn


def score(px, py, tf, fw, fh, segs, params, max_range, track_idx=None):
    e, n = project(px, py, tf, fw, fh, *params)
    ok = np.isfinite(e) & np.isfinite(n)
    if max_range:
        ok &= (np.hypot(e, n) <= max_range)
    if ok.sum() < 200:
        return float("inf"), 0
    if track_idx is None:
        d = min_dist_to_segments(e[ok], n[ok], segs)
        return float(np.median(d)), int(ok.sum())
    # Direction-aware: heading is computed on the FULL arrays (so a point's neighbours are its
    # real track neighbours) and only then masked, otherwise `ok` would silently make unrelated
    # samples adjacent.
    prev_i, next_i, valid = track_idx
    he, hn = headings(e, n, prev_i, next_i, valid & ok)
    cost, _ = min_dist_to_segments_directed(e[ok], n[ok], he[ok], hn[ok], segs)
    return float(np.median(cost)), int(ok.sum())


def fit(px, py, tf, fw, fh, segs, max_range, verbose=True, track_idx=None):
    """Coarse-to-fine search over (d_pitch, d_yaw, alt_scale, dx, dy)."""
    def score_(cand):
        return score(px, py, tf, fw, fh, segs, cand, max_range, track_idx)

    best = [0.0, 0.0, 1.0, 0.0, 0.0]
    best_score, _ = score_(best)
    if verbose:
        print(f"  start: median {best_score:.2f} m")

    schedule = [
        {"pitch": np.arange(-3, 3.01, 0.5), "yaw": np.arange(-4, 4.01, 1.0),
         "alt": np.arange(0.85, 1.16, 0.05), "xy": np.arange(-12, 12.1, 4.0)},
        {"pitch": np.arange(-0.8, 0.81, 0.2), "yaw": np.arange(-1.2, 1.21, 0.3),
         "alt": np.arange(0.96, 1.045, 0.02), "xy": np.arange(-4, 4.1, 1.0)},
        {"pitch": np.arange(-0.2, 0.21, 0.05), "yaw": np.arange(-0.3, 0.31, 0.1),
         "alt": np.arange(0.99, 1.011, 0.005), "xy": np.arange(-1.2, 1.21, 0.3)},
    ]
    for st in schedule:
        # Coordinate descent over the physical parameters, then the residual shift. Each
        # parameter is searched against the current best of the others rather than as a full
        # 5-D grid, which would be ~10^6 projections per stage.
        for _ in range(2):
            for idx, grid in ((0, st["pitch"]), (1, st["yaw"]), (2, st["alt"])):
                base = best[idx]
                for v in grid:
                    cand = list(best)
                    cand[idx] = base + v if idx != 2 else base * v
                    s, _ = score_(cand)
                    if s < best_score:
                        best_score, best = s, cand
            bx, by = best[3], best[4]
            for x in st["xy"]:
                for y in st["xy"]:
                    cand = list(best)
                    cand[3], cand[4] = bx + x, by + y
                    s, _ = score_(cand)
                    if s < best_score:
                        best_score, best = s, cand
        if verbose:
            print(f"  stage: pitch{best[0]:+.2f} yaw{best[1]:+.2f} alt x{best[2]:.3f} "
                  f"shift({best[3]:+.1f},{best[4]:+.1f}) -> median {best_score:.2f} m")
    return best, best_score


def analyse(model_dir, source_name=None, max_range=None):
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    source_name = source_name or meta["source_video"]
    fw, fh = meta["frame_width"], meta["frame_height"]
    tf, srt_name = telemetry_for(source_name)

    df = pd.read_csv(os.path.join(model_dir, "detections.csv"))
    if "speed_kmh" in df.columns:
        df = df[df["speed_kmh"] > 8.0]
    # Sample whole TRACKS, not random rows. The fit now scores direction of travel, and a
    # randomly sampled row has no neighbour to difference against - shuffling the rows would
    # destroy exactly the signal that separates the two halves of a divided carriageway.
    df = df.sort_values(["track_id", "frame"], kind="stable")
    # Thin by a uniform stride rather than by dropping tracks. Keeping only some tracks whole
    # wrecked spatial coverage (25 tracks, two thirds of them in one range band, and the fit
    # degraded from 1.4 m to 3.4 m). Striding keeps every track and every part of the scene,
    # and a longer baseline between neighbours actually makes the heading less noisy.
    if len(df) > FIT_POINTS:
        df = df.iloc[:: max(1, math.ceil(len(df) / FIT_POINTS))]
    px = ((df["x1"] + df["x2"]) / 2).to_numpy()
    py = df["y2"].to_numpy()

    # Neighbour indices for the heading difference, valid only strictly inside a track.
    tid = df["track_id"].to_numpy()
    idx = np.arange(len(tid))
    prev_i = np.clip(idx - 1, 0, len(tid) - 1)
    next_i = np.clip(idx + 1, 0, len(tid) - 1)
    valid = (tid[prev_i] == tid) & (tid[next_i] == tid) & (idx > 0) & (idx < len(tid) - 1)
    track_idx = (prev_i, next_i, valid)

    links, _ = load_network()
    segs = segment_arrays(links)
    dsegs = segment_arrays_directed(links)
    print(f"{os.path.relpath(model_dir, REPO_ROOT)}  ({len(px)} points from "
          f"{df['track_id'].nunique()} tracks, {valid.sum()} with heading, "
          f"telemetry {srt_name}, pitch {tf.gimbal_pitch} deg)")
    # NOTE: the fit still uses the plain nearest-link objective. Scoring the SEARCH with the
    # direction-aware cost was tried and measured head to head on identical inputs, and it did
    # not win (plain 3.76 m vs directional 4.48 m median on multiroad/10s) - both converged to
    # the edge of the parameter grid, which says the search itself has a problem that direction
    # does not fix. The directional residual is therefore reported as a DIAGNOSTIC below rather
    # than driving the fit; see the note printed after the range bands.
    params, sc = fit(px, py, tf, fw, fh, segs, max_range)

    e, n = project(px, py, tf, fw, fh, *params)
    ok = np.isfinite(e) & np.isfinite(n)
    if max_range:
        ok &= np.hypot(e, n) <= max_range
    resid = min_dist_to_segments(e[ok], n[ok], segs)
    rng = np.hypot(e[ok], n[ok])

    # Reported separately, because plain nearest-segment residual is exactly the metric that
    # cannot see a cross-road error: it just re-assigns the point to the other carriageway.
    # This one measures the distance to the carriageway the vehicle is legally allowed to be
    # on given its direction, which is what "correct lane, correct direction" actually needs.
    he, hn = headings(e, n, prev_i, next_i, valid & ok)
    _, resid_dir = min_dist_to_segments_directed(e[ok], n[ok], he[ok], hn[ok], dsegs)
    hm = (np.hypot(he[ok], hn[ok]) > 1e-9)
    dir_med = float(np.median(resid_dir[hm])) if hm.any() else float("nan")
    dir_lane = float((resid_dir[hm] < LANE_W).mean()) if hm.any() else float("nan")
    print(f"  direction-consistent residual: median {dir_med:.2f} m, "
          f"{dir_lane * 100:.0f}% within a lane  (plain nearest-link "
          f"{np.median(resid):.2f} m - blind to cross-road error)")

    print(f"\n  residual by range band (lane width {LANE_W} m):")
    bands, reliable_to = [], 0.0
    for lo, hi in [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600), (600, 1e9)]:
        m = (rng >= lo) & (rng < hi)
        if m.sum() < 30:
            continue
        med = float(np.median(resid[m]))
        within = float((resid[m] < LANE_W).mean())
        bands.append({"from_m": lo, "to_m": None if hi > 1e8 else hi, "n": int(m.sum()),
                      "median_resid_m": round(med, 2), "frac_within_lane": round(within, 3)})
        flag = "  <- lane-capable" if med < LANE_W / 2 else ("  <- degraded" if med > LANE_W else "")
        print(f"    {lo:4d}-{'inf' if hi > 1e8 else hi:>4} m  n={m.sum():5d}  "
              f"median {med:6.2f} m  within-lane {within * 100:3.0f}%{flag}")
        if med < LANE_W / 2:
            reliable_to = max(reliable_to, hi if hi < 1e8 else rng.max())

    out = {
        "source_video": source_name, "model": meta.get("model"),
        "gimbal_pitch_deg": tf.gimbal_pitch,
        "fitted": {"d_pitch_deg": round(params[0], 3), "d_yaw_deg": round(params[1], 3),
                   "alt_scale": round(params[2], 4),
                   "dx_m": round(params[3], 2), "dy_m": round(params[4], 2)},
        "median_resid_m": round(float(np.median(resid)), 2),
        "median_resid_directed_m": round(dir_med, 2),
        "frac_within_lane_directed": round(dir_lane, 3),
        "p90_resid_m": round(float(np.percentile(resid, 90)), 2),
        "frac_within_lane": round(float((resid < LANE_W).mean()), 3),
        "lane_reliable_to_m": round(reliable_to, 0),
        "range_bands": bands,
        "n_points": int(ok.sum()),
    }
    with open(os.path.join(model_dir, "projection_refit.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  => overall median {out['median_resid_m']} m, "
          f"{out['frac_within_lane'] * 100:.0f}% within a lane; "
          f"lane-capable out to ~{out['lane_reliable_to_m']:.0f} m")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--source-name", default=None)
    ap.add_argument("--max-range", type=float, default=None,
                    help="ignore points beyond this range when fitting (far field is "
                         "dominated by projection error and can dominate the objective)")
    a = ap.parse_args()
    analyse(a.model_dir, a.source_name, a.max_range)
