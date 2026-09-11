"""
Registers projected trajectories to the OSM road network, correcting the absolute-position
bias the telemetry cannot remove on its own.

Why this step exists at all. The camera model (geo_reference_telemetry.py) gives geometry that
is *internally* accurate - relative distances within the scene are right, which is why speeds
validate against walking pace. But its absolute placement on the Earth inherits two biases it
cannot see:
  * drone GPS absolute accuracy, typically 2-5 m even when the relative track is stable;
  * gimbal yaw bias - a 2 deg heading error is a 5 m displacement 150 m from the camera.
Lane width is ~3.5 m. Either bias alone is enough to put every vehicle confidently in the
wrong lane, so "correct lane" is meaningless until this is corrected.

The fix: vehicles drive on roads. So the correct rigid transform is the one that best puts
the observed trajectory cloud onto the mapped carriageways. We search (dx, dy, dtheta) to
minimise the MEDIAN distance from trajectory points to the nearest drivable centreline -
median, not mean, because a minority of tracks legitimately sit off-centreline (parking bays,
forecourts, the service road) and must not drag the fit.

The residual after fitting is reported, not hidden: it is the honest upper bound on how much
lane-level conclusions can be trusted.

Usage: python pipeline/register_to_map.py [model_dir]
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
from pipeline.road_network import load_network

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LINK_DIST_M = 400.0    # only links plausibly in frame
SAMPLE_POINTS = 4000       # enough for a stable median, small enough to grid-search fast


def segment_arrays(links, max_dist_m=MAX_LINK_DIST_M):
    """Flattens drivable link geometry into vectorised segment endpoint arrays."""
    e1, n1, e2, n2 = [], [], [], []
    for L in links:
        if not L["drivable"]:
            continue
        if min(math.hypot(e, n) for e, n in L["points"]) > max_dist_m:
            continue
        for s in L["segments"]:
            e1.append(s["e1"]); n1.append(s["n1"]); e2.append(s["e2"]); n2.append(s["n2"])
    return (np.array(e1), np.array(n1), np.array(e2), np.array(n2))


def segment_arrays_directed(links, max_dist_m=MAX_LINK_DIST_M):
    """As segment_arrays, plus each segment's unit direction and whether it is one-way.

    Needed to break a degeneracy that plain nearest-segment distance cannot see. A divided road
    is two parallel OSM ways ~15 m apart, so shifting every point sideways by a lane simply
    re-assigns them to the other carriageway and the nearest-segment cost barely moves. The cost
    surface is therefore almost flat across the road, and a lateral or yaw error can survive a
    fit that looks converged. Direction removes the symmetry: the two carriageways carry traffic
    in opposite directions, so a westbound vehicle only "fits" the westbound one.
    """
    e1, n1, e2, n2, ue, un, ow = [], [], [], [], [], [], []
    for L in links:
        if not L["drivable"]:
            continue
        if min(math.hypot(e, n) for e, n in L["points"]) > max_dist_m:
            continue
        for s in L["segments"]:
            dx, dy = s["e2"] - s["e1"], s["n2"] - s["n1"]
            ln = math.hypot(dx, dy) or 1e-12
            e1.append(s["e1"]); n1.append(s["n1"]); e2.append(s["e2"]); n2.append(s["n2"])
            ue.append(dx / ln); un.append(dy / ln); ow.append(bool(L["oneway"]))
    return (np.array(e1), np.array(n1), np.array(e2), np.array(n2),
            np.array(ue), np.array(un), np.array(ow))


DIR_COS_TOL = 0.5          # 60 deg: generous, so a lane change or a bend still counts as "along"
DIR_MISS_M = 60.0          # distance charged when no direction-consistent segment is in reach


def min_dist_to_segments_directed(pe, pn, he, hn, segs, chunk=256,
                                  cos_tol=DIR_COS_TOL, miss_m=DIR_MISS_M):
    """Distance to the nearest segment the point is allowed to be travelling on.

    Direction is applied as a hard MASK, not as an additive penalty, so the returned value stays
    a real distance in metres. An additive penalty was tried first and distorted the objective:
    with a penalty on every segment the optimiser found it cheaper to move points onto distant
    but perfectly-aligned geometry, and the fit ran to the edge of its parameter grid.

    A segment is admissible if the point's heading agrees with it - within cos_tol of the
    segment axis for a two-way link, and within cos_tol of the arrow for a one-way link, which
    is what separates the two halves of a divided road. Points with no heading fall back to
    plain nearest-segment distance. Points with no admissible segment in reach are charged
    miss_m, so "there is nothing here you could legally be doing" is expensive but bounded.
    """
    e1, n1, e2, n2, ue, un, ow = segs
    dx, dy = e2 - e1, n2 - n1
    denom = dx * dx + dy * dy
    denom[denom == 0] = 1e-12
    out = np.empty(len(pe))
    hlen = np.hypot(he, hn)
    has_h = hlen > 1e-9
    safe = np.where(has_h, hlen, 1.0)
    hx, hy = he / safe, hn / safe
    for i in range(0, len(pe), chunk):
        sl = slice(i, i + chunk)
        px, py = pe[sl, None], pn[sl, None]
        t = np.clip(((px - e1) * dx + (py - n1) * dy) / denom, 0.0, 1.0)
        d = np.hypot(px - (e1 + t * dx), py - (n1 + t * dy))

        dot = hx[sl, None] * ue + hy[sl, None] * un
        okdir = np.where(ow, dot >= cos_tol, np.abs(dot) >= cos_tol)
        okdir |= ~has_h[sl, None]                    # no heading -> every segment admissible
        out[sl] = np.min(np.where(okdir, d, np.inf), axis=1)
    return np.minimum(out, miss_m), out


def min_dist_to_segments(pe, pn, segs, chunk=256):
    """Distance from each point to the nearest segment, vectorised.

    Chunked over points so the (points x segments) intermediate never blows up memory -
    with ~4000 points and ~3000 segments a single array would be 12M floats per axis.
    """
    e1, n1, e2, n2 = segs
    dx, dy = e2 - e1, n2 - n1
    denom = dx * dx + dy * dy
    denom[denom == 0] = 1e-12
    out = np.empty(len(pe))
    for i in range(0, len(pe), chunk):
        px = pe[i:i + chunk, None]
        py = pn[i:i + chunk, None]
        t = ((px - e1) * dx + (py - n1) * dy) / denom
        t = np.clip(t, 0.0, 1.0)          # clamp to the segment, not the infinite line
        cx = e1 + t * dx
        cy = n1 + t * dy
        out[i:i + chunk] = np.min(np.hypot(px - cx, py - cy), axis=1)
    return out


def apply_transform(pe, pn, dx, dy, dtheta_deg):
    """Rotate about the origin (the drone's ground point, where yaw error pivots), then
    translate."""
    th = math.radians(dtheta_deg)
    c, s = math.cos(th), math.sin(th)
    return pe * c - pn * s + dx, pe * s + pn * c + dy


def fit_transform(pe, pn, segs, verbose=True):
    """Coarse-to-fine search for the rigid transform minimising median centreline distance."""
    best = (0.0, 0.0, 0.0)
    best_score = float(np.median(min_dist_to_segments(pe, pn, segs)))
    if verbose:
        print(f"  before registration: median {best_score:.2f} m to nearest carriageway")

    # Search bounds sized to the physical error sources: GPS a few metres, yaw a few degrees.
    stages = [
        (np.arange(-15, 15.1, 3.0), np.arange(-15, 15.1, 3.0), np.arange(-5, 5.1, 1.0)),
        (np.arange(-4, 4.1, 1.0), np.arange(-4, 4.1, 1.0), np.arange(-1.5, 1.6, 0.5)),
        (np.arange(-1.2, 1.21, 0.3), np.arange(-1.2, 1.21, 0.3), np.arange(-0.4, 0.41, 0.2)),
    ]
    for xs, ys, ths in stages:
        bx, by, bt = best
        for th in ths:
            for x in xs:
                for y in ys:
                    cx, cy, ct = bx + x, by + y, bt + th
                    te, tn = apply_transform(pe, pn, cx, cy, ct)
                    score = float(np.median(min_dist_to_segments(te, tn, segs)))
                    if score < best_score:
                        best_score, best = score, (cx, cy, ct)
        if verbose:
            print(f"  stage -> dx={best[0]:+.2f} dy={best[1]:+.2f} dtheta={best[2]:+.2f} deg "
                  f"| median {best_score:.2f} m")
    return best, best_score


def collect_points(model_dir, moving_only=True, max_points=SAMPLE_POINTS):
    """Trajectory points to register with. Moving vehicles only by default: a parked car sits
    off the carriageway by design and would bias the fit toward the kerb."""
    df = pd.read_csv(os.path.join(model_dir, "detections.csv"))
    if "east_m" not in df.columns:
        return None, "no east_m/north_m - run pipeline/level2_enrich.py first"
    df = df[df["east_m"].notna() & df["north_m"].notna()]
    if moving_only and "speed_kmh" in df.columns:
        df = df[df["speed_kmh"] > 8.0]
    if df.empty:
        return None, "no usable points"
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    return (df["east_m"].to_numpy(), df["north_m"].to_numpy()), None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", nargs="?", default=None)
    args = ap.parse_args()

    links, meta = load_network()
    segs = segment_arrays(links)
    print(f"road network: {len(segs[0])} drivable segments within {MAX_LINK_DIST_M:.0f} m\n")

    dirs = [args.model_dir] if args.model_dir else sorted(
        glob.glob(os.path.join(REPO_ROOT, "videos", "samples", "*", "*", "*")))
    results = {}
    for d in dirs:
        if not os.path.exists(os.path.join(d, "detections.csv")):
            continue
        name = os.path.relpath(d, os.path.join(REPO_ROOT, "videos", "samples"))
        pts, err = collect_points(d)
        if err:
            print(f"{name}: skipped - {err}")
            continue
        pe, pn = pts
        print(f"{name}  ({len(pe)} moving-vehicle points)")
        (dx, dy, dth), score = fit_transform(pe, pn, segs)

        te, tn = apply_transform(pe, pn, dx, dy, dth)
        resid = min_dist_to_segments(te, tn, segs)
        stats = {
            "dx_m": round(dx, 2), "dy_m": round(dy, 2), "dtheta_deg": round(dth, 2),
            "shift_m": round(math.hypot(dx, dy), 2),
            "median_resid_m": round(float(np.median(resid)), 2),
            "p90_resid_m": round(float(np.percentile(resid, 90)), 2),
            "frac_within_half_lane": round(float((resid < 1.75).mean()), 3),
            "frac_within_lane": round(float((resid < 3.5).mean()), 3),
            "n_points": int(len(pe)),
        }
        print(f"  => shift {stats['shift_m']} m, rotate {stats['dtheta_deg']} deg | "
              f"residual median {stats['median_resid_m']} m, p90 {stats['p90_resid_m']} m | "
              f"{stats['frac_within_lane'] * 100:.0f}% within one lane width\n")
        results[name] = stats
        with open(os.path.join(d, "map_registration.json"), "w") as f:
            json.dump(stats, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
