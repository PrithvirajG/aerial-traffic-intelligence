"""
Binds every trajectory to the road network: correct link, approach, direction, lane, moment
to moment (Level 4, stages D and E).

Runs on the refined camera model from refine_projection.py, so positions are already
registered to the mapped carriageways (~1.3 m median residual).

**Why nearest-link matching is not enough here.** Gopal Hari Deshmukh Marg is a dual
carriageway - two parallel one-way ways only ~10-15 m apart. Their centrelines are closer
together than the registration residual is large, so distance alone cannot say which
carriageway a vehicle is on. What *can* say is the vehicle's heading: each way is tagged
`oneway=yes`, so travelling against a way's digitisation direction is strong evidence you are
on the other carriageway. Matching therefore scores:

    distance  +  heading disagreement with the road  +  a hard penalty for illegal direction

and then smooths over time, because a track cannot legally jump between links frame to frame.
Without the temporal term, a vehicle straddling the median flickers between carriageways.

**Lane binding.** OSM gives a carriageway centreline plus a `lanes` count, not per-lane
geometry. Lane index therefore comes from the signed perpendicular offset from that
centreline, divided by lane width. Lane 1 is the leftmost in the direction of travel (India
drives on the left, so lane 1 is the near-side/slow lane).

**Range gating is mandatory, not decorative.** Registration residual grows with distance from
the drone (measured: ~1.2 m at 100-200 m, ~2.5 m at 300-400 m against a 3.5 m lane). Beyond
the per-scene `lane_reliable_to_m` from projection_refit.json, lane index is reported as None
rather than guessed - a wrong lane is worse than an honest gap.

Usage: python pipeline/map_match.py <model_dir>
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.geodesy import enu_to_wgs84
from pipeline.refine_projection import project, telemetry_for
from pipeline.road_network import load_network

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANE_W = 3.5
MAX_MATCH_DIST_M = 25.0     # beyond this, no link is a plausible explanation
HEADING_WEIGHT = 0.12       # metres of penalty per degree of heading disagreement
WRONG_WAY_PENALTY = 40.0    # metres-equivalent; dominates unless distance strongly disagrees
SWITCH_PENALTY = 6.0        # metres-equivalent cost of changing link between frames
MIN_SPEED_FOR_HEADING = 3.0  # km/h - below this a heading is noise, so don't trust it


def flatten_segments(links, max_dist_m=450.0):
    """Segment endpoint arrays plus the link index each segment belongs to."""
    e1, n1, e2, n2, brg, owner = [], [], [], [], [], []
    keep = []
    for L in links:
        if not L["drivable"]:
            continue
        if min(math.hypot(e, n) for e, n in L["points"]) > max_dist_m:
            continue
        idx = len(keep)
        keep.append(L)
        for s in L["segments"]:
            e1.append(s["e1"]); n1.append(s["n1"])
            e2.append(s["e2"]); n2.append(s["n2"])
            brg.append(s["bearing"]); owner.append(idx)
    return (np.array(e1), np.array(n1), np.array(e2), np.array(n2),
            np.array(brg), np.array(owner)), keep


def project_points_to_segments(pe, pn, segs, chunk=512):
    """For each point: distance, closest point, and index of every segment (chunked).

    Returns per-point arrays over ALL segments would be huge, so this yields, per chunk, the
    full distance matrix for the caller to reduce - keeping candidate selection flexible.
    """
    e1, n1, e2, n2, _, _ = segs
    dx, dy = e2 - e1, n2 - n1
    denom = dx * dx + dy * dy
    denom[denom == 0] = 1e-12
    for i in range(0, len(pe), chunk):
        px = pe[i:i + chunk, None]
        py = pn[i:i + chunk, None]
        t = np.clip(((px - e1) * dx + (py - n1) * dy) / denom, 0.0, 1.0)
        cx = e1 + t * dx
        cy = n1 + t * dy
        yield i, np.hypot(px - cx, py - cy), cx, cy


def ang_diff(a, b):
    """Smallest absolute difference between two bearings, in degrees."""
    d = np.abs((a - b) % 360.0)
    return np.minimum(d, 360.0 - d)


def match(model_dir, verbose=True):
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    refit_path = os.path.join(model_dir, "projection_refit.json")
    if not os.path.exists(refit_path):
        raise SystemExit("run pipeline/refine_projection.py first (no projection_refit.json)")
    refit = json.load(open(refit_path))
    f = refit["fitted"]
    reliable_to = float(refit.get("lane_reliable_to_m") or 200.0)

    source = meta["source_video"]
    tf, _ = telemetry_for(source)
    fw, fh = meta["frame_width"], meta["frame_height"]

    df = pd.read_csv(os.path.join(model_dir, "detections.csv")).sort_values(["track_id", "time_s"])
    px = ((df["x1"] + df["x2"]) / 2).to_numpy()
    py = df["y2"].to_numpy()
    e, n = project(px, py, tf, fw, fh, f["d_pitch_deg"], f["d_yaw_deg"],
                   f["alt_scale"], f["dx_m"], f["dy_m"])
    df["map_east_m"], df["map_north_m"] = e, n
    df["range_m"] = np.hypot(e, n)

    links, netmeta = load_network()
    segs, keep = flatten_segments(links)
    seg_brg, seg_owner = segs[4], segs[5]
    if verbose:
        print(f"{os.path.relpath(model_dir, REPO_ROOT)}: {len(df)} detections, "
              f"{len(keep)} candidate links, lane-reliable to {reliable_to:.0f} m")

    # Per-detection heading, from the track's own recent motion (a single-frame delta is
    # dominated by bbox jitter, so use a short backward window within the same track).
    heading = np.full(len(df), np.nan)
    tid = df["track_id"].to_numpy()
    for k in range(len(df)):
        j = k - 4
        if j >= 0 and tid[j] == tid[k]:
            dxe, dyn = e[k] - e[j], n[k] - n[j]
            if math.hypot(dxe, dyn) > 0.6:
                heading[k] = (math.degrees(math.atan2(dxe, dyn)) + 360.0) % 360.0
    df["heading_deg"] = heading

    speed = df["speed_kmh"].to_numpy() if "speed_kmh" in df.columns else np.full(len(df), np.nan)

    # Per-segment one-way flag, so the legality penalty can be applied as a whole-matrix
    # operation rather than per candidate.
    seg_oneway = np.array([keep[o]["oneway"] for o in seg_owner], dtype=bool)

    best_link = np.full(len(df), -1, dtype=int)
    best_dist = np.full(len(df), np.nan)
    best_cx = np.full(len(df), np.nan)
    best_cy = np.full(len(df), np.nan)
    best_brg = np.full(len(df), np.nan)

    for i0, dmat, cx, cy in project_points_to_segments(e, n, segs):
        m = dmat.shape[0]
        cost = dmat.copy()
        h = heading[i0:i0 + m][:, None]
        sp = speed[i0:i0 + m][:, None]
        trust_heading = ~np.isnan(h) & (np.isnan(sp) | (sp >= MIN_SPEED_FOR_HEADING))

        dev = ang_diff(h, seg_brg[None, :])        # 0 = with digitisation, 180 = against
        # Alignment term: being parallel to the road is good in EITHER direction, so fold
        # 180 deg back to 0. This is about "is this the road I'm driving along", not legality.
        align = np.minimum(dev, 180.0 - dev)
        cost += np.where(trust_heading, HEADING_WEIGHT * align, 0.0)
        # Legality term: travelling against a one-way way's digitisation. This is the signal
        # that separates the two carriageways of a dual carriageway, where distance alone
        # cannot (their centrelines sit closer together than our registration residual).
        wrong_way = trust_heading & (dev > 90.0) & seg_oneway[None, :]
        cost += np.where(wrong_way, WRONG_WAY_PENALTY, 0.0)

        idx = np.argmin(cost, axis=1)
        rows = np.arange(m)
        best_link[i0:i0 + m] = seg_owner[idx]
        best_dist[i0:i0 + m] = dmat[rows, idx]
        best_cx[i0:i0 + m] = cx[rows, idx]
        best_cy[i0:i0 + m] = cy[rows, idx]
        best_brg[i0:i0 + m] = seg_brg[idx]

    # --- temporal smoothing -------------------------------------------------------------
    # A vehicle cannot hop between links frame to frame. Without this, a track straddling the
    # median flickers between the two carriageways. Rolling mode within each track: cheap,
    # and enough for 10s clips where a vehicle stays on one link or transitions once.
    W = 9
    smoothed = best_link.copy()
    tid_arr = df["track_id"].to_numpy()
    start = 0
    for k in range(1, len(df) + 1):
        if k == len(df) or tid_arr[k] != tid_arr[start]:
            seg = best_link[start:k]
            if len(seg) >= 3:
                sm = seg.copy()
                for j in range(len(seg)):
                    lo, hi = max(0, j - W // 2), min(len(seg), j + W // 2 + 1)
                    vals, counts = np.unique(seg[lo:hi], return_counts=True)
                    sm[j] = vals[np.argmax(counts)]
                smoothed[start:k] = sm
            start = k
    n_changed = int((smoothed != best_link).sum())
    best_link = smoothed

    # Recompute geometry against the smoothed link choice, so offsets and lanes are measured
    # from the link the track is actually assigned to (not the per-frame argmin it lost).
    for k in range(len(df)):
        L = keep[best_link[k]]
        bd, bcx, bcy, bb = np.inf, np.nan, np.nan, np.nan
        for s in L["segments"]:
            dx, dy = s["e2"] - s["e1"], s["n2"] - s["n1"]
            dd = dx * dx + dy * dy or 1e-12
            t = max(0.0, min(1.0, ((e[k] - s["e1"]) * dx + (n[k] - s["n1"]) * dy) / dd))
            qx, qy = s["e1"] + t * dx, s["n1"] + t * dy
            d = math.hypot(e[k] - qx, n[k] - qy)
            if d < bd:
                bd, bcx, bcy, bb = d, qx, qy, s["bearing"]
        best_dist[k], best_cx[k], best_cy[k], best_brg[k] = bd, bcx, bcy, bb
    if verbose:
        print(f"  temporal smoothing changed {n_changed} of {len(df)} link assignments "
              f"({n_changed / len(df) * 100:.1f}%)")

    df["link_idx"] = best_link
    df["link_dist_m"] = best_dist
    df["seg_bearing"] = best_brg

    # Signed lateral offset: positive = left of the direction of travel.
    tb = np.radians(best_brg)
    dirx, diry = np.sin(tb), np.cos(tb)          # unit vector along the road (E, N)
    vx, vy = e - best_cx, n - best_cy
    signed = dirx * vy - diry * vx
    df["lateral_offset_m"] = signed

    names, hwy, lanes_tag, oneway = [], [], [], []
    for li in best_link:
        L = keep[li] if 0 <= li < len(keep) else None
        names.append(L["name"] if L else None)
        hwy.append(L["highway"] if L else None)
        lanes_tag.append(L["lanes"] if L else None)
        oneway.append(bool(L["oneway"]) if L else False)
    df["link_name"] = names
    df["link_highway"] = hwy
    df["link_lanes"] = lanes_tag
    df["link_oneway"] = oneway
    df["link_id"] = [keep[li]["id"] if 0 <= li < len(keep) else None for li in best_link]

    # Direction of travel relative to the way's digitisation.
    dev = ang_diff(df["heading_deg"].to_numpy(), best_brg)
    df["along_link"] = np.where(np.isnan(dev), None, dev < 90)

    # Lane index, gated by range and by having a lane count at all.
    nlanes = pd.to_numeric(df["link_lanes"], errors="coerce").to_numpy()
    nlanes = np.where(np.isnan(nlanes), 2.0, nlanes)     # assume 2 where untagged
    half = nlanes * LANE_W / 2.0
    raw = np.floor((half - signed) / LANE_W) + 1
    lane = np.clip(raw, 1, nlanes)
    usable = (df["range_m"].to_numpy() <= reliable_to) & (best_dist <= MAX_MATCH_DIST_M) \
        & (np.abs(signed) <= half + LANE_W)
    df["lane"] = np.where(usable, lane, np.nan)
    df["lane_confidence"] = np.where(
        usable, np.clip(1.0 - (best_dist / (LANE_W * 1.5)), 0.0, 1.0), 0.0).round(2)
    df.loc[best_dist > MAX_MATCH_DIST_M, ["link_id", "link_name", "lane"]] = np.nan

    lat, lon = [], []
    for ee, nn in zip(e, n):
        a, b = enu_to_wgs84(ee, nn, netmeta["origin_lat"], netmeta["origin_lon"])
        lat.append(round(a, 7)); lon.append(round(b, 7))
    df["lat"], df["lon"] = lat, lon

    out = os.path.join(model_dir, "map_matched.csv")
    cols = ["frame", "time_s", "track_id", "taxonomy_class", "lat", "lon",
            "map_east_m", "map_north_m", "range_m", "heading_deg", "speed_kmh",
            "link_id", "link_name", "link_highway", "link_lanes", "link_oneway",
            "along_link", "link_dist_m", "lateral_offset_m", "lane", "lane_confidence"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(out, index=False)

    matched = df["link_id"].notna()
    laned = df["lane"].notna()
    if verbose:
        print(f"  matched to a link: {matched.mean() * 100:.1f}%   "
              f"lane assigned: {laned.mean() * 100:.1f}%   "
              f"median link distance {np.nanmedian(best_dist):.2f} m")
        top = df[matched].groupby(df["link_name"].fillna("(unnamed)")).size() \
            .sort_values(ascending=False).head(5)
        for nm, c in top.items():
            print(f"    {c:6d}  {nm}")
    return df, keep, netmeta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    a = ap.parse_args()
    match(a.model_dir)
