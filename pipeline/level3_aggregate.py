"""
Level 3 — aggregate insight: summarises across many road users, time windows, and regions of
the road. Consumes the geo-referenced per-detection output of pipeline/level2_enrich.py
(east_m/north_m/speed_kmh) and emits one JSON per combo for the dashboard.

What it computes, mapping directly onto the Level 3 brief:
  * Classified counts by movement and interval  -> movements{}, intervals[]
  * Directional / turning movements             -> approach->exit O-D matrix, turn classification
  * Origin-destination distributions            -> od_matrix (by screenline-crossing pairs)
  * Speed profiles across the segment           -> speed_profile[] (binned along travel axis)
  * Where speeding concentrates                 -> speed_profile[].p85_kmh + hotspots[]
  * Lane volume and modal split by lane         -> lanes[] (offset across the travel axis)
  * Queue lengths                               -> queues[] (per interval, stopped-run extent)
  * Density, occupancy, flow-density            -> fundamental[] (per interval flow vs density)

Design decisions that matter for correctness:

**Robust statistics throughout.** Per-track max speed is unusable: ID switches make a track
"teleport", yielding physically impossible values (a car measured at 1331 km/h on real data).
Every aggregate here uses medians/percentiles over *per-detection* samples after an explicit
physical-plausibility filter, never per-track maxima. The count of rejected samples is
reported rather than silently dropped - a high reject rate is itself a tracking-quality signal.

**Direction from a PCA travel axis, not from image axes.** The road runs at an arbitrary angle
through the frame, so "lane offset" and "distance along the road" need the scene's own axes.
The principal component of all vehicle motion gives the travel direction; the perpendicular
gives lane offset. This is scene-agnostic - no hand-drawn lane polygons per video.

**Lanes are discovered, not assumed.** Lane boundaries come from clustering the perpendicular
offsets of observed vehicle paths, so the output reflects where traffic actually ran rather
than a guess about road geometry. Lane *count* is therefore an observation, and is reported
with its own occupancy so a spurious sparse "lane" is visible as such.

Usage: python pipeline/level3_aggregate.py            # all enriched combos
       python pipeline/level3_aggregate.py <model_dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Physical plausibility gate for a single speed sample, in km/h. Above this the sample is an
# artefact (ID switch / detection jitter), not a road user - see module docstring.
MAX_PLAUSIBLE_KMH = 120.0
# A road user slower than this is treated as stopped, for queue detection.
STOPPED_KMH = 4.0
INTERVAL_S = 2.0          # aggregation window for counts/flow/queues on a 10s clip
# Queued vehicles sit at jam spacing (~6-8 m). A larger gap along the travel axis separates two
# groups of stopped vehicles rather than extending one queue - see the queue block below.
MAX_QUEUE_GAP_M = 25.0
SPEED_BIN_COUNT = 8       # bins along the travel axis for the speed profile
VEHICLE_CLASSES = ("car", "LGV_candidate", "HGV_candidate", "bus", "motorcycle")


def travel_axis(df):
    """Returns (origin, along_unit, across_unit) for the scene's dominant travel direction.

    PCA on per-detection displacement vectors: the dominant eigenvector is the direction
    traffic actually moves. Displacements (not positions) are used so the axis reflects motion
    rather than where the road happens to sit in frame.
    """
    moving = df[df["speed_kmh"] > STOPPED_KMH]
    d = moving.groupby("track_id")[["east_m", "north_m"]].agg(["first", "last"])
    vecs = np.column_stack([
        d[("east_m", "last")] - d[("east_m", "first")],
        d[("north_m", "last")] - d[("north_m", "first")],
    ])
    vecs = vecs[np.isfinite(vecs).all(axis=1)]
    vecs = vecs[np.hypot(vecs[:, 0], vecs[:, 1]) > 2.0]   # ignore near-stationary tracks
    if len(vecs) < 5:
        return np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    # Sign-align before PCA: opposing traffic flows would otherwise cancel each other out.
    ref = vecs[np.argmax(np.hypot(vecs[:, 0], vecs[:, 1]))]
    flipped = np.where((vecs @ ref)[:, None] < 0, -vecs, vecs)
    _, _, vt = np.linalg.svd(flipped - flipped.mean(axis=0), full_matrices=False)
    along = vt[0] / np.linalg.norm(vt[0])
    across = np.array([-along[1], along[0]])
    origin = np.array([df["east_m"].median(), df["north_m"].median()])
    return origin, along, across


def compass(vec):
    """Cardinal label for a movement vector (world frame is East/North)."""
    ang = (np.degrees(np.arctan2(vec[0], vec[1])) + 360) % 360   # 0 = North, clockwise
    for lo, hi, name in [(337.5, 360, "N"), (0, 22.5, "N"), (22.5, 67.5, "NE"),
                         (67.5, 112.5, "E"), (112.5, 157.5, "SE"), (157.5, 202.5, "S"),
                         (202.5, 247.5, "SW"), (247.5, 292.5, "W"), (292.5, 337.5, "NW")]:
        if lo <= ang < hi:
            return name
    return "N"


def turn_type(entry_vec, exit_vec):
    """Classifies a movement as through / left / right / u-turn from entry and exit headings.
    Uses a signed 2D cross product, so it's correct regardless of the road's orientation in
    frame (written out scalar-wise: np.cross on 2-vectors is deprecated in NumPy 2)."""
    if np.linalg.norm(entry_vec) < 1e-6 or np.linalg.norm(exit_vec) < 1e-6:
        return "unknown"
    e = entry_vec / np.linalg.norm(entry_vec)
    x = exit_vec / np.linalg.norm(exit_vec)
    ang = np.degrees(np.arccos(float(np.clip(e @ x, -1, 1))))
    if ang < 30:
        return "through"
    if ang > 150:
        return "u-turn"
    cross_z = float(e[0] * x[1] - e[1] * x[0])
    return "left" if cross_z > 0 else "right"


def find_corridors(mv, min_share_pct=4.0):
    """Groups movements into directional corridors, then discovers lanes *within* each.

    Why not lanes directly across the whole scene: `multiroad` is a wide shot spanning ~315 m
    containing several roads at different angles. Perpendicular offset from one global travel
    axis conflates separate roads with lanes of one road - it produced an obviously wrong
    "16 lanes" before this was split out. A traffic engineer analyses lanes per approach, and
    that's what this does: bucket by direction of travel first (which is what distinguishes
    one road from another), then cluster lateral offset inside a corridor (which is what
    distinguishes lanes of the same road).

    Corridors below `min_share_pct` of traffic are dropped: with sparse data a handful of
    tracks can otherwise look like a "road" that isn't one.
    """
    corridors = []
    if mv.empty:
        return corridors
    for direction, grp in mv.groupby("from"):
        share = len(grp) / len(mv) * 100
        if share < min_share_pct or len(grp) < 5:
            continue
        # Lateral axis for THIS corridor, from its own mean heading.
        vecs = grp[["dx", "dy"]].to_numpy()
        mean_dir = vecs.mean(axis=0)
        if np.linalg.norm(mean_dir) < 1e-6:
            continue
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        perp = np.array([-mean_dir[1], mean_dir[0]])
        offs = grp[["start_e", "start_n"]].to_numpy() @ perp

        lanes = []
        order = np.argsort(offs)
        sorted_offs = offs[order]
        # ~3.5 m is a standard lane width; split on gaps wider than half that.
        splits = np.where(np.diff(sorted_offs) > 1.8)[0]
        for li, idxs in enumerate(np.split(order, splits + 1)):
            if len(idxs) < 3:
                continue
            members = grp.iloc[idxs]
            spd = members["median_kmh"].dropna()
            lanes.append({
                "lane": li, "centre_offset_m": round(float(np.median(offs[idxs])), 1),
                "vehicles": int(len(members)),
                "share_of_corridor_pct": round(len(members) / len(grp) * 100, 1),
                "median_kmh": round(float(spd.median()), 1) if len(spd) else None,
                "modal_split": members["cls"].value_counts().to_dict(),
            })
        cspd = grp["median_kmh"].dropna()
        corridors.append({
            "approach_from": direction,
            "heading_of_travel": compass(mean_dir),
            "vehicles": int(len(grp)),
            "share_pct": round(share, 1),
            "median_kmh": round(float(cspd.median()), 1) if len(cspd) else None,
            "p85_kmh": round(float(np.percentile(cspd, 85)), 1) if len(cspd) else None,
            "modal_split": grp["cls"].value_counts().to_dict(),
            "turn_counts": grp["turn"].value_counts().to_dict(),
            "lanes_observed": len(lanes),
            "lanes": lanes,
        })
    return sorted(corridors, key=lambda c: -c["vehicles"])


def analyse(model_dir):
    det_path = os.path.join(model_dir, "detections.csv")
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    df = pd.read_csv(det_path)
    if "speed_kmh" not in df.columns or not df["speed_kmh"].notna().any():
        return None, "no geo-referenced speeds - run pipeline/level2_enrich.py first"

    total_samples = int(df["speed_kmh"].notna().sum())
    df = df[df["east_m"].notna() & df["north_m"].notna()].copy()
    implausible = int((df["speed_kmh"] > MAX_PLAUSIBLE_KMH).sum())
    df.loc[df["speed_kmh"] > MAX_PLAUSIBLE_KMH, "speed_kmh"] = np.nan

    origin, along, across = travel_axis(df)
    pos = df[["east_m", "north_m"]].to_numpy() - origin
    df["along_m"] = pos @ along
    df["across_m"] = pos @ across
    duration_s = float(df["time_s"].max() - df["time_s"].min()) or 1.0

    veh = df[df["taxonomy_class"].isin(VEHICLE_CLASSES)]

    # ---- per-track movement: approach/exit heading, turn type, O-D --------------------
    movements, od = [], {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time_s")
        if len(g) < 6:
            continue
        p = g[["east_m", "north_m"]].to_numpy()
        entry_vec, exit_vec = p[min(4, len(p) - 1)] - p[0], p[-1] - p[max(0, len(p) - 5)]
        net = p[-1] - p[0]
        if np.hypot(*net) < 3.0:      # never really went anywhere: not a movement
            continue
        o, dst = compass(entry_vec), compass(exit_vec)
        movements.append({
            "track_id": int(tid), "cls": g["taxonomy_class"].mode().iat[0],
            "from": o, "to": dst, "turn": turn_type(entry_vec, exit_vec),
            "median_kmh": round(float(np.nanmedian(g["speed_kmh"])), 1)
                          if g["speed_kmh"].notna().any() else None,
            "lane_offset_m": round(float(np.nanmedian(g["across_m"])), 1),
            "t_enter": float(g["time_s"].iat[0]), "t_exit": float(g["time_s"].iat[-1]),
            # Kept for per-corridor lane analysis (find_corridors) - each corridor needs its
            # own lateral axis, so it needs the raw heading and entry position, not just the
            # global across_m offset.
            "dx": float(net[0]), "dy": float(net[1]),
            "start_e": float(p[0][0]), "start_n": float(p[0][1]),
        })
        od[f"{o}->{dst}"] = od.get(f"{o}->{dst}", 0) + 1
    mv = pd.DataFrame(movements)

    # ---- classified counts by movement and interval -----------------------------------
    intervals = []
    for start in np.arange(0, duration_s, INTERVAL_S):
        end = start + INTERVAL_S
        win = df[(df["time_s"] >= start) & (df["time_s"] < end)]
        if win.empty:
            continue
        wveh = win[win["taxonomy_class"].isin(VEHICLE_CLASSES)]
        # Density: mean simultaneous vehicle count in the window (per-frame occupancy).
        per_frame = wveh.groupby("frame")["track_id"].nunique()
        entering = mv[(mv["t_enter"] >= start) & (mv["t_enter"] < end)] if not mv.empty else pd.DataFrame()
        stopped = wveh[wveh["speed_kmh"] < STOPPED_KMH]["track_id"].nunique()
        intervals.append({
            "t_start": round(float(start), 1), "t_end": round(float(end), 1),
            "unique_vehicles": int(wveh["track_id"].nunique()),
            "unique_all": int(win["track_id"].nunique()),
            "by_class": win.groupby("taxonomy_class")["track_id"].nunique().to_dict(),
            "arrivals": int(len(entering)),
            "flow_veh_per_h": round(len(entering) / INTERVAL_S * 3600, 0),
            "mean_concurrent": round(float(per_frame.mean()), 1) if len(per_frame) else 0.0,
            "median_kmh": round(float(np.nanmedian(wveh["speed_kmh"])), 1)
                          if wveh["speed_kmh"].notna().any() else None,
            "stopped_vehicles": int(stopped),
            "turns": entering["turn"].value_counts().to_dict() if len(entering) else {},
        })

    # ---- speed profile along the segment, and where speeding concentrates -------------
    speed_profile, hotspots = [], []
    if veh["speed_kmh"].notna().any():
        lo, hi = np.nanpercentile(veh["along_m"], [2, 98])
        edges = np.linspace(lo, hi, SPEED_BIN_COUNT + 1)
        for i in range(SPEED_BIN_COUNT):
            b = veh[(veh["along_m"] >= edges[i]) & (veh["along_m"] < edges[i + 1])]
            s = b["speed_kmh"].dropna()
            if len(s) < 20:
                continue
            # p85 is the standard traffic-engineering measure for prevailing speed.
            p85 = float(np.percentile(s, 85))
            speed_profile.append({
                "bin": i, "along_from_m": round(float(edges[i]), 1),
                "along_to_m": round(float(edges[i + 1]), 1),
                "n_samples": int(len(s)), "median_kmh": round(float(s.median()), 1),
                "p85_kmh": round(p85, 1),
                "vehicles": int(b["track_id"].nunique()),
            })
        if speed_profile:
            worst = max(speed_profile, key=lambda b: b["p85_kmh"])
            hotspots.append({
                "kind": "highest_prevailing_speed",
                "along_from_m": worst["along_from_m"], "along_to_m": worst["along_to_m"],
                "p85_kmh": worst["p85_kmh"],
                "note": "p85 speed - the standard traffic-engineering measure of prevailing speed",
            })

    # ---- directional corridors, and lanes within each --------------------------------
    corridors = find_corridors(mv)

    # ---- queue lengths: spatial extent of stopped vehicles, per interval --------------
    queues = []
    stopped_all = veh[veh["speed_kmh"] < STOPPED_KMH]
    for start in np.arange(0, duration_s, INTERVAL_S):
        win = stopped_all[(stopped_all["time_s"] >= start) & (stopped_all["time_s"] < start + INTERVAL_S)]
        n = win["track_id"].nunique()
        if n < 2:
            continue
        # Queue length = along-axis extent of the longest CONTIGUOUS group of stopped vehicles.
        # Two things were wrong with taking max - min over every stopped vehicle in the window:
        #   1. no contiguity - two vehicles halted at opposite ends of the segment reported the
        #      whole segment as one queue (~32 m of "queue" per vehicle, i.e. mostly empty road);
        #   2. no separation between roads - `along_m` is the projection onto ONE global travel
        #      axis, so on the multiroad scene vehicles stopped on different carriageways land
        #      at similar along-values and chain into a single fictitious 350 m queue.
        # Clustering in real 2D metres fixes both: vehicles on a parallel road are tens of
        # metres away laterally and so never join the chain. Single-linkage, since a queue IS a
        # chain - each car near the one in front, with the queue arbitrarily long overall.
        grp = win.groupby("track_id")[["east_m", "north_m", "along_m"]].median().dropna()
        if len(grp) < 2:
            continue
        pts = grp[["east_m", "north_m"]].to_numpy()
        along = grp["along_m"].to_numpy()
        order = np.argsort(along)                  # sweep along the road, so neighbours adjoin
        pts, along = pts[order], along[order]

        clusters, cur = [], [0]
        for i in range(1, len(pts)):
            if float(np.hypot(*(pts[i] - pts[cur[-1]]))) > MAX_QUEUE_GAP_M:
                clusters.append(cur); cur = []
            cur.append(i)
        clusters.append(cur)

        best = max(clusters, key=lambda c: along[c[-1]] - along[c[0]])
        if len(best) < 2:
            continue
        extent = float(along[best[-1]] - along[best[0]])
        queues.append({
            "t_start": round(float(start), 1), "stopped_vehicles": int(n),
            "queued_vehicles": len(best),          # members of the longest contiguous chain
            "queue_length_m": round(extent, 1),
            "classes": win.groupby("taxonomy_class")["track_id"].nunique().to_dict(),
        })

    # ---- fundamental diagram: flow vs density per interval ---------------------------
    # Density here is vehicles per 100 m of the observed travel extent, so it's comparable
    # between scenes with different visible road lengths.
    extent_m = float(np.nanpercentile(veh["along_m"], 98) - np.nanpercentile(veh["along_m"], 2)) \
        if veh["along_m"].notna().any() else 0.0
    fundamental = []
    for iv in intervals:
        if extent_m <= 0:
            break
        density = iv["mean_concurrent"] / extent_m * 100
        fundamental.append({
            "t_start": iv["t_start"], "flow_veh_per_h": iv["flow_veh_per_h"],
            "density_veh_per_100m": round(density, 2),
            "median_kmh": iv["median_kmh"],
            "occupancy_pct": round(min(100.0, density * 5.0 / 100 * 100), 1),
        })

    result = {
        "source_video": meta.get("source_video"), "model": meta.get("model"),
        "duration_s": round(duration_s, 2), "tracker": meta.get("tracker"),
        "travel_axis": {
            "along_unit_east_north": [round(float(along[0]), 3), round(float(along[1]), 3)],
            "heading": compass(along),
            "observed_extent_m": round(extent_m, 1),
            "note": "PCA of vehicle displacement - the scene's own road direction, not an image axis",
        },
        "quality": {
            "speed_samples": total_samples,
            "rejected_implausible": implausible,
            "rejected_pct": round(implausible / total_samples * 100, 2) if total_samples else 0.0,
            "note": (f"samples above {MAX_PLAUSIBLE_KMH} km/h are dropped as ID-switch/jitter "
                     "artefacts; a high rate indicates tracking instability, not fast traffic"),
        },
        "totals": {
            "tracks": int(df["track_id"].nunique()),
            "vehicles": int(veh["track_id"].nunique()),
            "with_movement": int(len(mv)),
            "median_kmh": round(float(np.nanmedian(veh["speed_kmh"])), 1)
                          if veh["speed_kmh"].notna().any() else None,
            "p85_kmh": round(float(np.nanpercentile(veh["speed_kmh"].dropna(), 85)), 1)
                       if veh["speed_kmh"].notna().any() else None,
            "modal_split": (mv["cls"].value_counts().to_dict() if not mv.empty else {}),
            "turn_counts": (mv["turn"].value_counts().to_dict() if not mv.empty else {}),
        },
        "od_matrix": dict(sorted(od.items(), key=lambda kv: -kv[1])),
        "intervals": intervals,
        "speed_profile": speed_profile,
        "hotspots": hotspots,
        "corridors": corridors,
        "lanes_note": ("Lanes are reported per directional corridor, not scene-wide: a wide "
                       "shot can contain several roads, and lateral offset from one global "
                       "axis would conflate separate roads with lanes of one road. Lane count "
                       "is an observation of where traffic actually ran, not surveyed geometry."),
        "queues": queues,
        "fundamental": fundamental,
    }
    return result, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?", default=None)
    args = parser.parse_args()

    dirs = [args.model_dir] if args.model_dir else sorted(
        glob.glob(os.path.join(REPO_ROOT, "videos", "samples", "*", "*", "*")))
    written = 0
    for d in dirs:
        if not os.path.exists(os.path.join(d, "detections.csv")):
            continue
        res, err = analyse(d)
        name = os.path.relpath(d, os.path.join(REPO_ROOT, "videos", "samples"))
        if err:
            print(f"{name}: skipped - {err}")
            continue
        with open(os.path.join(d, "aggregate_insight.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        t = res["totals"]
        lanes_total = sum(c["lanes_observed"] for c in res["corridors"])
        print(f"{name}: {t['vehicles']} vehicles, {t['with_movement']} movements, "
              f"median {t['median_kmh']} km/h (p85 {t['p85_kmh']}), "
              f"{len(res['corridors'])} corridors / {lanes_total} lanes, "
              f"{len(res['queues'])} queue windows, "
              f"{res['quality']['rejected_pct']}% samples rejected")
        written += 1
    print(f"\nWrote aggregate_insight.json for {written} combo(s).")


if __name__ == "__main__":
    main()
