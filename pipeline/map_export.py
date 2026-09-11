"""
Level 4 stage F - map-native output.

Two deliverables, deliberately different in purpose:

1. `level4.geojson` - standard WGS84 GeoJSON, openable directly in QGIS, geojson.io, Google
   Earth or any GIS. This is what "map-native" means: real coordinates on the real road
   network, not pixels in a video frame.
2. `map_layers.json` - the same content pre-projected into local ENU metres for the dashboard's
   map panel, which draws the OSM geometry as vectors rather than fetching raster tiles. No
   tile server means no external dependency, no API key, and it still works with the network
   off - which matters for a live demo.

Layers produced (all four are Level 4 brief items):
  * roads        - the mapped carriageways themselves, with name/lanes/oneway
  * trajectories - each track as a line, carrying its matched link, direction and lane
  * desire_lines - aggregated origin-link -> destination-link VEHICLE movements, weighted by count
  * queues       - queue extents drawn ALONG the carriageway, not as a bounding box

Queue geometry is computed here rather than reused from Level 3: Level 3 reports a queue's
*length* in metres, but a map needs its *position and extent along the carriageway*, which
requires the map-matched along-link coordinate that only exists after stage D.

**Pedestrians are excluded from desire lines and lane_metrics, not silently mixed in.** map_
match.py binds every taxonomy class onto the nearest DRIVABLE link (a pedestrian standing
beside the road has no sidewalk geometry to bind to instead, since footways are fetched but
flagged non-drivable and excluded as match candidates). Left unfiltered here, that meant a
desire line's `vehicles` count and a lane's `median_kmh` were real numbers with pedestrians
silently inside them - caught by inspecting real output: one lane's 111 "vehicles" included
21 pedestrians, and walking pace (~5 km/h) was dragging that lane's reported speed down.
Vehicle aggregates below use VEHICLE_CLASSES (the same constant as level3_aggregate.py, so
the two levels report a consistent vehicle definition); pedestrian activity is reported
separately in `pedestrian_crossings`, since where people cross - not which "lane" they're
notionally in - is what's meaningful for a pedestrian bound onto a road link.

Usage: python pipeline/map_export.py <model_dir>
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.geodesy import enu_to_wgs84
from pipeline.road_network import load_network

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPPED_KMH = 4.0
INTERVAL_S = 2.0
MIN_QUEUE_VEHICLES = 3
# Queued vehicles sit at jam spacing (~6-8 m centre to centre). A gap wider than this along
# the link means we are looking at two separate groups of stopped vehicles, not one queue.
MAX_QUEUE_GAP_M = 25.0
# A track whose bounding displacement over the whole clip is under this never went anywhere:
# parked, not queuing. Set above map-matching jitter (median residual ~2 m) so a genuinely
# queued vehicle that creeps forward is not mistaken for a parked one.
PARKED_DISPLACEMENT_M = 8.0
# Same set as pipeline/level3_aggregate.py - kept identical so "vehicle" means the same thing
# in both levels' output rather than two silently different definitions.
VEHICLE_CLASSES = ("car", "LGV_candidate", "HGV_candidate", "bus", "motorcycle")


def along_link_position(L, e, n):
    """Distance along a link's polyline to the point nearest (e, n), in metres from its start.

    Needed for queue extents: a queue is a stretch OF the carriageway, so its endpoints have
    to be expressed in the link's own linear reference, not as raw coordinates.
    """
    best_d, best_s = float("inf"), 0.0
    run = 0.0
    for s in L["segments"]:
        dx, dy = s["e2"] - s["e1"], s["n2"] - s["n1"]
        dd = dx * dx + dy * dy or 1e-12
        t = max(0.0, min(1.0, ((e - s["e1"]) * dx + (n - s["n1"]) * dy) / dd))
        qx, qy = s["e1"] + t * dx, s["n1"] + t * dy
        d = math.hypot(e - qx, n - qy)
        if d < best_d:
            best_d, best_s = d, run + t * s["len"]
        run += s["len"]
    return best_s, best_d


def point_at_along(L, s_target):
    """Inverse of along_link_position: the (e, n) at a given distance along the link."""
    run = 0.0
    for s in L["segments"]:
        if run + s["len"] >= s_target:
            t = (s_target - run) / s["len"] if s["len"] else 0.0
            return (s["e1"] + t * (s["e2"] - s["e1"]), s["n1"] + t * (s["n2"] - s["n1"]))
        run += s["len"]
    last = L["segments"][-1]
    return (last["e2"], last["n2"])


def build(model_dir):
    mm_path = os.path.join(model_dir, "map_matched.csv")
    if not os.path.exists(mm_path):
        raise SystemExit("run pipeline/map_match.py first (no map_matched.csv)")
    df = pd.read_csv(mm_path)
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    links, netmeta = load_network()
    by_id = {L["id"]: L for L in links}
    olat, olon = netmeta["origin_lat"], netmeta["origin_lon"]

    def wgs(e, n):
        lat, lon = enu_to_wgs84(e, n, olat, olon)
        return [round(lon, 7), round(lat, 7)]     # GeoJSON is [lon, lat]

    features, layers = [], {"roads": [], "trajectories": [], "desire_lines": [], "queues": [],
                            "pedestrians": []}

    # ---- roads actually used by this scene, plus their near neighbours for context -------
    used = set(df["link_id"].dropna().astype(int))
    for L in links:
        if not L["drivable"]:
            continue
        near = min(math.hypot(e, n) for e, n in L["points"]) < 300
        if L["id"] not in used and not near:
            continue
        props = {"layer": "roads", "link_id": L["id"], "name": L["name"],
                 "highway": L["highway"], "lanes": L["lanes"], "oneway": L["oneway"],
                 "carried_traffic": L["id"] in used}
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "LineString",
                                      "coordinates": [wgs(e, n) for e, n in L["points"]]}})
        layers["roads"].append({**props, "points": [[round(e, 1), round(n, 1)]
                                                    for e, n in L["points"]]})

    # ---- trajectories --------------------------------------------------------------------
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time_s")
        if len(g) < 4:
            continue
        matched = g[g["link_id"].notna()]
        if matched.empty:
            continue
        link_id = int(matched["link_id"].mode().iat[0])
        lanes = matched["lane"].dropna()
        speeds = g["speed_kmh"].dropna() if "speed_kmh" in g.columns else pd.Series(dtype=float)
        props = {
            "layer": "trajectories", "track_id": int(tid),
            "class": g["taxonomy_class"].mode().iat[0],
            "link_id": link_id,
            "link_name": matched["link_name"].dropna().mode().iat[0]
            if matched["link_name"].notna().any() else None,
            "lane": int(lanes.mode().iat[0]) if len(lanes) else None,
            "lane_stability": round(float((lanes == lanes.mode().iat[0]).mean()), 2)
            if len(lanes) else None,
            "direction_along_link": bool(matched["along_link"].dropna().mode().iat[0])
            if matched["along_link"].notna().any() else None,
            "median_kmh": round(float(speeds.median()), 1) if len(speeds) else None,
            "n_points": int(len(g)),
        }
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "LineString",
                                      "coordinates": [wgs(e, n) for e, n in
                                                      zip(g["map_east_m"], g["map_north_m"])]}})
        layers["trajectories"].append({
            **props,
            "points": [[round(e, 1), round(n, 1)]
                       for e, n in zip(g["map_east_m"], g["map_north_m"])],
        })

    # ---- desire lines: vehicle-to-vehicle link movement, weighted ------------------------
    # Restricted to VEHICLE_CLASSES - see module docstring for the real contamination this
    # fixes (pedestrians were previously counted inside a line's "vehicles" weight).
    veh_df = df[df["taxonomy_class"].isin(VEHICLE_CLASSES)]
    od = defaultdict(int)
    for tid, g in veh_df.groupby("track_id"):
        m = g[g["link_id"].notna()].sort_values("time_s")
        if len(m) < 6:
            continue
        a, b = int(m["link_id"].iat[0]), int(m["link_id"].iat[-1])
        od[(a, b)] += 1
    for (a, b), w in sorted(od.items(), key=lambda kv: -kv[1]):
        if w < 2 or a not in by_id or b not in by_id:
            continue
        # Anchor each end at the mid-point of its link, so the desire line runs between the
        # actual road segments rather than between arbitrary vehicle positions.
        pa = by_id[a]["points"][len(by_id[a]["points"]) // 2]
        pb = by_id[b]["points"][len(by_id[b]["points"]) // 2]
        props = {"layer": "desire_lines", "from_link": a, "to_link": b, "vehicles": w,
                 "from_name": by_id[a]["name"], "to_name": by_id[b]["name"],
                 "same_link": a == b}
        features.append({"type": "Feature", "properties": props,
                         "geometry": {"type": "LineString",
                                      "coordinates": [wgs(*pa), wgs(*pb)]}})
        layers["desire_lines"].append({**props,
                                       "points": [[round(pa[0], 1), round(pa[1], 1)],
                                                  [round(pb[0], 1), round(pb[1], 1)]]})

    # ---- queue extents, drawn along the carriageway --------------------------------------
    # A queue is a CONTIGUOUS run of stopped vehicles, so its extent has to be built by
    # splitting on gaps - not as min-to-max of every stopped vehicle on the link. Taking the
    # raw span produced 310-482 m "queues" from 4-15 vehicles on 10s of footage (~32 m per
    # vehicle, i.e. mostly empty road), drawn as polygons that spanned the whole link and lay
    # across buildings on the map. Two further filters matter for the same reason:
    #   * pedestrians - a person standing at the kerb reads as stopped, and map_match binds
    #     them onto the carriageway. Same class of bug already fixed for desire lines and lane
    #     metrics; queues were still using the unfiltered frame.
    #   * parked vehicles - a car that never moves in the whole clip is not queuing. These
    #     were the giveaway: one service link reported an identical 310.3 m queue in EVERY
    #     interval, which is a stationary row of parked cars, not traffic.
    if "speed_kmh" in veh_df.columns:
        moved = veh_df.groupby("track_id").apply(
            lambda g: math.hypot(g["map_east_m"].max() - g["map_east_m"].min(),
                                 g["map_north_m"].max() - g["map_north_m"].min()),
            include_groups=False)
        mobile = set(moved[moved >= PARKED_DISPLACEMENT_M].index)
        stopped = veh_df[(veh_df["speed_kmh"] < STOPPED_KMH) & veh_df["link_id"].notna()
                         & veh_df["track_id"].isin(mobile)]
        tmax = float(veh_df["time_s"].max())
        for t0 in np.arange(0, tmax, INTERVAL_S):
            win = stopped[(stopped["time_s"] >= t0) & (stopped["time_s"] < t0 + INTERVAL_S)]
            for link_id, grp in win.groupby("link_id"):
                L = by_id.get(int(link_id))
                if L is None:
                    continue
                # One along-link position per TRACK (its median), so a vehicle sampled over
                # several frames in the window contributes one queue member, not several.
                per_track = {}
                for tid, e, n in zip(grp["track_id"], grp["map_east_m"], grp["map_north_m"]):
                    per_track.setdefault(tid, []).append(along_link_position(L, e, n)[0])
                svals = sorted(float(np.median(v)) for v in per_track.values())

                # Split into contiguous runs: consecutive queued vehicles sit at jam spacing
                # (~6-8 m); a gap wider than MAX_QUEUE_GAP_M means the queue has ended and a
                # separate group of stopped vehicles begins further up the link.
                runs, run = [], [svals[0]] if svals else []
                for s in svals[1:]:
                    if s - run[-1] > MAX_QUEUE_GAP_M:
                        runs.append(run); run = []
                    run.append(s)
                if run:
                    runs.append(run)

                for run in runs:
                    if len(run) < MIN_QUEUE_VEHICLES:
                        continue
                    s_lo, s_hi = run[0], run[-1]
                    if s_hi - s_lo < 3.0:
                        continue
                    pts = [point_at_along(L, s) for s in np.linspace(s_lo, s_hi, 12)]
                    props = {"layer": "queues", "t_start": round(float(t0), 1),
                             "link_id": int(link_id), "link_name": L["name"],
                             "vehicles": len(run),
                             "queue_length_m": round(float(s_hi - s_lo), 1)}
                    features.append({"type": "Feature", "properties": props,
                                     "geometry": {"type": "LineString",
                                                  "coordinates": [wgs(*p) for p in pts]}})
                    layers["queues"].append({**props,
                                             "points": [[round(p[0], 1), round(p[1], 1)]
                                                        for p in pts]})

    # ---- per-lane metrics on real geometry ----------------------------------------------
    # VEHICLE_CLASSES only: a pedestrian bound onto a carriageway has no meaningful "lane",
    # and their walking pace was measurably dragging lane speeds down before this filter.
    lane_metrics = []
    laned = veh_df[veh_df["lane"].notna() & veh_df["link_id"].notna()]
    for (link_id, lane), g in laned.groupby(["link_id", "lane"]):
        L = by_id.get(int(link_id))
        sp = g["speed_kmh"].dropna() if "speed_kmh" in g.columns else pd.Series(dtype=float)

        # Two speeds, reported separately because they answer different questions and
        # conflating them misleads. Checked against real data: quoting only the all-inclusive
        # median made one carriageway look like it had a 8x lane speed differential; once
        # never-moving vehicles were excluded that carriageway's lanes were identical
        # (13.7 vs 13.9 km/h), while the other kept a genuine 6.8 vs 16.4 split.
        #   * median_kmh        - every detection, stopped ones included. This is the lane's
        #                         actual throughput performance (level-of-service sense).
        #   * running_kmh       - only vehicles that moved at some point, i.e. running speed.
        #                         Excludes parked/stalled vehicles occupying the lane.
        per_track_max = g.groupby("track_id")["speed_kmh"].max() if len(sp) else pd.Series(dtype=float)
        movers = set(per_track_max[per_track_max > 8.0].index)
        gm = g[g["track_id"].isin(movers)]
        spm = gm["speed_kmh"].dropna() if len(gm) else pd.Series(dtype=float)

        lane_metrics.append({
            "link_id": int(link_id), "link_name": L["name"] if L else None,
            "lane": int(lane), "vehicles": int(g["track_id"].nunique()),
            "detections": int(len(g)),
            "median_kmh": round(float(sp.median()), 1) if len(sp) else None,
            "p85_kmh": round(float(np.percentile(sp, 85)), 1) if len(sp) else None,
            "running_kmh": round(float(spm.median()), 1) if len(spm) else None,
            "moving_vehicles": len(movers),
            "stationary_vehicles": int(g["track_id"].nunique() - len(movers)),
            "modal_split": g.groupby("taxonomy_class")["track_id"].nunique().to_dict(),
        })
    lane_metrics.sort(key=lambda r: (-r["vehicles"]))

    # ---- pedestrians, reported on their own terms ----------------------------------------
    # These are bound onto vehicle carriageways because footways are excluded as match
    # candidates, so their "lane" is meaningless - but WHERE they are on the carriageway is
    # exactly what a crossing analysis wants. Reported as points, not lanes.
    ped_layer = []
    peds = df[(df["taxonomy_class"] == "pedestrian") & df["link_id"].notna()]
    for (link_id), g in peds.groupby("link_id"):
        L = by_id.get(int(link_id))
        if L is None:
            continue
        for tid, t in g.groupby("track_id"):
            t = t.sort_values("time_s")
            e0, n0 = float(t["map_east_m"].iat[0]), float(t["map_north_m"].iat[0])
            e1, n1 = float(t["map_east_m"].iat[-1]), float(t["map_north_m"].iat[-1])
            # Lateral offset swing across the carriageway is the crossing signal: someone
            # walking ALONG the kerb barely changes it, someone crossing sweeps through it.
            lat_swing = float(t["lateral_offset_m"].max() - t["lateral_offset_m"].min())
            ped_layer.append({
                "layer": "pedestrians", "track_id": int(tid),
                "link_id": int(link_id), "link_name": L["name"],
                "lateral_swing_m": round(lat_swing, 1),
                "likely_crossing": bool(lat_swing > 4.0),
                "points": [[round(e0, 1), round(n0, 1)], [round(e1, 1), round(n1, 1)]],
            })
    crossings = sum(1 for p in ped_layer if p["likely_crossing"])
    layers["pedestrians"] = ped_layer

    geo = {"type": "FeatureCollection",
           "properties": {"source_video": meta.get("source_video"), "model": meta.get("model"),
                          "crs": "WGS84", "road_data": netmeta["source"]},
           "features": features}
    with open(os.path.join(model_dir, "level4.geojson"), "w") as f:
        json.dump(geo, f)
    payload = {"origin": {"lat": olat, "lon": olon},
               "source_video": meta.get("source_video"), "model": meta.get("model"),
               "layers": layers, "lane_metrics": lane_metrics,
               "road_data": netmeta["source"]}
    with open(os.path.join(model_dir, "map_layers.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"{os.path.relpath(model_dir, REPO_ROOT)}")
    print(f"  roads {len(layers['roads'])}  trajectories {len(layers['trajectories'])}  "
          f"desire lines {len(layers['desire_lines'])}  queue extents {len(layers['queues'])}")
    print(f"  pedestrians on carriageway: {len(ped_layer)} ({crossings} likely crossing) "
          f"- excluded from desire lines and lane metrics")
    print(f"  per-lane metrics rows: {len(lane_metrics)}")
    for r in lane_metrics[:6]:
        print(f"    {(r['link_name'] or '(unnamed)')[:26]:26s} lane {r['lane']}  "
              f"{r['vehicles']:4d} veh  all {str(r['median_kmh']):>5s}  "
              f"running {str(r['running_kmh']):>5s} km/h  ({r['stationary_vehicles']} parked)")
    print(f"  wrote level4.geojson ({os.path.getsize(os.path.join(model_dir,'level4.geojson'))/1e6:.1f} MB) "
          f"+ map_layers.json")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    a = ap.parse_args()
    build(a.model_dir)
