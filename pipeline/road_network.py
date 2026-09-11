"""
Fetches the OpenStreetMap road network around a scene and projects it into the same local
East/North metre frame the trajectories live in (pipeline/geo_reference_telemetry.py).

Cached to disk after the first fetch: the camera is fixed and the road layout doesn't change
between runs, so this hits the network once per site, never per run.

What's actually available at this site (verified against the live Overpass API, not assumed):
Gopal Hari Deshmukh Marg is mapped as a **dual carriageway** - two separate ways, each
`oneway=yes` with `lanes=2`, ~35-46 m from the drone. That's link, approach, direction and
lane count already present in the data, which is what makes lane-level binding possible here.

Usage: python pipeline/road_network.py            # fetch + cache + summarise
"""
import json
import math
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.geodesy import wgs84_to_enu

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "data_cache")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Both source clips are the same hover point (confirmed from telemetry - they differ only in
# gimbal yaw), so one network cache serves both scenes.
SITE = {"lat": 18.566225, "lon": 73.771845, "name": "pune_gopal_hari_deshmukh_marg"}
BBOX_DEG = 0.005          # ~550 m each way - comfortably covers the 315 m scene extent

# Ways that carry motor traffic. Footways/steps/paths are fetched but flagged non-drivable:
# they're useful context to draw on the map, but a vehicle must never map-match onto them.
DRIVABLE = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "service", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link",
}


def fetch_osm(lat, lon, d=BBOX_DEG, timeout=90):
    query = (f"[out:json][timeout:60];"
             f'(way["highway"]({lat - d},{lon - d},{lat + d},{lon + d}););'
             f"out geom tags;")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "flytbase-hackathon/1.0 (traffic analytics research)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def bearing_deg(e1, n1, e2, n2):
    """Compass bearing (0=N, clockwise) of the segment (e1,n1)->(e2,n2) in the ENU frame."""
    return (math.degrees(math.atan2(e2 - e1, n2 - n1)) + 360.0) % 360.0


def build_network(raw, origin_lat, origin_lon):
    """Converts raw Overpass output into links in local ENU metres.

    Each link keeps its per-segment bearings, because map matching needs to compare a
    vehicle's heading against the road's heading *at the point it's near* - a curved way has
    no single bearing, and using an average would break matching exactly on the bends where
    it matters most.
    """
    links = []
    for w in raw.get("elements", []):
        if w.get("type") != "way" or "geometry" not in w:
            continue
        tags = w.get("tags", {})
        hw = tags.get("highway")
        if not hw:
            continue
        pts = [wgs84_to_enu(p["lat"], p["lon"], origin_lat, origin_lon) for p in w["geometry"]]
        if len(pts) < 2:
            continue

        segs, length = [], 0.0
        for i in range(len(pts) - 1):
            (e1, n1), (e2, n2) = pts[i], pts[i + 1]
            L = math.hypot(e2 - e1, n2 - n1)
            if L < 1e-9:
                continue
            length += L
            segs.append({"e1": e1, "n1": n1, "e2": e2, "n2": n2,
                         "len": L, "bearing": bearing_deg(e1, n1, e2, n2)})
        if not segs:
            continue

        oneway = tags.get("oneway", "no")
        try:
            lanes = int(str(tags.get("lanes", "")).split(";")[0])
        except ValueError:
            lanes = None
        links.append({
            "id": w["id"],
            "name": tags.get("name"),
            "highway": hw,
            "drivable": hw in DRIVABLE,
            "oneway": oneway in ("yes", "true", "1", "-1"),
            "oneway_reversed": oneway == "-1",
            "lanes": lanes,
            "maxspeed": tags.get("maxspeed"),
            "points": pts,
            "segments": segs,
            "length_m": round(length, 1),
        })
    return links


def load_network(site=SITE, refresh=False):
    """Returns (links, meta), fetching from Overpass only if not already cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"osm_{site['name']}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as f:
            d = json.load(f)
        return d["links"], d["meta"]

    raw = fetch_osm(site["lat"], site["lon"])
    links = build_network(raw, site["lat"], site["lon"])
    meta = {"site": site, "bbox_deg": BBOX_DEG, "way_count": len(links),
            "origin_lat": site["lat"], "origin_lon": site["lon"],
            "source": "OpenStreetMap via Overpass API (ODbL)"}
    with open(path, "w") as f:
        json.dump({"links": links, "meta": meta}, f)
    return links, meta


def nearby(links, max_dist_m=350.0, drivable_only=True):
    """Links with any vertex within max_dist_m of the origin (the drone's ground point)."""
    out = []
    for L in links:
        if drivable_only and not L["drivable"]:
            continue
        d = min(math.hypot(e, n) for e, n in L["points"])
        if d <= max_dist_m:
            out.append((round(d, 1), L))
    return sorted(out, key=lambda x: x[0])


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    links, meta = load_network(refresh=refresh)
    print(f"{len(links)} ways cached  ({meta['source']})")
    drivable = [L for L in links if L["drivable"]]
    print(f"{len(drivable)} drivable, {len(links) - len(drivable)} non-drivable (context only)")

    print("\nDrivable links within 200 m of the drone's ground point:")
    print(f"  {'dist':>5s}  {'highway':13s} {'name':30s} {'lanes':>5s} {'1way':>5s} {'len_m':>7s}")
    for d, L in nearby(links, 200.0):
        print(f"  {d:5.0f}  {L['highway']:13s} {(L['name'] or '-')[:30]:30s} "
              f"{str(L['lanes'] or '-'):>5s} {('yes' if L['oneway'] else 'no'):>5s} "
              f"{L['length_m']:7.1f}")

    # The dual carriageway is the anchor for direction/lane binding - assert it's really there
    # rather than trusting the earlier one-off probe.
    ghdm = [L for L in links if (L["name"] or "").startswith("Gopal Hari")]
    assert len(ghdm) >= 2, f"expected a split dual carriageway, found {len(ghdm)} way(s)"
    assert all(L["oneway"] for L in ghdm), "carriageways should be oneway"
    print(f"\nAnchor road: {len(ghdm)} one-way carriageways of '{ghdm[0]['name']}', "
          f"lanes={[L['lanes'] for L in ghdm]}")
    print("Self-test OK - network cached and the dual carriageway is present and tagged.")
