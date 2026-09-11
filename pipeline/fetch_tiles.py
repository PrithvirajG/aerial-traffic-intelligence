"""
Downloads and caches official map tiles for the scene area, so the dashboard can show a real
basemap instead of hand-drawn vectors.

Why cache rather than fetch live from the browser:
  * it still works with the network off, which matters during a live demo;
  * it is the polite way to use the OSM tile service, whose usage policy discourages bulk or
    repeated automated fetching - we pull a few dozen tiles for one small area, once, and
    then never hit the server again.

Two basemaps are fetched:
  * `osm`  - OpenStreetMap standard. Shows road names, junctions, land use.
             (c) OpenStreetMap contributors, ODbL.
  * `sat`  - Esri World Imagery. Satellite view, which for drone footage is the more useful
             backdrop: you can see the actual carriageways the trajectories sit on.
             (c) Esri, Maxar, Earthstar Geographics.

Both attributions are rendered in the dashboard - they are required, not optional.

Usage: python pipeline/fetch_tiles.py [--zoom 18 19] [--radius-m 400]
"""
import argparse
import json
import math
import os
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE_DIR = os.path.join(REPO_ROOT, "web", "tiles")
SITE_LAT, SITE_LON = 18.566225, 73.771845

SOURCES = {
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "max_zoom": 19,
    },
    "sat": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "© Esri, Maxar, Earthstar Geographics",
        # Esri imagery goes deeper than OSM's standard style, which stops at 19. The extra
        # level is what makes zooming in stay sharp instead of upscaling.
        "max_zoom": 20,
    },
}

# Tile count grows 4x per zoom level, so the cached radius shrinks as zoom deepens. This
# keeps the one-off download polite to two free services while still giving a sharp close-up
# over the area that actually contains traffic.
# Per-zoom cache radius around the site. The two outer levels are deliberately wider than the
# analysed area: multiroad's trajectories reach ~580 m west of the drone, and the fit-to-data
# view then pads beyond that, so a 450 m radius left black bands down both edges of the map -
# which reads as "the plotting is off" even though the points themselves are correctly placed.
# The deep levels stay tight because tile count grows 4x per level and they are only ever
# visible when zoomed into a small area.
ZOOM_RADIUS_M = {17: 900, 18: 800, 19: 450, 20: 220}


def deg2tile(lat, lon, z):
    """WGS84 -> Web Mercator tile indices (float, so the fraction gives sub-tile position)."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch(zooms, radius_m, lat=SITE_LAT, lon=SITE_LON, delay=0.12):
    os.makedirs(TILE_DIR, exist_ok=True)

    index = {"site": {"lat": lat, "lon": lon}, "sources": {}, "zooms": []}
    for name, src in SOURCES.items():
        got, failed = 0, 0
        tiles = []
        for z in zooms:
            if z > src["max_zoom"]:
                continue
            # Per-zoom radius: deeper levels cover a tighter area (see ZOOM_RADIUS_M).
            r = ZOOM_RADIUS_M.get(z, radius_m)
            dlat = r / 110_600.0
            dlon = r / (111_320.0 * math.cos(math.radians(lat)))
            if z not in index["zooms"]:
                index["zooms"].append(z)
            x0, y0 = deg2tile(lat + dlat, lon - dlon, z)
            x1, y1 = deg2tile(lat - dlat, lon + dlon, z)
            xs = range(int(math.floor(min(x0, x1))), int(math.floor(max(x0, x1))) + 1)
            ys = range(int(math.floor(min(y0, y1))), int(math.floor(max(y0, y1))) + 1)
            for X in xs:
                for Y in ys:
                    path = os.path.join(TILE_DIR, name, str(z), str(X))
                    os.makedirs(path, exist_ok=True)
                    fp = os.path.join(path, f"{Y}.png")
                    tiles.append({"z": z, "x": X, "y": Y})
                    if os.path.exists(fp) and os.path.getsize(fp) > 0:
                        got += 1
                        continue
                    url = src["url"].format(z=z, x=X, y=Y)
                    try:
                        req = urllib.request.Request(url, headers={
                            "User-Agent": "flytbase-hackathon/1.0 (drone traffic analytics; "
                                          "one-off cache of a single small area)"})
                        with urllib.request.urlopen(req, timeout=30) as r:
                            data = r.read()
                        with open(fp, "wb") as f:
                            f.write(data)
                        got += 1
                        time.sleep(delay)     # be gentle with a free community service
                    except Exception as e:
                        failed += 1
                        print(f"  ! {name} z{z}/{X}/{Y}: {e}")
        index["sources"][name] = {
            "attribution": src["attribution"], "count": got, "failed": failed,
            "max_zoom": src["max_zoom"],
            "zooms": sorted({t["z"] for t in tiles}),
        }
        print(f"{name}: {got} tiles cached, {failed} failed")

    with open(os.path.join(TILE_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nwrote {os.path.join(TILE_DIR, 'index.json')}")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zoom", nargs="+", type=int, default=[17, 18, 19])
    ap.add_argument("--radius-m", type=float, default=450.0)
    a = ap.parse_args()
    fetch(a.zoom, a.radius_m)
