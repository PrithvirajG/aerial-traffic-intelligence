"""
Conversions between the local East/North/Up metre frame used by geo_reference_telemetry.py
and absolute WGS84 lat/lon.

The local frame's origin is the point on the ground directly beneath the drone, which the SRT
telemetry gives us as an absolute lat/lon - so turning local metres into real coordinates is
just an ENU->geodetic conversion about that origin.

Uses the local tangent-plane approximation (per-latitude metres-per-degree), not a full
ellipsoidal geodesic. Over the ~350 m scene extent the error from this is well under a
centimetre, i.e. several orders of magnitude below the drone's own GPS accuracy (~2-5 m) -
so a heavier geodesic library would add dependency and precision that the input data cannot
justify. See LEVEL4_SPATIAL_GROUNDING.md.
"""
import math

# WGS84
_A = 6378137.0                 # semi-major axis, metres
_F = 1.0 / 298.257223563       # flattening
_E2 = _F * (2 - _F)            # first eccentricity squared


def metres_per_degree(lat_deg):
    """Metres per degree of latitude and of longitude at a given latitude.

    Both vary with latitude - longitude strongly (cos lat), latitude slightly (ellipsoid
    curvature). Using a single 111320 m/deg constant for both is a common shortcut that would
    introduce ~5% longitude error at this site's latitude, i.e. metres of drift across the
    scene - enough to matter for lane-level work.
    """
    lat = math.radians(lat_deg)
    sin2 = math.sin(lat) ** 2
    # Meridional and normal radii of curvature.
    m = _A * (1 - _E2) / (1 - _E2 * sin2) ** 1.5
    n = _A / math.sqrt(1 - _E2 * sin2)
    return math.radians(1) * m, math.radians(1) * n * math.cos(lat)


def enu_to_wgs84(east_m, north_m, origin_lat, origin_lon):
    """Local ENU offsets (metres) -> absolute (lat, lon) degrees."""
    m_lat, m_lon = metres_per_degree(origin_lat)
    return origin_lat + north_m / m_lat, origin_lon + east_m / m_lon


def wgs84_to_enu(lat, lon, origin_lat, origin_lon):
    """Absolute (lat, lon) -> local ENU offsets (east_m, north_m) about an origin."""
    m_lat, m_lon = metres_per_degree(origin_lat)
    return (lon - origin_lon) * m_lon, (lat - origin_lat) * m_lat


if __name__ == "__main__":
    # Site of both source clips (identical hover point for the two scenes - confirmed from
    # telemetry, they differ only in gimbal yaw).
    LAT, LON = 18.566225, 73.771845

    m_lat, m_lon = metres_per_degree(LAT)
    print(f"At lat {LAT}: {m_lat:.1f} m/deg latitude, {m_lon:.1f} m/deg longitude")
    assert 110_500 < m_lat < 110_800, m_lat
    # cos(18.57 deg) ~= 0.948, so longitude degrees are shorter than latitude ones here.
    assert 105_000 < m_lon < 106_000, m_lon

    # Round-trip must return the input to sub-millimetre precision.
    for e, n in [(0, 0), (100, -250), (-315.5, 42.25), (1000, 1000)]:
        lat, lon = enu_to_wgs84(e, n, LAT, LON)
        e2, n2 = wgs84_to_enu(lat, lon, LAT, LON)
        assert abs(e - e2) < 1e-6 and abs(n - n2) < 1e-6, (e, n, e2, n2)

    # A known offset must land at a sane distance: 100 m north is ~0.0009 deg of latitude.
    lat, lon = enu_to_wgs84(0, 100, LAT, LON)
    assert abs((lat - LAT) - 100 / m_lat) < 1e-12
    print(f"100 m north of origin -> {lat:.6f}, {lon:.6f}")
    print(f"100 m east  of origin -> {enu_to_wgs84(100, 0, LAT, LON)[1]:.6f} lon")
    print("Self-test OK - round-trip exact, per-latitude scaling applied.")
