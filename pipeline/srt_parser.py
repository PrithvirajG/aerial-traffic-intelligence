"""
Parses a drone's telemetry .srt sidecar file into per-frame records.

This is the drone's OWN position/orientation per frame (GPS lat/long, altitude,
sometimes gimbal angle) - NOT any tracked vehicle's location. Turning a tracked
vehicle's pixel position into a real-world coordinate is a separate step (geo-
referencing, via homography or per-frame projection using this telemetry) - see
PLAN.md.

Field layout varies by drone model/firmware (DJI Mavic/Air/Mini/Matrice all differ
slightly, other manufacturers differ more). Written generically against regex
patterns for the commonly-seen fields rather than one exact model's fixed layout,
so it degrades gracefully (unmatched fields stay None) instead of hard-failing on
an unexpected format. Verify field names against the real sample the moment it's
available - this is a first-pass built ahead of having real footage, not a
guarantee it matches this event's exact drone/firmware.
"""
import re
from dataclasses import dataclass


@dataclass
class TelemetryFrame:
    index: int
    start_s: float
    end_s: float
    timestamp: str | None
    latitude: float | None
    longitude: float | None
    rel_altitude: float | None
    abs_altitude: float | None
    gimbal_pitch: float | None
    gimbal_yaw: float | None
    gimbal_roll: float | None
    raw_text: str


_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_FIELD_PATTERNS = {
    "latitude": r"lat(?:itude)?\s*:?\s*(-?\d+\.\d+)",
    "longitude": r"lon(?:gitude)?\s*:?\s*(-?\d+\.\d+)",
    "rel_altitude": r"rel_?alt(?:itude)?\s*:?\s*(-?\d+\.?\d*)",
    "abs_altitude": r"abs_?alt(?:itude)?\s*:?\s*(-?\d+\.?\d*)",
    "gimbal_pitch": r"gb_?pitch\s*:?\s*(-?\d+\.?\d*)",
    "gimbal_yaw": r"gb_?yaw\s*:?\s*(-?\d+\.?\d*)",
    "gimbal_roll": r"gb_?roll\s*:?\s*(-?\d+\.?\d*)",
}
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?")


def _to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _extract_field(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_srt(text: str) -> list[TelemetryFrame]:
    """Parses raw .srt file content into a list of TelemetryFrame, one per subtitle block."""
    blocks = re.split(r"\n\s*\n", text.strip())
    frames = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        time_match = _TIME_RE.search(lines[1])
        if not time_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = time_match.groups()
        start_s = _to_seconds(h1, m1, s1, ms1)
        end_s = _to_seconds(h2, m2, s2, ms2)

        body = "\n".join(lines[2:])
        ts_match = _TIMESTAMP_RE.search(body)

        frames.append(
            TelemetryFrame(
                index=index,
                start_s=start_s,
                end_s=end_s,
                timestamp=ts_match.group(0) if ts_match else None,
                latitude=_extract_field(_FIELD_PATTERNS["latitude"], body),
                longitude=_extract_field(_FIELD_PATTERNS["longitude"], body),
                rel_altitude=_extract_field(_FIELD_PATTERNS["rel_altitude"], body),
                abs_altitude=_extract_field(_FIELD_PATTERNS["abs_altitude"], body),
                gimbal_pitch=_extract_field(_FIELD_PATTERNS["gimbal_pitch"], body),
                gimbal_yaw=_extract_field(_FIELD_PATTERNS["gimbal_yaw"], body),
                gimbal_roll=_extract_field(_FIELD_PATTERNS["gimbal_roll"], body),
                raw_text=body,
            )
        )
    return frames


def parse_srt_file(path: str) -> list[TelemetryFrame]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_srt(f.read())


if __name__ == "__main__":
    # Synthetic sample in the common DJI-style layout - exercises the parser without
    # needing real footage. Replace with a real sample the moment one is available.
    _SAMPLE = """1
00:00:00,000 --> 00:00:00,033
<font size="28">SrtCnt : 1, DiffTime : 33ms
2026-08-22 09:00:00,123
[iso : 100] [shutter : 1/500.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 240] [latitude: 51.500000] [longitude: -0.140000] [rel_alt: 50.000 abs_alt: 100.000] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">SrtCnt : 2, DiffTime : 33ms
2026-08-22 09:00:00,156
[iso : 100] [shutter : 1/500.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 240] [latitude: 51.500010] [longitude: -0.140005] [rel_alt: 50.010 abs_alt: 100.010] </font>
"""
    parsed = parse_srt(_SAMPLE)
    assert len(parsed) == 2, f"expected 2 frames, got {len(parsed)}"
    assert parsed[0].latitude == 51.5
    assert parsed[0].longitude == -0.14
    assert parsed[0].rel_altitude == 50.0
    assert parsed[1].index == 2
    print(f"Self-test OK - parsed {len(parsed)} telemetry frames from synthetic sample.")
    for f in parsed:
        print(f)
