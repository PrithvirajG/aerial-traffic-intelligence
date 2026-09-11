"""
Level 1 sample viewer - Streamlit.

Shows tracked output (boxes, track IDs, class labels) and summary analytics for whichever
source video / duration / detector combination has been processed. Every combination lives
under its own folder - `videos/samples/<source>/<duration>/<model>/` - so there's no
ambiguity about what's on screen; the header reads `meta.json` directly.

Not every combination is necessarily processed - the selectors below only show what actually
exists. See PLAN.md for what's been run vs. still pending.

All data is pre-computed (SAHI-tiled detection + ByteTrack, run on Kaggle - see
kaggle/tracked_demo/) - this app makes no live model calls: process once, display the result.

Run: streamlit run app.py
"""
import glob
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SAMPLES_ROOT = "videos/samples"
ROAD_ZONES_PATH = "videos/road_zones/road_zones.json"

# Fixed order + validated categorical palette (dataviz skill's reference palette, slots
# 1-7) - color identifies class, never rank, and never gets reassigned when a filter
# changes which classes are visible. Past 3 series the palette's own docs require a
# secondary encoding for scatter use (not just adjacent bars/lines), so marker shape
# duplicates color here rather than relying on hue alone across 7 classes.
CLASS_ORDER = ["car", "LGV_candidate", "HGV_candidate", "bus", "motorcycle", "pedestrian", "other"]
CLASS_COLORS = {
    "car": "#2a78d6", "LGV_candidate": "#eb6834", "HGV_candidate": "#1baf7a",
    "bus": "#eda100", "motorcycle": "#e87ba4", "pedestrian": "#008300", "other": "#4a3aa7",
}
CLASS_SYMBOLS = {
    "car": "circle", "LGV_candidate": "square", "HGV_candidate": "diamond",
    "bus": "triangle-up", "motorcycle": "cross", "pedestrian": "star", "other": "x-thin",
}


def load_road_zone(source_video):
    """Cached zone box from the road-zone-detection Kaggle run, once verified - returns
    [x1, y1, x2, y2] in pixel space, or None if not pulled/verified yet."""
    if not os.path.exists(ROAD_ZONES_PATH):
        return None
    with open(ROAD_ZONES_PATH) as f:
        zones = json.load(f)
    entry = zones.get(source_video)
    if not entry or not entry.get("boxes"):
        return None
    return entry["boxes"][0]["pixel"]


def build_canvas_figure(df, frame_width, frame_height, source_video, time_cutoff, classes_shown):
    """Blank-canvas view of ground-contact points and per-track flow, in pixel space (no
    geo-referencing yet - see pipeline/geo_reference.py). One WebGL trace PER CLASS (not per
    track) - a ByteTrack combo can have 1000+ tracks, and hundreds/thousands of individual
    SVG Scatter traces is what made this laggy. Track paths are drawn as None-separated
    segments inside one Scattergl line trace per class, with a second Scattergl marker trace
    per class carrying the per-point hover info - visually identical, far fewer draw calls."""
    fig = go.Figure()

    zone_box = load_road_zone(source_video)
    if zone_box:
        zx1, zy1, zx2, zy2 = zone_box
        fig.add_shape(
            type="rect", x0=zx1, y0=zy1, x1=zx2, y1=zy2,
            line=dict(color="#898781", width=1, dash="dot"),
            fillcolor="rgba(137,135,129,0.10)", layer="below",
        )

    view = df[(df["time_s"] <= time_cutoff) & (df["taxonomy_class"].isin(classes_shown))].copy()
    # Vectorized, not a per-row .apply(ground_contact_point) - identical result (bottom-
    # centre of the box), but this rebuilds on every filter interaction so row-wise Python
    # calls were a real, avoidable chunk of the lag.
    view["gx"] = (view["x1"] + view["x2"]) / 2
    view["gy"] = view["y2"]

    for cls in CLASS_ORDER:
        cls_df = view[view["taxonomy_class"] == cls]
        if cls_df.empty:
            continue

        line_x, line_y = [], []
        marker_x, marker_y, marker_custom = [], [], []
        for track_id, track_df in cls_df.groupby("track_id"):
            track_df = track_df.sort_values("time_s")
            line_x.extend(track_df["gx"].tolist() + [None])
            line_y.extend(track_df["gy"].tolist() + [None])
            marker_x.extend(track_df["gx"].tolist())
            marker_y.extend(track_df["gy"].tolist())
            marker_custom.extend((track_id, t) for t in track_df["time_s"])

        fig.add_trace(go.Scattergl(
            x=line_x, y=line_y, mode="lines",
            line=dict(color=CLASS_COLORS[cls], width=1.5),
            name=cls, legendgroup=cls, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scattergl(
            x=marker_x, y=marker_y, mode="markers",
            marker=dict(symbol=CLASS_SYMBOLS[cls], size=5, color=CLASS_COLORS[cls]),
            name=cls, legendgroup=cls, showlegend=True,
            customdata=marker_custom,
            hovertemplate=f"{cls} · track %{{customdata[0]}}<br>t=%{{customdata[1]:.1f}}s<extra></extra>",
        ))

    fig.update_xaxes(range=[0, frame_width], visible=False)
    fig.update_yaxes(range=[frame_height, 0], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        # Transparent, not a hardcoded light hex - the previous version forced a white
        # panel that clashed hard against Streamlit's dark theme. Transparent blends into
        # whichever theme is active instead of guessing one.
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, font=dict(color="#9a9a95")),
    )
    return fig


def build_speed_by_class_figure(summary_df):
    """Per-class speed distribution, one box per class in the same fixed order/color as
    everywhere else - pixel-space (see pipeline/level2_enrich.py). Answers "how fast does
    each class typically move", not just how many of each class there are."""
    fig = go.Figure()
    for cls in CLASS_ORDER:
        cls_df = summary_df[(summary_df["taxonomy_class"] == cls) & summary_df["avg_speed_px_s"].notna()]
        if cls_df.empty:
            continue
        fig.add_trace(go.Box(
            y=cls_df["avg_speed_px_s"], name=cls, marker_color=CLASS_COLORS[cls],
            line=dict(color=CLASS_COLORS[cls]), boxpoints="all", jitter=0.4, pointpos=0,
            showlegend=False,
        ))
    fig.update_yaxes(title="avg speed (px/s)", gridcolor="#e1e0d9")
    fig.update_xaxes(categoryorder="array", categoryarray=CLASS_ORDER)
    fig.update_layout(
        plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0), height=360, font=dict(color="#52514e"),
    )
    return fig


def build_colour_table_html(summary_df, top_n=15):
    """Small HTML table with a colour swatch per row - st.dataframe has no swatch column
    type, and this data is our own pipeline output (hex codes, numbers), not untrusted
    input, so a hand-built table via unsafe_allow_html is a reasonable, simple choice."""
    rows = summary_df.sort_values("n_detections", ascending=False).head(top_n)
    html_rows = "".join(
        f"<tr><td>#{int(r.track_id)}</td><td>{r.taxonomy_class}</td>"
        f"<td><span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
        f"background:{r.dominant_color_hex};border:1px solid rgba(0,0,0,0.15);"
        f"vertical-align:middle;margin-right:6px'></span>{r.dominant_color_hex}</td>"
        f"<td>{r.avg_speed_px_s if pd.notna(r.avg_speed_px_s) else '—'}</td>"
        f"<td>{r.duration_s}</td></tr>"
        for r in rows.itertuples()
    )
    return (
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='text-align:left;border-bottom:1px solid #c3c2b7'>"
        "<th>Track</th><th>Class</th><th>Colour</th><th>Avg speed (px/s)</th><th>Duration (s)</th>"
        "</tr></thead><tbody>" + html_rows + "</tbody></table>"
    )


def discover():
    """Returns {source: {duration_label: {model: entry}}} from every meta.json found."""
    tree = {}
    for meta_path in sorted(glob.glob(f"{SAMPLES_ROOT}/*/*/*/meta.json")):
        model_dir = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        source = meta.get("source_video", "unknown")
        duration = f"{meta.get('duration_s', '?')}s"
        model = meta["model"]
        tree.setdefault(source, {}).setdefault(duration, {})[model] = {
            "dir": model_dir, "meta": meta,
        }
    return tree


st.set_page_config(page_title="Level 1 — Detection & Tracking", layout="wide")
st.title("Level 1 — Detection & Tracking")

tree = discover()
if not tree:
    st.error(
        "No tracked output found under `videos/samples/<source>/<duration>/<model>/meta.json`. "
        "Run `kaggle/tracked_demo/`, pull its output, and drop it in place."
    )
    st.stop()

col_source, col_duration, col_model = st.columns(3)
with col_source:
    source_choice = st.selectbox("Source video", list(tree.keys()))
with col_duration:
    duration_choice = st.selectbox("Duration", list(tree[source_choice].keys()))
with col_model:
    model_choice = st.selectbox("Detector", list(tree[source_choice][duration_choice].keys()))

entry = tree[source_choice][duration_choice][model_choice]
meta = entry["meta"]

if meta.get("known_issue"):
    st.warning(f"⚠️ {meta['known_issue']}")

st.caption(
    f"**{source_choice} / {duration_choice} / {model_choice}** · "
    f"weights `{meta.get('weights_source', '?')}` · tracker {meta.get('tracker', '?')} · "
    f"conf≥{meta.get('confidence_threshold', '?')} · "
    f"{meta.get('sliced_inference', {}).get('tool', '?')} "
    f"({meta.get('sliced_inference', {}).get('slice_size', '?')}px slices) · "
    f"{meta.get('frames_processed', '?')} frames @ {meta.get('fps', 0):.1f} fps"
)

video_path = f"{entry['dir']}/video_h264.mp4"
trails_path = f"{entry['dir']}/trails_h264.mp4"
csv_path = f"{entry['dir']}/detections.csv"
df = pd.read_csv(csv_path)

col_video, col_canvas = st.columns(2)

with col_video:
    st.caption("**Video**")
    if os.path.exists(trails_path):
        view = st.radio("View", ["Tracked (boxes)", "Trajectory trails"], horizontal=True)
        chosen = trails_path if view == "Trajectory trails" else video_path
    else:
        chosen = video_path
    if os.path.exists(chosen):
        st.video(chosen)
    else:
        st.info(f"Video not found at `{chosen}`.")

with col_canvas:
    st.caption("**Points & flow — blank canvas**")
    max_t = float(df["time_s"].max()) if len(df) else 0.0
    classes_present = [c for c in CLASS_ORDER if c in set(df["taxonomy_class"])]
    filt_classes, filt_time = st.columns([1, 2])
    with filt_classes:
        classes_shown = st.multiselect(
            "Classes", classes_present, default=classes_present, key="canvas_classes",
        )
    with filt_time:
        time_cutoff = st.slider(
            "Show up to time (s)", 0.0, max_t, max_t, step=0.1, key="canvas_time",
        )
    frame_w = meta.get("frame_width") or float(df["x2"].max())
    frame_h = meta.get("frame_height") or float(df["y2"].max())
    st.plotly_chart(
        build_canvas_figure(df, frame_w, frame_h, source_choice, time_cutoff, classes_shown),
        use_container_width=True,
    )
    if not os.path.exists(ROAD_ZONES_PATH):
        st.caption(
            "No road-zone overlay yet — pending NVIDIA LocateAnything-3B verification "
            "(see LEVEL1_DETECTION_TRACKING.md)."
        )

st.divider()
col_stats, col_by_class = st.columns(2)

with col_stats:
    st.metric("Frames processed", meta.get("frames_processed", df["frame"].nunique()))
    st.metric("Unique tracked objects", meta.get("unique_tracks", df["track_id"].nunique()))

with col_by_class:
    st.write("**Unique tracks by class**")
    by_class = df.groupby("taxonomy_class")["track_id"].nunique().sort_values(ascending=False)
    st.bar_chart(by_class)

st.divider()
summary_path = f"{entry['dir']}/tracks_summary.csv"
if os.path.exists(summary_path):
    summary_df = pd.read_csv(summary_path)
    st.subheader("Level 2 — colour, speed & duration per tracked object")
    st.caption(
        "Speed is pixel-space (px/s) — no geo-referencing calibration yet, see "
        "pipeline/geo_reference.py. Colour is each track's dominant roof colour from its "
        "largest, clearest crop. See pipeline/level2_enrich.py."
    )
    col_speed, col_colours = st.columns([3, 2])
    with col_speed:
        st.plotly_chart(build_speed_by_class_figure(summary_df), use_container_width=True)
    with col_colours:
        st.markdown(build_colour_table_html(summary_df), unsafe_allow_html=True)
else:
    st.caption(
        "Level 2 enrichment (colour + kinematics) not generated yet for this combo — run "
        "`pipeline/level2_enrich.py <source_clip> <model_dir>`."
    )

st.divider()
st.subheader("Detections over time")
per_second = df.groupby(["time_s", "taxonomy_class"]).size().unstack(fill_value=0)
st.line_chart(per_second)

with st.expander("Raw detection rows"):
    st.dataframe(df, use_container_width=True)

st.divider()
with st.expander("Known gaps - flag for review, not yet resolved"):
    st.markdown("""
- **LGV vs HGV not resolved** - shown as VisDrone's native `van`/`truck` split
  (`LGV_candidate`/`HGV_candidate`), not a final classification. Needs real-world vehicle size
  from geo-referencing to split reliably - see `pipeline/taxonomy.py`.
- **ByteTrack vs BoT-SORT** - BoT-SORT (appearance ReID) cut unique-track count ~80% vs
  ByteTrack on the same clip (332 vs 1,692) - strong evidence it's the better choice for
  occlusion/crossing-path robustness. Not every source/duration/model combo has been re-run
  with BoT-SORT yet - the selector above only shows combos actually processed.
- **No geo-referencing applied** - all coordinates are pixel-space, not real-world
  metres/GPS. Not needed for Level 1 itself, but the LGV/HGV split above depends on it.
""")
