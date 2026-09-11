# Drone traffic analytics dashboard (JS)

Standalone frontend alternative to the Streamlit app (`app.py`), which is left untouched and
still works. Built for the Level 2 demo: two synced windows on top, analytics underneath.

## Run

```bash
python web/export_data.py     # reads videos/samples/, writes web/data/ + web/clips/
python web/serve.py           # http://localhost:8600
```

Re-run `export_data.py` whenever a new combo lands in `videos/samples/`.

## Layout

**Top-left — source video** with a live canvas overlay. Independent toggles for `Boxes`,
`Trails`, `IDs`, so you can show any combination (including both boxes *and* trails, which the
Streamlit version couldn't do — it could only switch between two pre-rendered videos).

**Top-right — points & flow**, a blank canvas showing only ground-contact points and paths in
the same field of view. `Points` / `Trails` / `Full history` toggles; `Full history` draws every
filtered track's whole path so far, which is the "where does traffic actually flow" view.
Hovering a point shows that track's class, colour, speed, acceleration and lifetime.

**Live sync** — both panels are driven by the `<video>` element's own clock (`currentTime` →
frame index) inside one `requestAnimationFrame` loop, so they cannot drift from the footage.
This is the reason for a JS frontend at all: Streamlit's `st.video` exposes no playback clock
to the server, so a server-rendered second panel can't track playback even in principle.

**Colour & filter by category** — pick one category (Vehicle class · Colour · Speed band ·
Acceleration band · Direction · Turn) and it drives *both* the filter chips and the colour of
every drawn object, in both panels. Only the active category filters, so nothing is ever hidden
by a control you can't currently see; each category remembers its own selection when you switch
back. A category with no data in the current run is disabled rather than shown empty.

Each category uses the palette type that matches its data, which is the point of making it
switchable rather than always colouring by class:

| Category | Palette | Why |
|---|---|---|
| Vehicle class, Turn | categorical — distinct hues | unordered identities |
| Speed band, Accel band | **green → red**, low to high | ordered magnitude, traffic-light semantics |
| Direction | cyclic — hue wraps the wheel | a bearing wraps, so opposite headings get opposite colours |
| Colour | the swatch *is* the datum | a red car's chip is red |

**About the green→red ramp** (`#007f35 → #5d8900 → #c67300 → #f96050`): green-to-red is the
classic red-green colourblindness failure case, so this ramp was **constructed in OKLCH with
strictly monotonic lightness** (L = 0.52 → 0.685 in even steps) rather than picked by eye. That
is what makes it work: the low→high ordering is carried by *brightness* as well as hue, so it
still reads correctly under deuteranopia, protanopia, or full monochromacy. Measured against
the dark chart surface:

- contrast 3.39 / 4.18 / 4.85 / 5.65 — all clear 3:1, which small dots and thin lines need
- adjacent-pair CVD ΔE 10.8 / 18.0 / 10.1 — all clear the ≥8 floor
- endpoints ΔE 33.6 normal, 47.9 under deuteranopia — unmistakable either way

Hand-picked green→red attempts were checked first and failed badly: a naive green/yellow/orange/red
ramp measured **CVD ΔE 0.8–3.1 on the orange→red pair** — effectively identical — because hue
alone was carrying the ordering. Don't re-tune these hexes by eye; re-run the constructed
search if they need to change. Speed and acceleration share the ramp deliberately: only one
category is ever active, so the two can't appear together and be confused.

**Arrows** (right panel) — an arrowhead at each object's current position, pointing along its
recent heading. Heading is taken over the last few trail points, not two adjacent ones:
consecutive bbox centres jitter enough that a one-frame delta makes the arrow spin on the spot.
Objects that are effectively stationary get no arrow rather than one pointing at noise.

Filtering is computed once per change into a set of passing track IDs; the render loop only
tests set membership, so it stays cheap during playback.

**Bottom — analytics**: KPI tiles, tracks-by-class, speed distribution, objects-on-screen over
time (with a marker following playback), and a per-track detail table.

## Notes / honest limits

- **Speed and acceleration are pixel-space** (px/s, px/s²), not real-world units —
  `pipeline/geo_reference.py`'s calibration points aren't filled in yet. Band edges come from
  the real distribution across the processed 10s combos, so they're comparative, not physical.
- **Tracks shorter than the smoothing window have no kinematics** (see
  `pipeline/kinematics.py`) — they show as `—` and are hidden once a speed filter is narrowed.
  The count is reported under the filters.
- **Colour skews heavily to grey/silver.** That's the real signal, not a bug: from altitude
  you see sunlit metallic roofs, which are mostly desaturated. The classifier checks
  saturation before hue for exactly this reason.
- **Direction has a `stationary` bucket, and that's deliberate.** A vehicle that never travels
  3 m has no heading. On the congested `intersection` clip that's 193 of 362 tracks — the
  standing queue itself — so it's bucketed and coloured neutral grey rather than dropped from
  the view. Genuinely *undeterminable* tracks (too few samples to say anything) stay excluded,
  and the count is reported under the chips. The two cases are kept distinct on purpose.
- **Direction is a true compass bearing only where telemetry geo-referencing exists.** For a run
  without it (e.g. the older `visdrone-yolov8x` combo) the buckets are image-relative and the UI
  says so explicitly rather than implying north.
- **Clips are transcoded, not copied.** The source clips are 10-bit H.264 (`High 10`), which
  no browser decodes — a plain copy gives a black video panel. `export_data.py` forces 8-bit
  `yuv420p` and 1080p. Overlay alignment is aspect-ratio-based, so downscaling doesn't shift
  boxes.
- `web/serve.py` implements HTTP Range requests, which stdlib `http.server` omits and video
  seeking requires.
- No build step and no dependencies — there's no Node toolchain on this machine, and a canvas
  redrawn every frame gains nothing from a virtual DOM.
