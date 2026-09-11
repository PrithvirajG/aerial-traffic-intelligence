# Level 3 — Aggregate Insight: decisions, tests, and enhancement plan

*One-page reference, same format as `LEVEL1_DETECTION_TRACKING.md` and `LEVEL2_APPROACH.md`.
Implementation: `pipeline/level3_aggregate.py`. Output: `aggregate_insight.json` per combo,
surfaced in the JS dashboard's "Aggregate insight" section.*

## What the brief asked for, and where each item is answered

| Brief item | Where it lands | Status |
|---|---|---|
| Classified counts by movement and interval | `intervals[]` — per 2s window: arrivals, flow (veh/h), concurrent count, by-class breakdown, turn split | ✅ |
| Turning movements at intersections | `totals.turn_counts` + per-corridor `turn_counts` — through/left/right/u-turn | ✅ |
| Directional and lane volumes on open road | `corridors[]` → per-approach volume, share, and `lanes[]` within each | ✅ |
| Origin–destination distributions | `od_matrix` — approach→exit compass pairs, ranked | ✅ |
| Speed profiles across the segment | `speed_profile[]` — median + p85 per distance bin along the road axis | ✅ |
| Where speeding concentrates | `hotspots[]` — the bin with the highest p85 prevailing speed | ✅ |
| Lane volume and modal split by lane | `corridors[].lanes[].modal_split` | ✅ |
| Queue lengths | `queues[]` — spatial extent in metres of stopped (<4 km/h) groups per interval | ✅ |
| Density, occupancy, flow–density | `fundamental[]` — flow vs density per interval, speed as third variable | ✅ |

Level 3 is only meaningful because Level 2's geo-referencing landed first: queue *length*,
lane *offset*, and *density* are all distances. In pixels they'd be uncomparable between the
two scenes; in metres they're real traffic engineering quantities.

## Why each decision was made

**Robust statistics everywhere — never per-track maxima.** An ID switch makes a track appear to
teleport, which produced a car measured at **1331 km/h** in the real Level 2 output. Every
aggregate here is a median or percentile over *per-detection* samples, after dropping samples
above 120 km/h as physically impossible. The rejection rate is reported rather than hidden,
because it's a tracking-quality signal in its own right (0.03% on `intersection` vs **2.4%** on
`multiroad` — higher on the wide oblique shot, exactly as the Level 2 far-field analysis
predicts).

**p85 speed, not mean speed.** The 85th-percentile speed is the standard traffic-engineering
measure of prevailing speed, and it's what speed-limit and enforcement decisions are based on —
means are dragged down by stopped vehicles and up by outliers.

**Direction from a PCA travel axis, not image axes.** The road runs at an arbitrary angle
through the frame, so "distance along the road" and "lateral lane offset" need the scene's own
axes. The principal component of vehicle *displacement* (not position) gives the travel
direction. Displacements are sign-aligned before PCA — otherwise opposing traffic flows cancel
out and the axis collapses. This is scene-agnostic: no hand-drawn lane polygons per video.

**Lanes are grouped per directional corridor, not scene-wide — a bug this caught.** The first
implementation clustered lateral offset across the whole scene and reported **"16 lanes"** on
`multiroad`. That's obviously wrong, and the cause was real: `multiroad` is a wide shot spanning
~315 m containing *several roads at different angles*, so offset from one global axis conflates
separate roads with lanes of one road. Fixed by bucketing movements by direction of travel
first (which distinguishes one road from another), then clustering lateral offset *inside* each
corridor (which distinguishes lanes of the same road) — which is also how a traffic engineer
analyses an approach. Result dropped to a credible 8 corridors / 11 lanes.

**Lane counts are observations, not surveyed geometry.** Lanes come from where traffic actually
ran, so a corridor with too few vehicles reports "too sparse to resolve" rather than inventing a
lane. Corridors below 4% of traffic are dropped entirely — with sparse data a handful of tracks
otherwise looks like a road that isn't one.

**Level 3 deliberately ignores the dashboard's Level 2 filters.** A lane volume or flow–density
point computed from a colour-filtered subset would be a misleading traffic statistic. The panel
states this on screen rather than silently applying filter state.

## Tests performed and results

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Run on all 4 geo-referenced combos | 4/4 produced output; the 2 combos without telemetry-derived speed were **skipped with an explicit reason**, not silently emptied | Pipeline handles partial data honestly |
| 2 | **Speed profile through the intersection** | Median speed by distance: **16.8 → 13.5 → 6.1 → 2.0 km/h** approaching the junction | **Strong physical validation** — deceleration into a junction is exactly what must happen; the profile recovers it from tracks alone |
| 3 | Lane/corridor plausibility (initial) | "16 lanes" on `multiroad` | **Real bug found** — global lateral offset conflates separate roads (fixed, see above) |
| 4 | Lane/corridor plausibility (after fix) | `multiroad`: 8 corridors, dominant SE 23% / E 23% / W 16%. `intersection`: dominant E 24% / SE 22% / W 19% | Credible: a wide multi-road shot vs. one main arterial crossing |
| 5 | Turn-type distribution | `intersection`: through 60, left 20, right 18, u-turn 2 | Through-dominant with real turning volumes — expected for a signalised crossing |
| 6 | Queue detection | `intersection`: peak **122.2 m queue, 54 stopped vehicles at t=8s** (20 cars, 28 motorcycles, 3 HGV, 4 LGV, 1 bus) | Detects a genuine standing queue, and explains the scene's low 4.5 km/h median speed |
| 7 | Cross-scene comparison | `multiroad` median 18.4 km/h / p85 46.8 vs `intersection` median 4.5 / p85 17.3 | Free-flow arterial vs congested junction — the two scenes separate exactly as their footage looks |

**The single most demo-ready finding:** the `intersection` clip is a signal-controlled junction
with a **122 m standing queue of 54 vehicles**, of which **28 are motorcycles** — a modal split
that matters for the scene's location, and precisely the kind of insight that's invisible in raw
detections but obvious once aggregated.

## Known gaps (real, not yet fixed)

- **Compass bucketing spreads noise across 8 directions.** Small corridors (4–7% share) are
  partly jitter rather than distinct roads, and some "u-turns" are likely ID switches rather
  than real manoeuvres. The dominant 3 corridors per scene are trustworthy; the tail is not.
- **10s is a short window for flow statistics.** Flow rates extrapolated to veh/h from a 2s
  interval are arithmetically correct but statistically thin. The 90s clip would give far more
  stable numbers — it just needs a Level 2 enrichment pass first.
- **Occupancy is approximated**, using an assumed ~5 m mean vehicle length rather than measured
  per-vehicle length (which needs the oriented-bounding-box work from Level 2's gap list).
- **No signal-phase awareness.** Queue and delay statistics would be far more useful segmented
  by signal cycle, which could be *inferred* from the stop/go pattern rather than needing
  external signal data.
- **O-D is screenline-free**, using entry/exit heading rather than defined gates, so a track that
  starts mid-frame (appearing from behind the tree occlusion noted in Level 1) contributes a
  slightly wrong origin.

## Enhancement plan

1. **Enrich + aggregate the 90s clips** — the single biggest quality win, no new code: 9× the
   observation window makes every flow, queue, and O-D figure meaningfully more stable.
2. **Infer signal phase from the stop/go pattern**, then report queue length, delay, and
   saturation flow *per cycle* — the standard way intersection performance is actually assessed.
3. **Explicit screenlines / O-D gates** per scene (derivable once, reused — the camera is fixed),
   replacing heading-based O-D and fixing the mid-frame-origin problem.
4. **Per-lane time-space diagrams** — with real positions along the road axis and real time, this
   is directly plottable and is how shockwaves and stop-and-go waves are diagnosed.
5. **Measured vehicle length** (from Level 2's oriented-box work) → true occupancy, and the
   LGV/HGV split that's been pending since Level 1.

## What's built

| File | Purpose | Verified |
|---|---|---|
| `pipeline/level3_aggregate.py` | All Level 3 aggregates → `aggregate_insight.json` | Run on 4 real combos; found + fixed the lane-conflation bug |
| `pipeline/level2_enrich.py` | Now telemetry-aware: writes `speed_kmh`, `accel_mps2`, `east_m`/`north_m` | Validated against walking speed (5.8 km/h) |
| `pipeline/geo_reference_telemetry.py` | Telemetry → ground-plane projector | 4 self-tests incl. closed-form nadir check |
| `web/` (JS dashboard) | "Aggregate insight" section: speed profile, flow–density, turns/O-D, queues, corridors, intervals | JS parses clean, all IDs resolve, data verified over HTTP |
