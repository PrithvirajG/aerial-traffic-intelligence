# Level 2 — Fine-grained classification & kinematics: decisions, tests, and enhancement plan

*One-page reference, same format as `LEVEL1_DETECTION_TRACKING.md`. Planning detail in
`LEVEL2_PLAN.md`; full logs in `PLAN.md`.*

## Why each decision was made

**Kinematics built before classification.** The three Level 2 asks aren't independent: real-unit
velocity/acceleration needs pixel→metres geo-referencing, and so does real-world size/body
class. Building that one piece unlocks two asks at once *and* closes Level 1's still-open
LGV/HGV split. Colour is genuinely independent and was built in parallel.

**License plate recognition dropped — physical limit, not an engineering gap.** Confirmed
empirically against real telemetry: both sources hover at **~70.5 m** (`rel_alt`). On
`multiroad` (−18.5° pitch) vehicles are ~27 px wide, far below plate-reading resolution. On
`intersection` (−63.1°, near-nadir) we see roofs, not plate-facing surfaces. Neither is fixable
by a better model at this capture geometry.

**Smooth first, then differentiate.** Raw frame-to-frame position deltas are dominated by
bbox-edge jitter. Velocity/acceleration come from a Savitzky-Golay-smoothed trajectory. A
self-test caught a real numerical issue here (see Tests #2) — acceleration, a 2nd derivative,
needs a much wider smoothing window than velocity.

**Ground-contact point, not bbox centroid.** Every projection uses the bbox **bottom-centre**.
A vehicle has height; at oblique angles the centroid sits partway up the vehicle, off the
ground plane, which injects systematic position error that worsens for tall vehicles and
shallower pitch. Used consistently across geo-referencing, trails, and the dashboard.

**Geo-referencing from telemetry, not manual calibration.** Original plan was to click ≥4
reference points per scene and estimate real distances (lane width ≈3.5 m). Inspecting the real
`.srt` showed that's unnecessary — it carries everything a pinhole camera model needs:

```
[focal_len: 24.00] [rel_alt: 70.469] [gb_pitch: -18.5] [gb_yaw: -83.4] [gb_roll: 0.0]
```

Focal length + frame size + altitude + gimbal orientation fully determine the mapping, so the
model is derived from **measured data rather than eyeballed correspondences**. Both sources are
stationary hovers, so one telemetry sample describes the whole clip — no per-frame reprojection.

**Make/model deliberately not attempted.** Make/model recognition in the literature relies on
grille, headlight, and badge detail from near-eye-level cameras — none of it visible from
overhead. The one public pretrained candidate found (`Jordo23/vehicle-classifier`, VMMRdb) has
**0 downloads/month** and is trained on ground-level imagery: domain mismatch on top of
near-zero adoption. Reported as a known limitation rather than shipped as a weak feature.
`COFGA` (aerial-native fine-grained dataset) would be the right basis, but has no public
checkpoint — a build, not a download.

## Tests performed and metrics

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Colour extraction (k-means, k=3) on a synthetic 70%-red crop | Recovered BGR≈(22,22,199) | Method sound; built on real crops next |
| 2 | Kinematics self-test, synthetic constant-acceleration track (true 2.0 m/s²) with 0.02 m jitter | Equal smoothing windows: speed error 0.07 m/s (fine) but **accel error 1.16 m/s²** (unusable). Split windows (speed 9, accel 21): **accel error 0.25 m/s²** | **Real numerical bug found and fixed.** 2nd derivatives need a much wider window — a genuine property of the estimator, not a coding error |
| 3 | Level 2 enrichment on 4 real combos (both scenes × both trackers, 10s) | See coverage table below | Works end-to-end on real data; colour sampled for **100%** of tracks |
| 4 | Telemetry camera-model self-tests (nadir symmetry, horizon rejection, oblique scale growth, vectorized-vs-scalar agreement) | All pass. Nadir: 1000 px off-axis at 100 m ⇒ exactly 37.5 m, as geometry predicts. Sky pixels correctly return *no ground intersection* rather than a bogus distance | Rotation convention and intrinsics verified against closed-form expectations |
| 5 | **Empirical scale validation** — projected 694 real tracks' ground footprints and compared against known vehicle sizes | **Pedestrians 0.63 m (multiroad) / 0.79 m (intersection)** vs expected 0.3–0.9 m ✓. Motorcycles 0.69 m / 1.40 m. Cars measured 4.68 m / 3.25 m vs a car's 1.85 m *width* | **Scale is trustworthy; the discrepancy is a measurement artefact, not a scale error** — see below |

**Why the "car too wide" result is not a scale error.** Two independent pieces of evidence: (1)
pedestrians — whose footprint is roughly circular, so bbox width ≈ true width at *any*
orientation — come out correct in **both** scenes, which wouldn't happen if focal length or
altitude were misinterpreted; (2) the implied correction factors differ between scenes (0.40 vs
0.57), whereas a genuine focal/altitude error would be one constant. The real cause is that an
axis-aligned bbox around an elongated, rotated object spans its **diagonal**, not its width —
4.68 m on `multiroad` is closely tracking real car *length* (~4.5 m). `SCALE_CORRECTION` is
therefore left at **1.0**; the fix for size-class work is oriented boxes, not rescaling.

**Level 2 coverage by combo (10s clips, real runs):**

| Scene | Tracker | Tracks | With kinematics | Median track life |
|---|---|---|---|---|
| multiroad | ByteTrack | 1,505 | 541 (35.9%) | 0.53 s |
| multiroad | **BoT-SORT** | 332 | **256 (77.1%)** | **3.67 s** |
| intersection | ByteTrack | 1,081 | 396 (36.6%) | 0.50 s |
| intersection | **BoT-SORT** | 362 | **202 (55.8%)** | **1.23 s** |

**This table is the strongest Level 1 → Level 2 evidence we have.** Kinematics needs enough
consecutive samples to smooth and differentiate; fragmented tracks are too short to qualify.
Switching tracker more than doubles kinematics coverage (35.9% → 77.1% on multiroad) and raises
median track life ~7×. Tracking stability isn't a Level 1 nicety — it's a hard prerequisite for
Level 2 having anything to measure.

**Colour distribution (694 BoT-SORT tracks, both scenes):** grey 48.1%, silver 36.7%, white
6.6%, black 3.5%, orange 2.7%, red 1.6%, remainder <1%. The heavy grey/silver skew is the real
signal, not a bug — from altitude you see sunlit metallic roofs, which are largely desaturated.
The classifier therefore tests **saturation before hue**; hue-first bucketing would scatter
near-grey roofs into arbitrary colour names.

## Real-world units — wired in and validated

The telemetry projector is now wired into `pipeline/level2_enrich.py`, so all four 10s combos
carry `speed_kmh`, `accel_mps2`, and ground positions (`east_m`/`north_m`). **The speed
validation is the strongest evidence that the camera model is correct:**

| Class | Median speed (multiroad) | Independent sanity check |
|---|---|---|
| pedestrian | **5.8 km/h** | normal walking pace ≈ 5 km/h ✓ |
| motorcycle | 24.9 km/h | plausible urban two-wheeler ✓ |
| car | 31.7 km/h | plausible urban arterial ✓ |
| bus | 44.8 km/h | plausible ✓ |

Four different classes each land in their own physically sensible range, from a single camera
model with no per-class tuning. The **pedestrian figure is the decisive one** — walking speed is
a universal constant, so recovering 5.8 km/h from drone telemetry and pixel tracks alone
confirms the scale independently of any vehicle assumption.

**Per-track maxima are NOT usable** and are excluded from every aggregate: an ID switch makes a
track appear to teleport, producing values like 1331 km/h for a car. All reported statistics are
medians/percentiles over per-detection samples, after dropping samples above 120 km/h. The
rejection rate is reported as a quality signal (0.03% on `intersection`, 2.4% on `multiroad` —
the higher rate on the wide oblique shot is expected, since a pixel of jitter covers more ground
there).

## Known gaps (real, not yet fixed)
- **`rel_alt` is height above *takeoff*, not above the road.** Any elevation difference between
  the takeoff spot and the road scales all distances proportionally. Pedestrian validation
  suggests it's small here, but it's an unquantified assumption.
- **Flat-ground assumption.** Fine for these road scenes; wrong on slopes, flyovers, or
  multi-level junctions.
- **Far-field accuracy degrades sharply at low pitch.** Measured on `multiroad` (−18.5°): ground
  scale runs **0.042 m/px near-field vs 0.076 m/px far-field** — one pixel of jitter is worth
  ~1.8× more ground error at the far end of the same frame. Far-field speeds are coarser and
  shouldn't be compared like-for-like against near-field ones.
- **LGV/HGV still unresolved**, now for a narrower reason than before: geo-referencing exists,
  but axis-aligned bboxes don't give reliable vehicle *length* (see the diagonal artefact above).
- **~23–44% of tracks have no kinematics at all** (too short to smooth). Reported honestly in
  the UI as `—` rather than back-filled with noisy raw differencing.

## Enhancement plan

**Immediate, high value:**
1. **Oriented bounding boxes** (or PCA on each track's ground path to recover heading, then
   measure the footprint along/across that heading) → unlocks the real vehicle length that
   LGV/HGV needs, and fixes the diagonal artefact in Test #5.
2. **Enrich the 90s clips** — 9× the observation window makes every derived statistic more
   stable. Pure post-processing, no re-detection needed.
3. **Absolute GPS per track** — telemetry carries drone lat/long, so ground offsets in metres
   convert directly to lat/long, making output overlayable on a real map.

**Medium:**
4. Body/size class from real footprint dimensions, once (1) lands.
5. Per-vehicle-length occupancy, replacing the assumed ~5 m mean used in Level 3's density
   figures.

*(Per-track heading and turn detection, listed here previously, is now built — it became the
basis of Level 3's turning movements and O-D matrix. See `LEVEL3_AGGREGATE_INSIGHT.md`.)*

**Deliberately not pursued:** make/model classification (no viable pretrained option for this
viewing angle — see above), and ALPR (physically impossible at this capture geometry).

## What's built

| File | Purpose | Verified |
|---|---|---|
| `pipeline/geo_reference_telemetry.py` | Telemetry-driven pixel→metres projector | 4 self-tests incl. closed-form nadir check |
| `pipeline/validate_scale.py` | Empirical scale check vs known vehicle sizes | Run on 694 real tracks |
| `pipeline/kinematics.py` | Savgol-smoothed speed/acceleration | Self-test; caught + fixed a real accel bug |
| `pipeline/colour.py` | k-means dominant roof colour | Self-test + 100% coverage on real runs |
| `pipeline/level2_enrich.py` | Per-track summary (colour + kinematics) from a Kaggle run | Run on all 4 real combos |
| `pipeline/trajectory_trails.py` | Per-track fading trail video, local CPU only | Visually confirmed |
| `pipeline/geo_reference.py` | Original manual-homography path | Superseded by telemetry version for these sources |
| `web/` (JS dashboard) | Video + live-synced flow canvas, Level 2 filters, analytics | Data/serving verified; see `web/README.md` |
| `app.py` (Streamlit) | Original dashboard, still functional | Rendering verified via Streamlit test harness |
