# Level 2 plan — fine-grained classification & kinematics

*Companion to `LEVEL1_DETECTION_TRACKING.md`. Level 2 requirements per the brief: fine-grained
classification (make/model, body type, size class, colour), license plate recognition, and
kinematics (per-object velocity/acceleration in real units). ALPR is dropped from scope — see
below. SAHI performance and tracking/occlusion fixes are shared infrastructure with Level 1
and are tracked there, not duplicated here.*

## The three asks, and why kinematics comes first

These aren't independent. Kinematics needs geo-referencing (pixel → real-world metres); size/
body class needs the *same* geo-referencing; so building that one piece of infrastructure
unlocks two of the three asks at once, plus finishes Level 1's still-open LGV/HGV split.
Build order: geo-referencing → kinematics + size/body class together → colour (independent,
parallel-able) → make/model last (lowest confidence, see below).

## License plate recognition — dropped

Confirmed empirically (crop + zoom on a real vehicle, cross-checked against actual gimbal
pitch from telemetry): `multiroad`'s vehicles are too few pixels across at altitude for
plate-reading resolution regardless of angle; `intersection`'s -63.1° near-nadir pitch means
we see roofs, not plate-facing surfaces, regardless of resolution. Both are physical capture-
geometry limits, not engineering gaps a better model would fix. Not pursuing further.

## Kinematics — the foundation

**Geo-referencing.** Both source videos are stationary-camera captures (confirmed via
telemetry — GPS/pitch unchanged across the full clip), so **one fixed homography per source
video** is enough; no per-frame reprojection needed. Two ways to calibrate it:
- *Rigorous*: match pixel points to real GPS coordinates via satellite imagery, using the
  already-parsed SRT telemetry (altitude, gimbal angle, focal length — `pipeline/srt_parser.py`)
  to build the camera model.
- *Pragmatic (recommended starting point)*: use known real-world reference distances already
  visible in the scene (standard lane width ≈3.5m, a crosswalk's known width) to scale a
  simpler planar homography. Traffic kinematics doesn't need survey-grade precision — a few %
  error in speed is fine — and this is far faster to stand up than satellite correspondence.

**Ground-contact point, not box centroid.** Use each bounding box's **bottom-centre** for
projection, not its centre. Vehicles have height; at an oblique camera angle the box centroid
sits partway up the vehicle, not on the ground plane. Using it introduces systematic position
error that gets worse for tall vehicles (buses/trucks) and worse at shallower camera angles.
Small implementation detail, meaningfully more accurate result.

**Differentiate a smoothed trajectory, not raw frame deltas.** Raw per-frame position
differences are noisy (detection jitter) — velocity/acceleration should come from a smoothed
track (e.g. Savitzky-Golay) before differentiating. This is also *why* Level 1's tracking-
stability problem matters beyond Level 1 itself: a track that fragments into short pieces
(confirmed still happening — see `LEVEL1_DETECTION_TRACKING.md` test #6) gives few-sample,
noisy kinematics. Stable IDs are a prerequisite for usable Level 2 kinematics, not a nicety.

## Fine-grained classification

- **Colour** — dominant-colour extraction (k-means on the bbox crop) on the vehicle roof,
  which is most of what's visible from directly above. Cheap, no new infrastructure, can be
  built independently of the homography work.
- **Body type / size class** — extension of the *same* homography used for kinematics:
  real-world length/width, not pixel box size, is the right signal for sedan vs. SUV vs.
  hatchback vs. van, the same way it's the right signal for Level 1's pending LGV/HGV split.
- **Make/model — scoped down, not pursued as a general feature.** Make/model recognition in
  the literature is done from street-level/near-eye-level cameras (grille shape, headlight
  design, badges) — none of that is visible from directly overhead. The one public pretrained
  candidate found, `Jordo23/vehicle-classifier` (VMMRdb-trained), has **0 downloads in the
  last month** and was trained on ground-level imagery — domain mismatch on top of near-zero
  community adoption. Not using it. Scoped-down plan: best-effort only where genuinely
  visible — readable branding/livery text on buses/commercial vehicles, visually distinctive
  vehicle types — reported honestly as best-effort, not general make/model classification.
  `COFGA` (arXiv, Feb 2025) is a real aerial-native fine-grained dataset that would be a much
  better match than VMMRdb, but no public pretrained checkpoint was found — would be a build,
  not a download, if pursued further.

## Built and self-tested so far

- `pipeline/geo_reference.py` — `GroundPlaneHomography` (fails fast if calibration points
  aren't set, rather than silently using placeholder data) + `ground_contact_point()`.
  **Calibration points are not filled in yet** — needs >= 4 real pixel↔metres
  correspondences per source video, picked from an actual frame. Not done yet.
- `pipeline/kinematics.py` — smoothed velocity/acceleration via Savgol filtering. Self-test
  caught a real numerical issue: acceleration needs a noticeably wider smoothing window than
  velocity to be trustworthy (with equal windows, tiny synthetic jitter gave a fine speed
  estimate but a badly noisy acceleration one - a genuine property of 2nd-derivative
  estimation, not a bug). Fixed with separate `speed_window`/`accel_window` parameters,
  re-verified: accel error dropped from 1.16 to 0.25 m/s² on the same synthetic test.
- `pipeline/colour.py` — k-means dominant-colour extraction, self-tested against a synthetic
  crop.

All three tested against synthetic data only so far (geometry/math correctness) - none
tested against a real vehicle crop yet, since that needs the calibration points above.

**Coordinating with a parallel session:** a forked session is working on the Level 1
BoT-SORT/occlusion fix (`kaggle/tracked_demo/`) at the same time - confirmed no file overlap.
It also created a new Kaggle dataset, `prithvirajgotepatil/flytbase-hackathon-clips-compressed`
(10s/90s clips, both sources, compressed), intended as the dataset source for Level 2 Kaggle
work.

## Cross-references

- SAHI optimization plan (ROI-cropping, overlap tuning, FP16, frame subsampling): see
  `LEVEL1_DETECTION_TRACKING.md` → Enhancement plan.
- Tracking/occlusion fix plan (ByteTrack buffer → BoT-SORT/StrongSORT → SAM 3.1 escalation):
  same file — directly affects kinematics quality, per the smoothing note above.
- Trajectory-trail visualization (derivable from saved CSV/JSON, no Kaggle reprocessing):
  same file, still not yet built.
