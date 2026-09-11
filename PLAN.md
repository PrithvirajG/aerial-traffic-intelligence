# Hackathon plan — actual problem statement (supersedes CONTEXT.md's framing)

**This replaces the original Video-RAG/long-form-CCTV-Q&A plan.** That framing (retrieve-
then-ground over arbitrary text queries) is not this problem. Infra built for it (Kaggle/
Colab setup, hosted API keys, local venv) still applies; SigLIP2-as-retrieval-index and most
of the temporal-grounding-search papers do not. See `KAGGLE.md` for infra notes (still
valid), `papers/notes/` for background reading (SAM3 notes are directly relevant here; LAVAD's
VLM-captions-to-narrative technique is reusable for the final report step).

## The actual problem

Given hours of drone footage + a geolocation `.srt` telemetry file, no annotations: extract
every road user's trajectory and derive traffic insight from it. **Detection and tracking are
the foundation, not the objective** — scoring is on depth of derived insight. Gamified, in
levels.

## Architecture (five stages)

1. **Detect** — road users per frame: car, LGV, HGV, bus, motorcycle, pedestrian (Level 1's
   list). Small, dense, top-down/oblique, frequently occluded — a harder detection regime
   than street-level CCTV.
2. **Track** — associate detections into continuous per-object identities across frames,
   surviving occlusion, crossing paths, and long dwell times (stationary/queued objects).
3. **Geo-reference** — project each track from pixel coordinates into real-world metric/GPS
   coordinates, using the drone's own telemetry (`.srt`) + a camera/homography model. The
   `.srt` gives the **drone's** position/orientation per frame, not any vehicle's — this
   projection step is what turns that into a per-vehicle location.
4. **Derive insight from trajectories, analytically** — near-misses/conflicts (inter-track
   distance + closing speed), queue length in metres, turning counts by class, speed
   profiles, congestion origin vs. symptom. This is geometry/statistics on trajectory data,
   not additional model inference — this is explicitly the graded part.
5. **Narrate** — turn the numeric/geometric findings into a human-readable report. This is
   where an LLM/VLM re-enters (Gemini/NVIDIA/Groq, already wired up) — same shape as LAVAD's
   caption→narrative technique, applied to trajectory analytics instead of generic anomalies.

## Level 1 — Detection & Tracking (current)

**Requirement:** detect + track every road user; classify by car / LGV / HGV / bus /
motorcycle / pedestrian; hold stable identity through occlusion, crossing paths, long dwell
times.

### Detector — chosen and verified: `dronefreak/visdrone-yolov8x`
Tested head-to-head against a generic COCO-pretrained `yolo11n` on the same real crop from
`videos/Multi_Road_Merged.MP4` (see `videos/samples/crop_road_junction.jpg`):

| Model | Detections | Notes |
|---|---|---|
| `yolo11n` (COCO, generic) | 18 (14 car, 3 truck, 1 false-positive "boat") | **Zero** pedestrians or motorcycles detected, despite both being visible in the crop |
| `dronefreak/visdrone-yolov8x` (VisDrone-pretrained) | 93, across all required classes incl. 31 pedestrian, 11 motor | Visually verified — real cars/vans/a bus correctly boxed; VisDrone's native `van`/`truck` split maps directly to LGV/HGV candidates |

VisDrone-pretrained wins decisively, especially on the two classes the brief requires that a
generic detector completely missed. Mapping to the required taxonomy lives in
`pipeline/taxonomy.py`. VisDrone also natively fires on `tricycle`/`awning-tricycle`
(auto-rickshaws — common in this footage's setting) even though those aren't in Level 1's
named class list; mapped to `"other"` for now rather than silently dropped.

**Decision: stop evaluating further candidate models.** NVIDIA LocateAnything-3B (one-frame
exploratory test) and SAM 3.1 (gated, needs manual HF approval) were both looked at but
dropped — Level 1 is the foundation, not the objective, and is meant to be fast. Going
forward with the two VisDrone YOLO variants only (both kept — see below, no confident visual
verdict between them yet from still-frame exports). No further model searching for Level 1.

### Both VisDrone variants kept, not forced to a single choice
Couldn't get a confident visual verdict between `visdrone-yolov8x` and `visdrone-yolov11s`
from the benchmark notebook's per-frame exports (fixed large label size meant for a full
3840px frame made them too cluttered to judge). Rather than pick blind, both are run through
a proper tracked demo (`kaggle/tracked_demo/` — SAHI-tiled detection fed into `supervision`'s
ByteTrack, smaller/legible labels) and both are kept selectable in `app.py` — look at the
actual tracked clips and decide, or flag if a longer/different comparison is worth the time.

### Standard validation clips — created once, reused across levels
Stop re-deriving ad hoc samples per test (was doing this repeatedly - wasteful). Fixed set,
created once from `videos/Multi_Road_Merged.MP4` (t=0 to t=10/20/30s, `-map 0 -c copy` so both
the video *and* the mov_text telemetry subtitle stream survive the trim — a first attempt
without `-map 0` silently dropped the subtitle stream, caught by verifying with `ffprobe`
before trusting it):
- `videos/clips/clip_10s.MP4`, `clip_20s.MP4`, `clip_30s.MP4` — local, gitignored (large
  binaries, same as the master file).
- Also uploaded as their own small Kaggle Dataset, `prithvirajgotepatil/flytbase-hackathon-clips`
  (separate from the full-footage dataset — small enough that any future notebook can mount
  just this and skip the 5GB master file entirely for iteration). Hit a Windows-specific
  kaggle-cli path bug uploading this (embeds the relative `-p` path into a cache filename,
  breaks on the literal `/`) — fixed by running `kaggle datasets create` from inside the
  target directory (`-p .`) instead of pointing at it from elsewhere.

**Use these for all Level 1+ iteration/validation** going forward instead of re-slicing the
master video or the full ~305s clip each time — smaller file, faster mount, faster iteration.

### Output convention: segregated per model, video + CSV + JSON + meta.json
Every model's output lives under its own folder — `videos/samples/<model_name>/{video_h264.mp4,
detections.csv, detections.json, meta.json}` — never a shared/flat namespace between
candidates. `meta.json` (model name, weights source, tracker, frame count, unique track count)
is what `app.py` reads to label what's on screen, rather than inferring it from a filename.

**Next:** confirm at full-video scale (not just one crop) and integrate with ByteTrack.

**Kaggle infra note — mount-path mystery, now actually solved:** `tracked_demo` and
`locate_anything_test` both failed 3x with "No .MP4 found under the mounted dataset" at
`/kaggle/input/flytbase-hackathon-footage/...`, despite `dataset_sources` being correctly
recorded server-side (verified by pulling and diffing kernel metadata against the working
benchmark kernel — functionally identical). Root cause, confirmed by printing the actual
`/kaggle/input` directory tree: the real mount path was
`/kaggle/input/datasets/prithvirajgotepatil/flytbase-hackathon-footage/...` — an extra
`datasets/<username>/` segment neither notebook assumed. The earlier benchmark kernel
apparently got the older/short mount path form; these two got a newer one — looks like a
Kaggle-side platform inconsistency during a mount-path format migration, not anything in our
config. Fixed by broadening the search to `/kaggle/input/**/*.MP4` (recursive, no assumed
prefix) instead of hardcoding either path form — future notebooks should do the same rather
than assume a fixed mount path.

**Update after a quick local no-tiling test (`pipeline/quick_track_demo.py`, first 10s,
299 frames, 7,879 detections):** running the same detector on full 4K frames *without* SAHI
tiling doesn't just reduce recall on small/distant objects (zero pedestrians detected across
the whole sample, despite the detector natively supporting that class) - visually confirmed
it actively **misclassifies** vehicles. Several `truck` labels in the output sit on car-sized
bounding boxes, not genuinely large vehicles - a downscale-resolution artifact, not real
signal. This raises the stakes on tiling: it's not just a recall optimization, it's necessary
for classification correctness itself. Reinforces that the Kaggle benchmark (SAHI-tiled) is
the result to trust, not this quick local sample - the local run's class counts, especially
`HGV_candidate`, should be treated as unreliable.

### Tracker — recommendation
**ByteTrack** as the baseline (via `ultralytics`'s built-in `track()` or `supervision`) —
associates using both high- and low-confidence detections, which specifically helps recover
partially-occluded/dim detections a naive high-threshold tracker would drop. If occlusion/
crossing-path ID switches remain bad empirically, upgrade to **BoT-SORT** or **StrongSORT**
(add appearance re-identification on top of motion association) — caveat: appearance features
are less reliable on very small aerial crops, so this needs empirical validation once real
footage is available, not an assumption.

### Class taxonomy mapping — open, tracked in `pipeline/taxonomy.py`
LGV vs HGV isn't natively output by any candidate detector. Plan: detect at whatever
granularity the chosen model supports (e.g. van/truck), keep those labels literal rather than
guessing LGV/HGV, and refine the van/truck boundary using real-world vehicle length once
geo-referencing gives a size reference (a pixel-only heuristic can't reliably separate them
without one).

### What's built and verified, ahead of having footage
- `pipeline/srt_parser.py` — parses drone telemetry `.srt` into structured per-frame records
  (lat/long/altitude/gimbal angle where present). Self-tested against a synthetic sample in
  the common DJI-style layout — **the real sample's exact field layout is unconfirmed**, verify/
  adjust the moment it's available; drone telemetry formats vary by model/firmware.
- `pipeline/taxonomy.py` — class-mapping scaffold, deliberately left unfilled until a
  detector is chosen (see above) rather than guessing a mapping now.
- `requirements.txt` — added `ultralytics`, `supervision`, `lapx`, `motmetrics`.

## Confirmed from the actual footage (`videos/Multi_Road_Merged.MP4`)

- **File is ~305s (~5 min), not literally hours** — 3840x2160 @ 29.97fps, 9140 frames, H.264,
  ~4.96GB. Telemetry is muxed in as an mp4 `mov_text` subtitle stream (one entry per video
  frame), not a separate sidecar `.srt` — extracted via
  `ffmpeg -i videos/Multi_Road_Merged.MP4 -map 0:s:0 videos/Multi_Road_Merged.srt`.
- **Real telemetry field layout matches `pipeline/srt_parser.py`'s assumptions almost exactly**
  (`latitude`, `longitude`, `rel_alt`, `abs_alt`, `gb_yaw`, `gb_pitch`, `gb_roll` all present,
  same field names) — parser needs no changes, confirmed by actually running it against the
  real file's values.
- **Drone (DJI Mavic 3D) is genuinely stationary for the entire clip** — GPS unchanged to 6
  decimal places, pitch fixed at -18.5°, yaw drifts <0.5° over 5 minutes (hover noise, not
  intentional panning), checked at 5 points across the file. **Resolves the homography
  question: one fixed homography for the whole file, no per-frame reprojection needed.**
- **Oblique angle, not nadir** — gimbal pitch -18.5° (looking down at an angle, not straight
  down). Affects the geo-referencing projection math and means side-occlusion between
  vehicles is more likely than in a true top-down view.
- **Wide establishing shot, not a tight intersection crop** — the frame covers several city
  blocks: a closer city street with visible queued traffic, and a further multi-lane highway/
  expressway with a median divider in the background. Location: Pune, India
  (18.566°N 73.772°E), drone ~70m above ground (~607m absolute altitude).
- **"Multi_Road_Merged" likely means an actual road-merge point** (the highway and the local
  street both visible, or two merging lanes), not multiple spliced clips — directly relevant
  to the brief's interest in "failed merges" as an interaction metric, worth treating as
  probably deliberate footage selection rather than a coincidence.
- **Vehicles are only a handful of pixels across in the full 3840x2160 frame** — confirmed by
  visual inspection, not assumed. Cropped to the actual road region, cars/motorcycles/a bus
  become clearly resolvable and individually distinguishable; pedestrians and a roadside
  vendor cluster (occlusion source) are also visible. **This confirms tiled/ROI-based
  inference (crop to the road corridor, or SAHI-style sliced inference) is likely necessary**
  — running a detector on the full downscaled wide frame directly would likely miss most
  vehicles.

## Open items — still need to resolve
- Full level list beyond Level 1, if the organizers provide one — changes prioritization.
- Whether HGV-class vehicles (trucks) appear elsewhere in the clip — none clearly visible in
  the two sampled frames yet; worth checking the highway segment specifically.
- Actual detector choice — pending a real test (see below).
