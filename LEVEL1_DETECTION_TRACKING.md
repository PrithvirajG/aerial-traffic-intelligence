# Level 1 — Detection & Tracking: decisions, tests, and enhancement plan

*One-page reference for how we got here and what's next. Full detail/logs in `PLAN.md`.*

## Why each decision was made

**Detector.** The brief needs car/LGV/HGV/bus/motorcycle/pedestrian from oblique aerial footage —
small, dense, frequently occluded objects, no ground-truth labels for this exact footage.
Generic COCO-trained detectors are trained on street-level imagery; VisDrone-trained ones are
trained on the same aerial/drone regime we actually have. We tested this belief rather than
assume it (see Tests below) — it held.

**Both VisDrone-YOLO variants kept, not one chosen.** `visdrone-yolov8x` vs `visdrone-yolov11s`
differ 3x in speed with no clear accuracy loser. The only comparison tool we had (SAHI's
still-frame exports) used a fixed large label size meant for a full 3840px frame — illegible
once cropped or viewed dense. Rather than force a coin-flip decision from an unreadable image,
we built a proper tracked-video demo for both and left the actual call to visual review of
real, legible output.

**SAHI tiling is mandatory, not optional.** A local no-tiling test on full 4K frames didn't just
miss small objects (zero pedestrians in ~8,000 rows) — it **misclassified** cars as trucks,
visually confirmed. Downscaling a 4K frame to a detector's normal input size destroys the
resolution needed for correct classification, not just recall. This is why every real test
since has gone through SAHI.

**ByteTrack chosen as the tracking baseline** for its use of both high- and low-confidence
detections (recovers partially-occluded detections a naive high-threshold tracker drops), with
BoT-SORT/StrongSORT flagged from the start as the likely upgrade path once occlusion proved to
be a real problem (it has — see Known Gaps).

## Tests performed and metrics (chronological)

| # | Test | Frames | Speed | Result | Verdict |
|---|---|---|---|---|---|
| 1 | Single crop, `yolo11n` (COCO) vs `visdrone-yolov8x` | 1 crop | — | COCO: 18 dets, 0 pedestrian/motorcycle. VisDrone: 93 dets, 31 pedestrian, 11 motor | VisDrone decisively better |
| 2 | Kaggle benchmark, SAHI-tiled, 3 models, 18 sample frames (T4) | 18 | `yolo11x`: 1.95s/f · `v8x`: 2.38s/f · `v11s`: 0.80s/f | `yolo11x`: hallucinated classes (train/boat/airplane); `v8x`/`v11s`: full class coverage | Dropped `yolo11x`; kept both VisDrone variants |
| 3 | Local, no SAHI tiling, `v8x`, 10s multiroad clip | 299 | — | 7,879 dets, 134 tracks, **0 pedestrians**, visually confirmed car→truck misclassification | Confirmed tiling is required for correctness, not just recall |
| 4 | Kaggle tracked demo v1, SAHI+ByteTrack, both models, 10s multiroad, conf=0.15 | 300 | `v8x`: 2.14s/f · `v11s`: 0.74s/f | `v8x`: 94,963 dets/**1,692 tracks** · `v11s`: 88,840 dets/**1,682 tracks** | Track-ID churn — implausible track count for 10s; confidence threshold too low |
| 5 | Kaggle tracked demo v2, `v11s` only, 10s intersection, conf=**0.35** | 300 | 1.28s/f | 46,984 dets/**880 tracks** | Improvement (~46% fewer tracks), not a clean controlled comparison (different scene) |
| 6 | Kaggle tracked demo v3, `v11s` only, 90s multiroad, conf=0.35 | 2,698 | ~0.67s/f (1806s total) | 623,200 dets/**12,338 tracks** (≈137/s vs. 169/s at conf=0.15 — only ~19% reduction) | **Confidence-threshold fix alone is insufficient.** Same scene/model, only threshold changed — this is the clean comparison test 4/5 lacked. Avg track lifespan ≈1.7s, too short to be real vehicles. Confirms the tracker itself needs the appearance-based fix, not more threshold tuning. |
| 7 | Kaggle tracked demo v10, `v11s` + **BoT-SORT** (ReID, `track_buffer=60`), 10s multiroad, conf=0.35 | 302 | ~0.84s/f (253s total) | 37,526 dets/**332 unique tracks** (vs. 1,692 for ByteTrack on the same 10s clip — **~80% reduction**) | **Major improvement, strong signal the ReID/appearance fix is working** — 332 tracks in 10s of dense traffic is a plausible real number. Not a fully clean comparison (confidence threshold also differs from the original ByteTrack@0.15 baseline), but the earlier threshold-only test (row 6) got just 19% reduction, so the bulk of this drop is very likely the tracker swap itself. **Visually confirmed** — pulled the actual output video, track IDs are low/stable (single-double digits like #18, #24, #38, #44), boxes correctly on real vehicles, no obvious false positives at multiple zoom levels. |

**Footage facts that shaped all of the above:** both source videos (`multiroad`,
`intersection`) are stationary-camera captures (one fixed homography will suffice for
geo-referencing, no per-frame reprojection needed) but differ in geometry — `multiroad` is a
wide oblique shot (-18.5° pitch, small/distant vehicles, needs tiling for recall);
`intersection` is a near-nadir shot (-63.1° pitch, larger/closer vehicles, but shows roofs
almost exclusively — confirmed empirically that license plates are not visible at this
altitude/angle in either video, hence dropping ALPR from scope).

## Known gaps (real, not yet fixed)

- **Occlusion re-association is motion-only, no appearance matching.** A vehicle that changes
  direction while occluded (e.g., turning behind the tree at the intersection's cross-road)
  will likely get a new track ID rather than recovering its old one — a real, identified
  failure mode, not hypothetical. See enhancement plan below.
- **LGV/HGV split** still uses VisDrone's native van/truck labels as candidates, not a final
  classification — needs real-world vehicle length from geo-referencing to resolve.
- **No geo-referencing built yet** — blocks kinematics, size/body class, and the LGV/HGV split
  above. All three depend on the same one piece of infrastructure.

## Enhancement plan

**Object detection / SAHI performance**, roughly in order of expected value:
1. Restrict tiling to the road-containing region per source video (fixed camera → fixed ROI) —
   likely the single biggest win; large fractions of the wide shot are buildings/sky/never
   contain vehicles.
2. Reduce slice overlap 0.2 → 0.1 (matches commonly-benchmarked SAHI configs; we're currently
   computing more redundant overlap than typical).
3. FP16 inference on the T4 — low-risk, usually free speed.
4. Frame subsampling — not required to match the video's native 30fps (this is offline batch
   analytics, not a live system; confirmed as a standing principle going forward) — directly
   cuts total compute at the whole-video level.
5. *(Stretch)* Adaptive slice count (per the 2026 "ASAHI" paper) instead of a fixed grid — real
   recent research direction, more engineering than the above four, not urgent.

**Tracking / occlusion robustness:**
1. Increase ByteTrack's lost-track buffer (one parameter, free to try).
2. Swap to BoT-SORT/StrongSORT via `boxmot` (adds appearance re-identification — same
   plug-in shape as current `sv.ByteTrack()`, directly targets the tree-occlusion case).
3. *(Escalation only)* SAM 3.1 video mask propagation — memory-based identity, gated on HF
   approval, heavier compute; hold in reserve.

**Fine-grained classification (Level 2, noted here since it shares detection infra):**
colour (k-means on crop, cheap) and body/size class (real-world dimensions from the same
homography) are solid plans. Make/model has no strong pretrained option for this aerial
viewing angle (the one public candidate found, `Jordo23/vehicle-classifier`, has 0 downloads/
month and was trained on ground-level imagery — domain mismatch on top of low adoption) —
treat as a known limitation, not a committed feature, unless real fine-tuning effort is
invested.

**Trajectory-trail visualization — built and verified.** `pipeline/trajectory_trails.py`:
pure local CPU post-processing (no Kaggle involved), draws each track's last ~1.5s of motion
as a fading, per-track-colored polyline onto the clean source clip, ending in a dot at the
current position. Uses bbox bottom-centre (ground-contact point), not centroid - the same
convention `PLAN.md` specifies for the upcoming geo-referencing step. Run against the
BoT-SORT output (`multiroad`/10s) - visually confirmed: distinct colors per track, trails
correctly follow the road's curve, dots land on real vehicles. Some trails show minor
zigzag/jitter noise from frame-to-frame box-edge variation - real, not a script bug; the
Savitzky-Golay smoothing already planned for kinematics would clean this up too, so treat it
as a shared fix for later rather than something to patch in the trail script itself.
