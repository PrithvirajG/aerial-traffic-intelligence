# Second-tier papers — triage only (abstract-level, NOT full notes)

Per Task 1 only the six priority papers get full notes. These eight are downloaded and text-
extracted in `papers/raw/`; this file records whether each is worth opening tomorrow.

## 🔥 Read this one — it may change tomorrow's plan
### 2607.08075 — UAV-OVVIS / **AeroTrack** (Jul 2026, CAS Xi'an + TeleAI)
Training-free open-vocab video instance segmentation **for UAV/aerial footage**, which is
half our expected dataset. Code: https://github.com/Dmygithub/AeroTrack

Their framework is a **direct, cheaper answer to the SAM 3 cost problem** in note 2511.16719:
- **Recognizer**: open-vocab detection every **τ frames** (pluggable: Grounding DINO,
  YOLO-World, **or SAM 3**) — *not* every frame.
- **Segmenter**: SAM 2 / SAM 3 propagates masks *within* each `[k, kτ)` segment only, and
  **resets its memory at every τ boundary** — bounding memory and drift.
- **LIA**: output-level association stitches per-segment local IDs into video-global IDs.

Why this matters to us: it is the identical two-rate structure we want (cheap dense pass +
expensive sparse pass), validated on exactly our domain, and it names the aerial-specific
failure modes we should expect — **small dense objects, top-down viewpoint, joint camera+target
motion, frequent entry/exit of the frame**. They also state plainly that per-frame detection or
whole-video propagation is infeasible for practical UAV processing. If the 11:00 drop is
drone-heavy, read §Method properly and set τ as a swept parameter.
Benchmark: AeroVIS, 9 UAV categories, 8,279 trajectories.

## Worth a skim if time allows
### 2511.05489 — TimeSearch-R (ByteDance + PKU, Nov 2025)
The current SOTA that supersedes T\* on its own benchmark. Reformulates temporal search as
interleaved text-video "thinking" trained with RL (GRPO + Completeness Self-Verification).
SOTA on LongVideoBench, +4.1% over Qwen2.5-VL base.
**Verdict: cite as SOTA context, do not implement — it requires SFT cold-start plus RL
training, which is impossible in 7 hours.** Its value to us is rhetorical: it confirms the
field has moved to *adaptive, iterative* search rather than one-shot retrieval, which is the
direction our re-ranking loop points in.

### 2603.13306 — Benchmarking Compact VLMs for Clip-Level Surveillance Anomaly Detection
(J. Imaging, Mar 2026, MTUCI Moscow). Closest paper to our actual deployment regime.
Standardised protocol comparing PEFT-adapted compact VLMs vs training-free VLM pipelines vs
weakly-supervised baselines on CCTV, reporting **accuracy, precision, recall, F1, ROC-AUC and
average per-clip latency together**.
Two things to take: (1) **their metric set is the one we should print** — it already pairs
quality with per-clip latency, which is our "measurable performance" requirement; (2) they
find adaptation **reduces prompt sensitivity**, i.e. training-free pipelines like ours are
prompt-fragile. That is a known weakness we should measure rather than hide — a
prompt-variance row in our results table would be unusually honest and is cheap to produce.

## Reference / background — open only if a specific question comes up
- **2502.14786 SigLIP 2** — our embedding model. Open when we need input resolution, patch
  size, or the NaFlex variant details.
- **2504.13181 Perception Encoder** — the backbone inside SAM 3. Relevant only if we consider
  using PE embeddings instead of SigLIP 2. Note its headline claim: the best visual embeddings
  are in *intermediate* layers, not at the output — worth remembering if PE embeddings
  underperform out of the box.
- **2502.12558 MomentSeeker** — benchmark for long-video moment retrieval. Useful as a source
  of query *categories* when we write `eval/queries.json`, so our 20 queries span the same
  task taxonomy a benchmark would use.
- **2606.13141 Rethinking RAG in Long Videos: What to Retrieve and How to Use It?** (Jun 2026)
  Title suggests it directly ablates retrieval-unit choice (frame vs clip vs caption) — the
  exact design decision in our `ingest.py`. **Promote this to a real read if we have a spare
  30 minutes tonight**; it is the one second-tier paper that could save us a wrong turn.
- **2412.01095 VERA** — explainable VAD via verbalized learning. Overlaps LAVAD; only relevant
  if we want to output *why* something was flagged.
