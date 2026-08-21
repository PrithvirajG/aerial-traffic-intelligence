# Setup Checklist — FlytBase Visual Intelligence Hackathon

Based on the organizers' "Hackathon Prerequisites & Resources Guide" PDF. Their required
setup is lighter than CONTEXT.md's original conda/local-GPU plan — the guide expects most
teams to use **Kaggle/Colab T4 notebooks + hosted APIs**, not a fully local pipeline. This
checklist reconciles the two: local machine does orchestration + lightweight tooling only,
Kaggle does the heavy model work, and Gemini/NVIDIA NIM are fallback/supplementary APIs.

Legend: **[done]** = Claude did this already · **[YOU]** = only you can do this (account
creation, phone verification, browser signup) · **[ ]** = not started.

## 1. Coding agent (guide's "Coding Setup Guide")
- [done] Claude Code is already installed and working, authenticated with your own
  subscription (Opus 5) — confirmed `ANTHROPIC_BASE_URL`/`ANTHROPIC_MODEL` are unset, meaning
  you're on the default Anthropic auth, not routed through OpenRouter.
- **You do not need the OpenRouter/free-model section of the guide at all.** That's only for
  people without a Claude subscription. Skip it.

## 2. Local machine — orchestration environment
- [done] Python 3.11.9 installed via winget (`Python.Python.3.11`), confirmed on PATH for new
  terminal sessions (registry-checked: `Python311` precedes the WindowsApps stub in your User
  PATH, so plain `python` will resolve correctly in any *new* terminal/Claude Code session —
  this current session needed a manual PATH prepend since it started before the install).
- [done] pip upgraded; lightweight packages installed: `requests`, `python-dotenv`,
  `google-genai`, `openai` (used generically for any OpenAI-compatible endpoint, incl. NVIDIA
  NIM), `pandas`, `pyarrow`, `pillow`.
- [done] ffmpeg confirmed present (v9.0, full build, **with NVENC/NVDEC/CUDA + built-in
  Whisper filter** — useful, means no separate ffmpeg install needed even for ASR later).
- [done] git confirmed present (2.55.0); repo initialized, first commit made (papers + notes).
- [done] Local GPU detected: **GTX 1650, 4 GB VRAM**. Enough for ffmpeg hardware decode and
  maybe a tiny int4 model in a pinch, but **not** the primary compute — matches the guide's
  framing (Kaggle/Colab T4 or hosted APIs are the real workhorses). Did *not* install `torch`
  locally on purpose — no reason to pull a multi-GB CUDA wheel onto a 4GB card we won't
  train/infer heavy models on.
- [done] `.env.example` created (copy to `.env`, fill in keys — `.env` is gitignored).
- [done] `scripts/check_setup.py` written — one command verifies ffmpeg, local GPU, and both
  hosted APIs once you've pasted in keys.

## 3. Free GPU runtimes (guide Part 1) — **[YOU]**, ~10 min, must be done in browser
- [ ] **Kaggle**: create account at kaggle.com → Settings → Phone Verification (GPU stays
  locked until you do this) → Create → Notebook → Session options → Accelerator →
  **GPU T4 x2** → run `import torch; print(torch.cuda.is_available())` in a cell, confirm
  `True`. You get 30 GPU-hours/week; keep an eye on the sidebar counter tomorrow.
- [ ] **Colab**: open colab.research.google.com, sign in, New notebook → Runtime → Change
  runtime type → **T4 GPU** → Save → run the same `torch.cuda.is_available()` check.
  (Only needed as a backup if Kaggle's queue is bad tomorrow — Kaggle's 30 free GPU-hrs/week
  is the primary plan since Colab free-tier availability is less predictable.)

I can prepare a ready-to-upload starter notebook (imports, model downloads, a smoke test)
for whichever one you pick — say the word once your account/GPU access is confirmed.

## 4. Hosted model APIs (guide Part 2) — **[YOU]** to get keys, then automated
- [ ] **NVIDIA NIM**: build.nvidia.com → create account → verify phone → open any vision
  model at build.nvidia.com/explore/vision → "Get API Key" → copy it (shown once) →
  paste into `.env` as `NVIDIA_API_KEY`. Free tier ≈ 40 req/min, OpenAI-SDK compatible.
- [ ] **Gemini API**: aistudio.google.com → sign in → Get API key → Create API key → paste
  into `.env` as `GEMINI_API_KEY`. Free tier = Flash/Flash-Lite only (not Pro) — build against
  `gemini-2.5-flash` from the start. Note: **Gemini accepts video/images directly**, so it's
  a genuinely useful fallback path if the local/Kaggle pipeline is behind schedule at 4 PM —
  point it straight at a clip for a quick-and-dirty answer.
- [ ] After both keys are in `.env`, run: `python scripts/check_setup.py` — should print
  `[OK]` for both NVIDIA NIM and Gemini.

### Note on data sovereignty (from CONTEXT.md §6)
CONTEXT.md flagged checking with organizers before sending footage to a hosted API. Since the
**organizers themselves** hand out Gemini and NVIDIA NIM as first-class, recommended options
in their own prerequisites guide, that concern is resolved for this event — hosted APIs are
sanctioned. The "kept it local" pitch angle from CONTEXT.md still works as a *design choice*
(and is worth keeping for the demo narrative), just not as a compliance requirement.

## 5. Before-you-arrive checklist (verbatim from the guide, tracked here)
- [ ] Kaggle account created and phone-verified
- [ ] Colab opened at least once
- [ ] NVIDIA API key generated and saved (in `.env`)
- [ ] Gemini API key generated and saved (in `.env`)
- [ ] One test call made successfully with each (`python scripts/check_setup.py` covers NVIDIA
      + Gemini in one shot; Kaggle/Colab GPU checks are separate manual steps in-browser)

## What's left for you to do manually, right now
1. Kaggle signup + phone verification + GPU test (§3)
2. Colab GPU test (§3, lower priority)
3. NVIDIA NIM key (§4)
4. Gemini key (§4)
5. Paste both keys into `.env` (copy from `.env.example`), then run
   `python scripts/check_setup.py` and confirm both show `[OK]`

Everything else on this list is already done.
