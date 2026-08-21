# Setup Checklist — FlytBase Visual Intelligence Hackathon

Based on the organizers' "Hackathon Prerequisites & Resources Guide" PDF. Their required
setup is lighter than CONTEXT.md's original conda/local-GPU plan — the guide expects most
teams to use **Kaggle/Colab T4 notebooks + hosted APIs**, not a fully local pipeline. This
checklist reconciles the two: local machine does orchestration + lightweight tooling only,
Kaggle does the heavy model work, and NVIDIA NIM/Gemini/OpenRouter/Groq are a swappable
fallback chain of hosted model providers, each best suited to a different pipeline role
(see §4) rather than one being "the" choice over the others.

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
- [done] `scripts/check_setup.py` written — one command verifies ffmpeg, local GPU, and all
  four hosted API providers once you've pasted in keys (skips any that are missing).

## 3. Free GPU runtimes (guide Part 1) — **[YOU]**, ~10 min, must be done in browser

⚠️ **Known blocker (2026-08-21): Kaggle phone verification is stuck** — your phone number is
already registered to some other Kaggle account you can't identify (Kaggle dedupes phone
numbers globally across accounts to stop GPU-quota farming). Almost certainly an old account
of yours. Recovery steps, in order of speed:
1. On kaggle.com's login page, try **"Continue with Google"** with every Google email you own
   (personal, work, this hackathon's email, any old college/coursera email) — Kaggle accounts
   are very often created silently via Google OAuth and forgotten.
2. Try **"Sign in with GitHub"** the same way if you've ever linked it.
3. Try **"Forgot password"** with likely email guesses — a reset email confirms the account exists.
4. Kaggle support ticket (kaggle.com/contact) to release the number — realistically too slow
   for tomorrow, file it but don't depend on it.
5. **Given this, Colab is now the primary GPU plan for tomorrow**, not the backup — it only
   needs Google sign-in, no phone verification, so it sidesteps the blocker entirely. Keep
   trying Kaggle in parallel since 30 GPU-hrs/week is nicer, but don't block on it.
6. **Test NVIDIA NIM's phone verification tonight, not tomorrow** — the prereqs guide requires
   phone verification there too, and if it's the same underlying dedup problem, better to find
   out now while there's time to sort it out.

- [ ] **Kaggle** (secondary if blocked): create account at kaggle.com → Settings → Phone
  Verification (GPU stays locked until you do this) → Create → Notebook → Session options →
  Accelerator → **GPU T4 x2** → run `import torch; print(torch.cuda.is_available())` in a
  cell, confirm `True`. 30 GPU-hours/week if it works.
- [ ] **Colab** (primary for now): open colab.research.google.com, sign in, New notebook →
  Runtime → Change runtime type → **T4 GPU** → Save → run the same
  `torch.cuda.is_available()` check.

I can prepare a ready-to-upload starter notebook (imports, model downloads, a smoke test)
for whichever one you pick — say the word once your account/GPU access is confirmed.

## 4. Hosted model APIs — **[YOU]** to get keys, then automated

Four free, no-card options, each best at a different pipeline role. None of these compete with
Claude Code's own subscription — they're only used by *our pipeline's* code (captioning, VQA,
answer generation), never for the coding agent itself.

| Provider | Native video? | Free tier | Best role in our pipeline |
|---|---|---|---|
| **NVIDIA NIM** | No (frames/images) | ~40 req/min, generous | Bulk per-chunk captioning if Kaggle setup has friction |
| **Gemini Flash** | **Yes** — raw clip upload | Rate limits opaque, check AI Studio | Query-time "cheat mode": hand it a retrieved clip + query directly if the real pipeline is behind schedule |
| **OpenRouter** | No (frames/images) | Flaky — free-model IDs rotate/rate-limit | Model diversity, A/B testing captioners, extra fallback |
| **Groq** | No (frames/images) | Free, very fast inference | Low-latency demo-time answer generation |

- [ ] **NVIDIA NIM**: build.nvidia.com → create account → verify phone → open any vision
  model at build.nvidia.com/explore/vision → "Get API Key" → copy it (shown once) →
  paste into `.env` as `NVIDIA_API_KEY`.
- [ ] **Gemini API**: aistudio.google.com → sign in → Get API key → Create API key → paste
  into `.env` as `GEMINI_API_KEY`. Free tier = Flash/Flash-Lite only (not Pro) — build against
  `gemini-2.5-flash` from the start.
- [ ] **OpenRouter** *(optional but recommended)*: openrouter.ai → sign up → API Keys → Create
  key → paste into `.env` as `OPENROUTER_API_KEY`. This is a **separate use** from the guide's
  "OpenRouter for Claude Code" section — you don't need that part since Claude Code already
  has your subscription. This key is only for our own pipeline's model calls.
- [ ] **Groq** *(optional but recommended)*: console.groq.com → sign up → API Keys → Create
  API Key → paste into `.env` as `GROQ_API_KEY`.
- [ ] After keys are in `.env`, run: `python scripts/check_setup.py` — should print `[OK]` for
  every provider you filled in (missing ones are skipped with a `[WARN]`, not an error).

### Note on data sovereignty (from CONTEXT.md §6)
CONTEXT.md flagged checking with organizers before sending footage to a hosted API. Since the
**organizers themselves** hand out Gemini and NVIDIA NIM as first-class, recommended options
in their own prerequisites guide, that concern is resolved for this event — hosted APIs are
sanctioned. The "kept it local" pitch angle from CONTEXT.md still works as a *design choice*
(and is worth keeping for the demo narrative), just not as a compliance requirement.

## 5. Before-you-arrive checklist (guide's list, plus OpenRouter/Groq)
- [ ] Kaggle account created and phone-verified
- [ ] Colab opened at least once
- [ ] NVIDIA API key generated and saved (in `.env`)
- [ ] Gemini API key generated and saved (in `.env`)
- [ ] OpenRouter API key generated and saved (in `.env`) — optional, recommended
- [ ] Groq API key generated and saved (in `.env`) — optional, recommended
- [ ] One test call made successfully with each (`python scripts/check_setup.py` covers all
      four API providers in one shot; Kaggle/Colab GPU checks are separate manual steps in-browser)

## What's left for you to do manually, right now
1. Kaggle signup + phone verification + GPU test (§3)
2. Colab GPU test (§3, lower priority)
3. NVIDIA NIM key (§4)
4. Gemini key (§4)
5. OpenRouter key (§4, optional)
6. Groq key (§4, optional)
7. Paste keys into `.env` (copy from `.env.example`), then run
   `python scripts/check_setup.py` and confirm `[OK]` for whichever you filled in

Everything else on this list is already done.
