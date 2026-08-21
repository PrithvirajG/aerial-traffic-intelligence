"""
Run this after filling in .env to verify every piece of the hackathon setup works.
Usage: python scripts/check_setup.py
"""
import os
import sys
import shutil
import subprocess

from dotenv import load_dotenv

load_dotenv()

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"


def ok(msg):
    print(f"{GREEN}[OK]{RESET} {msg}")


def fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        fail("ffmpeg not found on PATH")
        return
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    first_line = out.stdout.splitlines()[0]
    ok(f"ffmpeg found: {first_line}")


def check_nvidia_nim():
    key = os.getenv("NVIDIA_API_KEY")
    if not key or key.startswith("nvapi-...") or key == "":
        warn("NVIDIA_API_KEY not set in .env - skipping NIM test")
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        ok(f"NVIDIA NIM responded: {text!r}")
    except Exception as e:
        fail(f"NVIDIA NIM call failed: {e}")


def check_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "":
        warn("GEMINI_API_KEY not set in .env - skipping Gemini test")
        return
    try:
        from google import genai

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents="Reply with exactly: OK"
        )
        ok(f"Gemini responded: {resp.text!r}")
    except Exception as e:
        fail(f"Gemini call failed: {e}")


def check_openrouter():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or key.startswith("sk-or-v1-...") or key == "":
        warn("OPENROUTER_API_KEY not set in .env - skipping OpenRouter test")
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        resp = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning:free",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        ok(f"OpenRouter responded: {text!r}")
    except Exception as e:
        fail(
            f"OpenRouter call failed: {e}\n"
            "       (free-model IDs rotate - check https://openrouter.ai/models?max_price=0 "
            "and update the model string in check_openrouter() if this model is gone)"
        )


def check_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key or key.startswith("gsk_...") or key == "":
        warn("GROQ_API_KEY not set in .env - skipping Groq test")
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        ok(f"Groq responded: {text!r}")
    except Exception as e:
        fail(f"Groq call failed: {e}")


def check_local_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            ok(f"Local GPU detected: {out.stdout.strip()}")
        else:
            warn("nvidia-smi ran but returned nothing - no local GPU usable")
    except FileNotFoundError:
        warn("No local NVIDIA GPU detected (nvidia-smi not found). Kaggle/Colab T4 is the plan.")


def check_torch_cuda():
    try:
        import torch

        avail = torch.cuda.is_available()
        if avail:
            ok(f"torch sees CUDA: {torch.cuda.get_device_name(0)}")
        else:
            warn("torch installed but CUDA not available locally (expected - GTX 1650 4GB is too small for the real models; use Kaggle/Colab for those).")
    except ImportError:
        warn("torch not installed locally (fine - heavy models run on Kaggle/Colab, not here).")


if __name__ == "__main__":
    print("=== Hackathon setup check ===\n")
    print(f"Python: {sys.version.split()[0]}")
    check_ffmpeg()
    check_local_gpu()
    check_torch_cuda()
    print()
    print("--- Hosted API access (pipeline model providers) ---")
    check_nvidia_nim()
    check_gemini()
    check_openrouter()
    check_groq()
    print()
    print("Reminder: Kaggle GPU (30 hrs/week) and Colab GPU still need to be verified")
    print("manually in the browser - see CHECKLIST.md.")
