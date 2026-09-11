"""
Confirms SigLIP2 actually loads and runs on this machine's CPU, and reports real specs
(param count, download size, load time, single-embedding latency) instead of recalled ones.
Usage: .venv/Scripts/python scripts/check_siglip_local.py
"""
import time

CKPT = "google/siglip2-so400m-patch14-384"

t0 = time.time()
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

print(f"Loading {CKPT} on CPU (first run downloads the weights)...")
model = AutoModel.from_pretrained(CKPT).eval()
processor = AutoProcessor.from_pretrained(CKPT)
load_time = time.time() - t0

n_params = sum(p.numel() for p in model.parameters())
n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

print(f"\nParameters: {n_params/1e6:.1f}M")
print(f"In-memory weight size: {n_bytes/1e9:.2f} GB (fp32)")
print(f"Load time (incl. any download): {load_time:.1f}s")

image = load_image(
    "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg"
)
inputs = processor(images=[image], return_tensors="pt")

t1 = time.time()
with torch.no_grad():
    image_output = model.get_image_features(**inputs)
embed_time = time.time() - t1

embedding = getattr(image_output, "pooler_output", image_output)
print(f"\nSingle-image embedding shape: {tuple(embedding.shape)}")
print(f"Single-image embedding latency (CPU, warm): {embed_time:.2f}s")

text_inputs = processor(text=["a photo of a dog"], padding="max_length", return_tensors="pt")
t2 = time.time()
with torch.no_grad():
    text_output = model.get_text_features(**text_inputs)
text_time = time.time() - t2
text_embedding = getattr(text_output, "pooler_output", text_output)
print(f"Single-query text embedding shape: {tuple(text_embedding.shape)}")
print(f"Single-query text embedding latency (CPU, warm): {text_time:.2f}s")
