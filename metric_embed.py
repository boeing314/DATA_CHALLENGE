# metric_embedding_generator.py
# Generates metric embeddings using intfloat/multilingual-e5-large
# Saves: embeddings_out/metric_embeddings.npz

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
METRIC_DESC_PATH = "metric_descriptions.json"      # your metric descriptions file
OUT_DIR = Path("embeddings_out")
MODEL_NAME = "intfloat/multilingual-e5-large"
SHOW_PROGRESS = False
SEED = 42
# ----------------------------------------

np.random.seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Load metric descriptions --------
metric_desc_raw = json.load(open(METRIC_DESC_PATH, "r", encoding="utf-8"))

# Can be dict OR list
if isinstance(metric_desc_raw, dict):
    metric_map = dict(metric_desc_raw)
elif isinstance(metric_desc_raw, list):
    # List of {metric_name, description}
    metric_map = {}
    for item in metric_desc_raw:
        if not isinstance(item, dict):
            continue
        name = item.get("metric_name") or item.get("name") or item.get("metric")
        desc = item.get("description", "")
        if name:
            metric_map[name] = desc
else:
    raise ValueError("metric_descriptions must be dict or list")

metrics = list(metric_map.keys())
print(f"Loaded {len(metrics)} metrics.")

# -------- Format text for embedding --------
metric_texts = [f"[METRIC] {m} - {metric_map[m]}" for m in metrics]

# -------- Load embedding model --------
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# -------- Embed (no progress bar) --------
print("Encoding metrics...")
metric_embs = model.encode(
    metric_texts,
    convert_to_tensor=False,
    show_progress_bar=SHOW_PROGRESS
)

metric_embs = np.array(metric_embs)
print("Metric embedding shape:", metric_embs.shape)

# -------- Save --------
out_path = OUT_DIR / "metric_embeddings.npz"
np.savez_compressed(
    out_path,
    metrics=np.array(metrics, dtype=object),
    metric_texts=np.array(metric_texts, dtype=object),
    metric_embs=metric_embs
)

print("Saved metric embeddings to:", out_path)
