from sentence_transformers import SentenceTransformer, util
import numpy as np
import random
from tqdm import tqdm
import copy

def generate_synthetic_samples(
    train_data,
    metric_desc_map,
    n_synth_per_example=1,
    bottom_k=3,
    low_thresh=0.45,
    med_thresh=0.45,
    low_score_range=(0.0, 2.0),
    mid_score_range=(3.0, 6.0),
    model_name="intfloat/multilingual-e5-large",
    seed=42
):
    """
    Notebook-friendly synthetic data generator.
    No argparse. No CLI. Pure function.
    """

    random.seed(seed)
    np.random.seed(seed)

    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)

    # ---------------------------------------------------------
    # Precompute metric embeddings
    # ---------------------------------------------------------
    metrics = list(metric_desc_map.keys())

    def format_metric_text(metric_name):
        return f"[METRIC] {metric_name} - {metric_desc_map.get(metric_name,'')}"
    
    metric_texts = [format_metric_text(m) for m in metrics]
    metric_embs = model.encode(metric_texts, convert_to_tensor=True, show_progress_bar=False)
    metric_to_idx = {m:i for i,m in enumerate(metrics)}

    augmented = copy.deepcopy(train_data)
    synthetic_count = 0

    # ---------------------------------------------------------
    # Iterate through examples
    # ---------------------------------------------------------
    for ex in tqdm(train_data, desc="Generating synthetic"):

        orig_metric = ex["metric_name"]
        if orig_metric not in metric_to_idx:
            continue  # skip unknown metrics

        orig_idx = metric_to_idx[orig_metric]
        orig_emb = metric_embs[orig_idx]

        # similarity of orig metric to all metrics
        sims = util.cos_sim(orig_emb, metric_embs)[0].cpu().numpy()
        sims[orig_idx] = 1.0  # force original to max similar so not in bottom-k

        sorted_idx = list(np.argsort(sims))  # ascending = least similar first
        bottom_candidates = sorted_idx[:min(bottom_k, len(sorted_idx))]

        # Create synthetic samples
        for _ in range(n_synth_per_example):
            cand_idx = random.choice(bottom_candidates)
            new_metric = metrics[cand_idx]
            sim_val = float(sims[cand_idx])

            # decide score based on similarity
            if sim_val <= low_thresh:
                score_val = round(random.uniform(*low_score_range), 1)
            elif sim_val <= med_thresh:
                score_val = round(random.uniform(*mid_score_range), 1)
            else:
                continue  # too similar → skip

            new_ex = copy.deepcopy(ex)
            new_ex["metric_name"] = new_metric
            new_ex["score"] = str(score_val)
            new_ex["_synthetic_info"] = {
                "orig_metric": orig_metric,
                "new_metric": new_metric,
                "similarity": sim_val,
                "assigned_score": score_val
            }

            augmented.append(new_ex)
            synthetic_count += 1

    print("Synthetic generated:", synthetic_count)
    print("Final dataset size:", len(augmented))

    return augmented
import json

train_data = json.load(open("train_data.json"))
metric_desc = json.load(open("metric_descriptions.json"))
augmented = generate_synthetic_samples(
    train_data,
    metric_desc,
    n_synth_per_example=2,
    bottom_k=3,
    low_thresh=0.811,
    med_thresh=0.83
)
json.dump(augmented, open("train_augmented.json", "w"), ensure_ascii=False, indent=2)
