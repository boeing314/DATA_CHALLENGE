# notebook_embedding_pipeline.py
# Notebook-friendly: will not recompute embeddings if .npz files exist.
# Requires: pip install sentence-transformers numpy pandas scikit-learn

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------- CONFIG (edit as needed) ----------
TRAIN_PATH = "train_augmented.json"
TEST_PATH = "test_data.json"
METRIC_PATH = "metric_descriptions.json"
OUT_DIR = Path("embeddings_out")
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64    # increase if you have GPU memory, reduce if OOM
TOP_K = 5
N_SENT = 3
OVERWRITE = True    # set True to force recompute embeddings even if files exist
# ----------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_metric_dict(metric_obj):
    if isinstance(metric_obj, dict):
        metric_map = dict(metric_obj)
    elif isinstance(metric_obj, list):
        metric_map = {}
        for item in metric_obj:
            if not isinstance(item, dict):
                continue
            key = item.get("metric_name") or item.get("name") or item.get("metric")
            desc = item.get("description") or item.get("desc") or item.get("description_text") or ""
            if key:
                metric_map[key] = desc
    else:
        raise ValueError("metric_descriptions must be dict or list")
    if not metric_map:
        raise ValueError("No metrics found")
    metrics = list(metric_map.keys())
    return metric_map, metrics

def trunc_sentences(text, n_sent=N_SENT):
    if not isinstance(text, str):
        return ""
    s = text.replace("؟", ".").replace("।", ".").split(".")
    s = [p.strip() for p in s if p.strip()]
    if not s:
        return text
    return (". ".join(s[:n_sent]) + ('.' if len(s) > n_sent else ''))

def format_example_text(metric_name, metric_map, sys_p, user_p, resp, n_sent=N_SENT):
    desc = metric_map.get(metric_name, "")
    sys_text = sys_p if sys_p else "none"
    return f"[METRIC] {metric_name} - {desc}\n[SYSTEM] {sys_text}\n[USER] {trunc_sentences(user_p, n_sent)}\n[RESPONSE] {trunc_sentences(resp, n_sent)}"

def batched_encode_texts(model, texts, batch_size=BATCH_SIZE, convert_to_tensor=True):
    """
    Encodes list of texts in batches. Returns tensor if convert_to_tensor True, else numpy array.
    Progress bars disabled.
    """
    if convert_to_tensor:
        # collect tensors then stack (to avoid multiple GPU->CPU moves)
        embs = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        return embs  # torch tensor
    else:
        embs_np = model.encode(texts, batch_size=batch_size, convert_to_tensor=False, show_progress_bar=False)
        return np.array(embs_np)

def save_npz_metric(out_path, metrics_list, metric_texts, metric_embs_np):
    np.savez_compressed(out_path, metrics=np.array(metrics_list, dtype=object),
                        metric_texts=np.array(metric_texts, dtype=object),
                        metric_embs=metric_embs_np)

def save_npz_examples(out_path, texts_count, example_embs_np, scores_list=None, metric_names_list=None):
    np.savez_compressed(out_path, ids=np.arange(texts_count), example_embs=example_embs_np,
                        scores=np.array(scores_list, dtype=object) if scores_list is not None else None,
                        metric_names=np.array(metric_names_list, dtype=object) if metric_names_list is not None else None)

def load_npz_metric(path):
    d = np.load(path, allow_pickle=True)
    return list(d["metrics"]), list(d["metric_texts"]), np.array(d["metric_embs"])

def load_npz_examples(path):
    d = np.load(path, allow_pickle=True)
    return np.array(d["ids"]), np.array(d["example_embs"]), d.get("scores", None), d.get("metric_names", None)

# ---- Main pipeline ----
def run_embedding_pipeline(train_path=TRAIN_PATH, test_path=TEST_PATH, metric_path=METRIC_PATH,
                           out_dir=OUT_DIR, model_name=MODEL_NAME, batch_size=BATCH_SIZE, top_k=TOP_K,
                           overwrite=OVERWRITE):
    # load raw jsons
    train = load_json(train_path)
    test = load_json(test_path)
    metric_obj = load_json(metric_path)
    metric_map, metrics_list = ensure_metric_dict(metric_obj)

    # load model once
    print("Loading model:", model_name)
    model = SentenceTransformer(model_name)

    # -------- metric embeddings (compute or load) --------
    metric_npz = out_dir / "metric_embeddings.npz"
    if metric_npz.exists() and not overwrite:
        print("Loading metric embeddings from", metric_npz)
        metrics_list_loaded, metric_texts_loaded, metric_embs_np = load_npz_metric(metric_npz)
        # If loaded metric list differs, warn
        if metrics_list_loaded != metrics_list:
            print("Warning: metrics in file differ from current metric list. Using file's metric list.")
            metrics_list = metrics_list_loaded
            # rebuild metric_map minimal
            metric_map = {m: metric_map.get(m, "") for m in metrics_list}
    else:
        print("Encoding metric descriptions (batched, no progress bar)...")
        metric_texts = [f"[METRIC] {m} - {metric_map.get(m,'')}" for m in metrics_list]
        metric_embs_tensor = batched_encode_texts(model, metric_texts, batch_size=batch_size, convert_to_tensor=True)
        metric_embs_np = metric_embs_tensor.cpu().numpy()
        save_npz_metric(metric_npz, metrics_list, metric_texts, metric_embs_np)
        print("Saved metric embeddings to", metric_npz)

    # create a tensor version for similarity computations
    metric_embs_tensor = model.encode([f"[METRIC] {m} - {metric_map.get(m,'')}" for m in metrics_list],
                                      batch_size=min(batch_size, len(metrics_list)), convert_to_tensor=True, show_progress_bar=False)

    # -------- train embeddings (compute or load) --------
    train_npz = out_dir / "train_emb.npz"
    if train_npz.exists() and not overwrite:
        print("Loading train embeddings from", train_npz)
        ids_train, train_embs_np, train_scores, train_metric_names = load_npz_examples(train_npz)
        # also create tensor for sims
        train_embs_tensor = model.encode([format_example_text(ex.get("metric_name",""), metric_map,
                                                              ex.get("system_prompt",""), ex.get("user_prompt",""), ex.get("response",""))
                                         for ex in train],
                                        batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    else:
        print("Encoding train examples (batched, no progress bar)...")
        train_texts = [format_example_text(ex.get("metric_name",""), metric_map,
                                           ex.get("system_prompt",""), ex.get("user_prompt",""), ex.get("response","")) for ex in train]
        train_embs_tensor = batched_encode_texts(model, train_texts, batch_size=batch_size, convert_to_tensor=True)
        train_embs_np = train_embs_tensor.cpu().numpy()
        train_scores = [ex.get("score", None) for ex in train]
        train_metric_names = [ex.get("metric_name","") for ex in train]
        save_npz_examples(train_npz, len(train_texts), train_embs_np, train_scores, train_metric_names)
        print("Saved train embeddings to", train_npz)

    # -------- test embeddings (compute or load) --------
    test_npz = out_dir / "test_emb.npz"
    if test_npz.exists() and not overwrite:
        print("Loading test embeddings from", test_npz)
        ids_test, test_embs_np, _, test_metric_names = load_npz_examples(test_npz)
        test_embs_tensor = model.encode([format_example_text(ex.get("metric_name",""), metric_map,
                                                             ex.get("system_prompt",""), ex.get("user_prompt",""), ex.get("response",""))
                                        for ex in test],
                                       batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    else:
        print("Encoding test examples (batched, no progress bar)...")
        test_texts = [format_example_text(ex.get("metric_name",""), metric_map,
                                          ex.get("system_prompt",""), ex.get("user_prompt",""), ex.get("response","")) for ex in test]
        test_embs_tensor = batched_encode_texts(model, test_texts, batch_size=batch_size, convert_to_tensor=True)
        test_embs_np = test_embs_tensor.cpu().numpy()
        test_metric_names = [ex.get("metric_name","") for ex in test]
        save_npz_examples(test_npz, len(test_texts), test_embs_np, None, test_metric_names)
        print("Saved test embeddings to", test_npz)

    # -------- compute similarities (example vs metric) using positional util.cos_sim(A,B) --------
    print("Computing example↔metric cosine similarities (positional util.cos_sim)...")
    train_sims = util.cos_sim(train_embs_tensor, metric_embs_tensor)  # shape (N_train, M)
    test_sims  = util.cos_sim(test_embs_tensor, metric_embs_tensor)   # shape (N_test, M)
    train_sims_np = train_sims.cpu().numpy()
    test_sims_np  = test_sims.cpu().numpy()

    # -------- compute top-K and declared sim, build feature dataframes --------
    def build_df_from_sims(dataset, sims_np, metrics_list, top_k, include_score=False):
        rows = []
        idxs = np.argsort(-sims_np, axis=1)[:, :top_k]  # top-k indices
        for i, ex in enumerate(dataset):
            row = {"id": i, "metric_name": ex.get("metric_name","")}
            if include_score:
                row["score"] = ex.get("score", None)
            # declared sim
            declared_metric = ex.get("metric_name","")
            if declared_metric in metrics_list:
                declared_idx = metrics_list.index(declared_metric)
                row["metric_sim_declared"] = float(sims_np[i, declared_idx])
            else:
                row["metric_sim_declared"] = float("nan")
            # top-k sims and metric names
            for k in range(top_k):
                row[f"topk_sim_{k+1}"] = float(sims_np[i, idxs[i,k]])
                row[f"topk_metric_{k+1}"] = metrics_list[idxs[i,k]]
            rows.append(row)
        return pd.DataFrame(rows)

    train_df = build_df_from_sims(train, train_sims_np, metrics_list, top_k, include_score=True)
    test_df  = build_df_from_sims(test, test_sims_np, metrics_list, top_k, include_score=False)

    # ---- add z-scored sim features based on train stats ----
    sim_cols = [f"topk_sim_{k+1}" for k in range(top_k)] + ["metric_sim_declared"]
    train_sim_vals = train_df[sim_cols].astype(float).fillna(train_df[sim_cols].mean())
    test_sim_vals = test_df[sim_cols].astype(float).fillna(train_df[sim_cols].mean())
    scaler = StandardScaler()
    scaler.fit(train_sim_vals)
    train_scaled = scaler.transform(train_sim_vals)
    test_scaled = scaler.transform(test_sim_vals)
    for i, col in enumerate(sim_cols):
        train_df[col + "_z"] = train_scaled[:, i]
        test_df[col + "_z"] = test_scaled[:, i]

    # ---- save features and embeddings (npz saved earlier if computed) ----
    train_feat_path = out_dir / "train_features.csv"
    test_feat_path = out_dir / "test_features.csv"
    train_df.to_csv(train_feat_path, index=False)
    test_df.to_csv(test_feat_path, index=False)
    print("Saved feature CSVs to:", train_feat_path, test_feat_path)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_emb_path": out_dir / "train_emb.npz",
        "test_emb_path": out_dir / "test_emb.npz",
        "metric_emb_path": out_dir / "metric_embeddings.npz"
    }

# ---- Run pipeline ----
res = run_embedding_pipeline(TRAIN_PATH, TEST_PATH, METRIC_PATH, OUT_DIR, MODEL_NAME, BATCH_SIZE, TOP_K, OVERWRITE)
print("Done. Train rows:", len(res["train_df"]), "Test rows:", len(res["test_df"]))
