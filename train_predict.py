import os, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import lightgbm as lgb

# ---------------- CONFIG ----------------
OUT_DIR = Path("embeddings_out")
TRAIN_EMB = OUT_DIR / "train_emb.npz"
TRAIN_FEAT = OUT_DIR / "train_features.csv"   
TEST_EMB = OUT_DIR / "test_emb.npz"           
TEST_FEAT = OUT_DIR / "test_features.csv"      
METRIC_EMB = OUT_DIR / "metric_embeddings.npz" 

OUT_CSV = Path("predictions_lgb_three_stage.csv")

SEED = 42
TEST_ID_OFFSET = 1  

# classifier training controls
VALID_FRAC = 0.15
HIGH_KEEP_FRACTION = 0.25     
OVERSAMPLE_MULTIPLIERS = {0:1.0, 1:3.0, 2:1.0}  
# lgb params 
LGB_CLS_PARAMS = {
    "objective":"multiclass",
    "num_class":3,
    "boosting_type":"gbdt",
    "n_estimators":3000,
    "learning_rate":0.05,
    "num_leaves":127,
    "min_data_in_leaf":20,
    "verbosity":-1,
    "seed":SEED,
    "n_jobs":-1
}
LGB_REG_PARAMS = {
    "objective":"regression",
    "boosting_type":"gbdt",
    "n_estimators":5000,
    "learning_rate":0.05,
    "num_leaves":127,
    "min_data_in_leaf":20,
    "verbosity":-1,
    "seed":SEED,
    "n_jobs":-1
}

EARLY_STOPPING_ROUNDS_CLS = 100
EARLY_STOPPING_ROUNDS_REG = 100
VERBOSE = 100  

# ---------- helper ----------

def score_to_bin(s):
    if s < 3.5: return 0
    if s < 7.5: return 1
    return 2

def safe_float(x):
    try: return float(x)
    except: return math.nan

# ---------------- load data ----------------
if not TRAIN_EMB.exists():
    raise FileNotFoundError(f"train embedding file not found: {TRAIN_EMB}")

train_np = np.load(TRAIN_EMB, allow_pickle=True)
train_feat_df = pd.read_csv(TRAIN_FEAT) if TRAIN_FEAT.exists() else None

X_all = np.array(train_np["example_embs"])
scores_raw = np.array(train_np["scores"], dtype=object)
metric_names = np.array(train_np.get("metric_names", np.array([""]*len(X_all))), dtype=object)

# optional test
test_exists = TEST_EMB.exists() and TEST_FEAT.exists()
if test_exists:
    test_np = np.load(TEST_EMB, allow_pickle=True)
    X_test_all = np.array(test_np["example_embs"])
    test_feat_df = pd.read_csv(TEST_FEAT)
else:
    X_test_all = None
    test_feat_df = None

# append metric embeddings if present
if METRIC_EMB.exists():
    met = np.load(METRIC_EMB, allow_pickle=True)
    metrics_list = list(met["metrics"])
    metric_embs = np.array(met["metric_embs"])
    metric_to_idx = {m:i for i,m in enumerate(metrics_list)}
    Dm = metric_embs.shape[1]
    def get_metric_embs(names):
        out=[]
        for nm in names:
            if nm in metric_to_idx:
                out.append(metric_embs[metric_to_idx[nm]])
            else:
                out.append(np.zeros(Dm))
        return np.vstack(out)
    X_metric = get_metric_embs(metric_names)
    X_all = np.hstack([X_all, X_metric])
    if test_exists:
        test_metric_emb = get_metric_embs(np.array(test_np.get("metric_names", np.array([""]*len(X_test_all))), dtype=object))
        X_test_all = np.hstack([X_test_all, test_metric_emb])

# append sim features if present in train_features.csv (optional)
sim_cols = []
if train_feat_df is not None:
    # look for topk_sim_1...topk_sim_5 and metric_sim_declared
    for k in range(1,6):
        col = f"topk_sim_{k}"
        if col in train_feat_df.columns:
            sim_cols.append(col)
    if "metric_sim_declared" in train_feat_df.columns:
        sim_cols.append("metric_sim_declared")
    if sim_cols:
        train_sim = train_feat_df[sim_cols].astype(float).fillna(train_feat_df[sim_cols].mean()).values
        X_all = np.hstack([X_all, train_sim])
        if test_exists and test_feat_df is not None:
            test_sim = test_feat_df[sim_cols].astype(float).fillna(train_feat_df[sim_cols].mean()).values
            X_test_all = np.hstack([X_test_all, test_sim])

print("X_all shape:", X_all.shape, "X_test_all shape:", None if X_test_all is None else X_test_all.shape)

# ---------- prepare target and drop NaNs ----------
y_all = np.array([safe_float(x) for x in scores_raw], dtype=float)
mask_has = ~np.isnan(y_all)
if not mask_has.all():
    print("Dropping rows with missing scores:", np.sum(~mask_has))
    X_all = X_all[mask_has]
    y_all = y_all[mask_has]
    metric_names = metric_names[mask_has]
    if train_feat_df is not None:
        train_feat_df = train_feat_df.loc[mask_has].reset_index(drop=True)

bins = np.array([score_to_bin(v) for v in y_all], dtype=int)
print("Overall bin counts:", np.bincount(bins))

# ---------- train/val split (stratified by bin) ----------
idx = np.arange(len(X_all))
tr_idx, val_idx = train_test_split(idx, test_size=VALID_FRAC, stratify=bins, random_state=SEED)
X_tr_full = X_all[tr_idx]; y_tr_full = y_all[tr_idx]; bins_tr_full = bins[tr_idx]
X_val = X_all[val_idx]; y_val = y_all[val_idx]; bins_val = bins[val_idx]

# ---------- undersample high bin for classifier train ----------
high_label = 2
high_inds_local = np.where(bins_tr_full == high_label)[0]
n_high = len(high_inds_local)
n_keep = max(1, int(round(n_high * float(HIGH_KEEP_FRACTION))))
rng = np.random.RandomState(SEED)
if n_keep < n_high:
    keep_high = rng.choice(high_inds_local, size=n_keep, replace=False)
else:
    keep_high = high_inds_local
non_high_local = np.where(bins_tr_full != high_label)[0]
selected_local = np.concatenate([non_high_local, keep_high])
rng.shuffle(selected_local)

X_tr_cls = X_tr_full[selected_local]
y_tr_cls = bins_tr_full[selected_local]   # classifier target is bin label
print("Classifier train bin counts after undersample:", np.bincount(y_tr_cls))

# ---------- scale features (fit on classifier train undersampled set) ----------
scaler = StandardScaler()
scaler.fit(X_tr_cls)
X_tr_cls_s = scaler.transform(X_tr_cls)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test_all) if X_test_all is not None else None

# ---------- compute classifier sample weights to oversample middle bin ----------
# base inverse-frequency on undersampled classifier train set
unique, counts = np.unique(y_tr_cls, return_counts=True)
class_counts = dict(zip(unique, counts))
base_class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
class_weights = {}
for c in base_class_weights:
    mult = OVERSAMPLE_MULTIPLIERS.get(int(c), 1.0)
    class_weights[c] = base_class_weights[c] * float(mult)

sample_weight_cls = np.array([class_weights[int(lbl)] for lbl in y_tr_cls], dtype=float)

# ---------- train LGBMClassifier ----------
print("Training LightGBM classifier...")
clf = lgb.LGBMClassifier(**LGB_CLS_PARAMS)

# fit supports early_stopping_rounds in sklearn API
clf.fit(
    X_tr_cls_s, y_tr_cls,
    sample_weight=sample_weight_cls,
    eval_set=[(X_val_s, bins_val)],
    eval_metric="multi_logloss",
    early_stopping_rounds=EARLY_STOPPING_ROUNDS_CLS,
    verbose=VERBOSE
)

# classifier validation metrics
val_bin_preds = clf.predict(X_val_s)
print("Classifier validation accuracy:", accuracy_score(bins_val, val_bin_preds))
print("Classifier classification report:\n", classification_report(bins_val, val_bin_preds, digits=4))

# ---------- train per-bin regressors on full training pool (recommended) ----------
# Use full training partition (not undersampled) so regressors see all examples for their bin
X_reg_pool = X_tr_full
y_reg_pool = y_tr_full
bins_reg_pool = bins_tr_full

# scale whole regressor pool with same scaler
X_reg_s = scaler.transform(X_reg_pool)

regressors = {}
reg_val_rmse = {}

for b in [0,1,2]:
    idxs = np.where(bins_reg_pool == b)[0]
    if len(idxs) < 10:
        print(f"Skipping regressor for bin {b} (only {len(idxs)} samples).")
        regressors[b] = None
        reg_val_rmse[b] = None
        continue

    Xb = X_reg_s[idxs]
    yb = y_reg_pool[idxs]

    # validation rows that belong to this bin
    val_idxs_local = np.where(bins_val == b)[0]
    if len(val_idxs_local) == 0:
        Xb_val = X_val_s
        yb_val = y_val
    else:
        Xb_val = X_val_s[val_idxs_local]
        yb_val = y_val[val_idxs_local]

    print(f"Training regressor for bin {b}: train_samples={len(idxs)} val_samples={len(val_idxs_local)}")
    reg = lgb.LGBMRegressor(**LGB_REG_PARAMS)
    reg.fit(
        Xb, yb,
        eval_set=[(Xb_val, yb_val)],
        eval_metric="rmse",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS_REG,
        verbose=VERBOSE
    )
    regressors[b] = reg

    # evaluate on the validation rows for that bin
    if len(val_idxs_local) > 0:
        yb_val_pred = reg.predict(Xb_val, num_iteration=reg.best_iteration_)
        rmse_b = root_mean_squared_error(yb_val, yb_val_pred)
        reg_val_rmse[b] = rmse_b
        print(f"Bin {b} regressor val RMSE: {rmse_b:.4f}")
    else:
        reg_val_rmse[b] = None

# ---------- inference: classifier -> per-bin regressors for validation and test ----------
# Validation: continuous predictions
val_preds_cont = np.zeros(len(X_val_s), dtype=float)
with np.errstate(all='ignore'):
    for i in range(len(X_val_s)):
        pred_bin = int(val_bin_preds[i])  # classifier prediction on validation
        reg = regressors.get(pred_bin)
        if reg is None:
            # fallback mean of that bin from regressor pool
            mask = (bins_reg_pool == pred_bin)
            val_preds_cont[i] = float(np.mean(y_reg_pool[mask])) if mask.any() else float(np.mean(y_reg_pool))
        else:
            val_preds_cont[i] = reg.predict(X_val_s[i:i+1], num_iteration=reg.best_iteration_).item()
val_preds_cont = np.clip(val_preds_cont, 0.0, 10.0)

val_rmse = root_mean_squared_error(y_val, val_preds_cont)
val_mae = mean_absolute_error(y_val, val_preds_cont)
print(f"Final validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
print("Per-bin regressor val RMSE:", reg_val_rmse)

# Test inference (if test exists)
if X_test_s is not None:
    test_bin_preds = clf.predict(X_test_s)
    test_preds = np.zeros(len(X_test_s), dtype=float)
    for i in range(len(X_test_s)):
        pb = int(test_bin_preds[i])
        reg = regressors.get(pb)
        if reg is None:
            mask = (bins_reg_pool == pb)
            test_preds[i] = float(np.mean(y_reg_pool[mask])) if mask.any() else float(np.mean(y_reg_pool))
        else:
            test_preds[i] = reg.predict(X_test_s[i:i+1], num_iteration=reg.best_iteration_).item()
    test_preds = np.clip(test_preds, 0.0, 10.0)
    out_df = pd.DataFrame({"id": np.arange(TEST_ID_OFFSET, TEST_ID_OFFSET + len(test_preds)), "score": np.round(test_preds, 4)})
    out_df.to_csv(OUT_CSV, index=False)
    print("Saved test predictions to", OUT_CSV)
else:
    # Save validation preds (if no test file provided)
    out_df = pd.DataFrame({"id": np.arange(1, len(val_preds_cont)+1), "score": np.round(val_preds_cont, 4)})
    out_df.to_csv(OUT_CSV, index=False)
    print("Saved validation predictions to", OUT_CSV)

# ---------- final summary ----------
print("Done. Classifier params:", LGB_CLS_PARAMS)
print("Done. Regressor params:", LGB_REG_PARAMS)
