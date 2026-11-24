# %% [markdown]
# # Explainability on CIC-IDS2017 (SHAP)
# Final model: Random Forest (100 trees), no class weights, no resampling,
# correlation-filtered features.
#
# 1. Global: which features matter overall, and for each attack class
# 2. Why are XSS and Brute Force confused? (single-feature separability)
# 3. Family view: web attacks grouped into one class
# 4. Local: "alert cards" that explain individual flagged flows
#
# Requires: pip install shap

# %% Imports and settings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

OUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_TRAIN_PER_CLASS = 50_000
CORR_THRESHOLD = 0.95
STAT_SAMPLE = 100_000
N_TREES = 100
N_PER_CLASS = 80      # flows explained per class (lower this if SHAP is too slow)
TOP_N = 8             # features shown per alert card
WEB = ["Web Attack - Brute Force", "Web Attack - Sql Injection", "Web Attack - XSS"]

# %% Load data, split, train the final model (same procedure as script 05)
df = pd.read_parquet(OUT_DIR / "cicids2017_clean.parquet")
meta_cols = ["Label", "source_file", "is_attack"]
all_features = [c for c in df.columns if c not in meta_cols and df[c].nunique() > 1]
X = df[all_features].astype("float32")
y = df["Label"]
del df

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

rng = np.random.RandomState(RANDOM_STATE)
keep = []
for _, grp in y_tr.groupby(y_tr):
    ids = grp.index.values
    if len(ids) > MAX_TRAIN_PER_CLASS:
        ids = rng.choice(ids, MAX_TRAIN_PER_CLASS, replace=False)
    keep.append(ids)
keep = np.concatenate(keep)
X_tr, y_tr = X_tr.loc[keep], y_tr.loc[keep]

sample = X_tr.sample(min(STAT_SAMPLE, len(X_tr)), random_state=RANDOM_STATE)
corr = sample.corr(method="spearman").abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
feats = [c for c in X_tr.columns if c not in drop]
print(f"Using {len(feats)} features")

model = RandomForestClassifier(
    n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE
).fit(X_tr[feats], y_tr)
classes = list(model.classes_)
pred_test = pd.Series(model.predict(X_te[feats]), index=X_te.index)

# %% Family view: are the three web attacks separable from each other, or just from the rest?
fine = precision_recall_fscore_support(
    y_te, pred_test, labels=WEB, zero_division=0
)
fam_true = y_te.isin(WEB)
fam_pred = pred_test.isin(WEB)
p, r, f1, _ = precision_recall_fscore_support(
    fam_true, fam_pred, average="binary", zero_division=0
)
print("\nWEB ATTACKS: individual classes")
for lab, pi, ri, fi in zip(WEB, fine[0], fine[1], fine[2]):
    print(f"  {lab:30s} precision {pi:.3f}  recall {ri:.3f}  F1 {fi:.3f}")
print(f"WEB ATTACKS: as one family   precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")

# %% Choose flows to explain: up to N_PER_CLASS test flows per class
rng = np.random.RandomState(RANDOM_STATE)
pick = []
for _, grp in y_te.groupby(y_te):
    ids = grp.index.values
    if len(ids) > N_PER_CLASS:
        ids = rng.choice(ids, N_PER_CLASS, replace=False)
    pick.append(ids)
pick = np.concatenate(pick)

Xs = X_te.loc[pick, feats]
ys = y_te.loc[pick]
ps = pred_test.loc[pick]
print(f"\nExplaining {len(Xs)} flows with SHAP (this can take several minutes)...")

# %% Compute SHAP values
t0 = time.time()
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xs, check_additivity=False)
if isinstance(sv, list):                 # older SHAP versions: list of (n, f) per class
    sv = np.stack(sv, axis=-1)
sv = np.asarray(sv)                      # shape: (flows, features, classes)
print(f"SHAP done in {time.time() - t0:.0f}s, shape {sv.shape}")

# %% 1. Global and per-class importance
# per_class[f, k] = mean |SHAP| of feature f for the model output of class k
per_class = pd.DataFrame(np.abs(sv).mean(axis=0), index=feats, columns=classes)
global_imp = per_class.mean(axis=1).sort_values(ascending=False)
per_class.round(4).to_csv(OUT_DIR / "shap_per_class_importance.csv")
global_imp.round(4).to_csv(OUT_DIR / "shap_global_importance.csv")

print("\nTOP 15 FEATURES (mean |SHAP| over all classes)")
print(global_imp.head(15).round(4))

gini = pd.Series(model.feature_importances_, index=feats)
rho = spearmanr(global_imp.loc[feats], gini.loc[feats]).correlation
print(f"\nRank agreement between SHAP and Random Forest (Gini) importance: "
      f"Spearman = {rho:.2f}")

top_feats = pd.DataFrame(
    {c: per_class[c].sort_values(ascending=False).head(5).index.tolist()
     for c in classes}
).T
top_feats.columns = [f"top{i}" for i in range(1, 6)]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print("\nTOP 5 FEATURES FOR EACH CLASS (what defines each attack)")
print(top_feats)
top_feats.to_csv(OUT_DIR / "shap_top_features_per_class.csv")

# Bar chart: global importance
top20 = global_imp.head(20).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
top20.plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Mean |SHAP value| (averaged over classes)")
ax.set_title("Global feature importance (SHAP)")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_global_importance.png", dpi=150)
plt.close()

# Heatmap: top 20 features x classes (each class scaled to its own maximum)
hm = per_class.loc[global_imp.head(20).index]
hm = hm / hm.max(axis=0).replace(0, 1)
fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(hm.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=90, fontsize=8)
ax.set_yticks(range(len(hm)))
ax.set_yticklabels(hm.index, fontsize=8)
ax.set_title("Which features drive each class (scaled within each class)")
fig.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_class_feature_heatmap.png", dpi=150)
plt.close()

# %% 2. Why are XSS and Brute Force confused?
# For every feature: AUC of that single feature at separating the two classes.
# AUC 0.5 = no separation at all; 1.0 or 0.0 = perfect separation.
a, b = "Web Attack - XSS", "Web Attack - Brute Force"
mask = y_te.isin([a, b])
Xp = X_te.loc[mask, feats]
yp = (y_te[mask] == a).astype(int)

rows = []
for f in feats:
    auc = roc_auc_score(yp, Xp[f])
    rows.append({
        "feature": f,
        "auc_xss_vs_bruteforce": round(auc, 3),
        "separation": round(abs(auc - 0.5) * 2, 3),   # 0 = identical, 1 = perfect
        "median_xss": float(Xp.loc[yp == 1, f].median()),
        "median_bruteforce": float(Xp.loc[yp == 0, f].median()),
    })
sep = pd.DataFrame(rows).sort_values("separation", ascending=False)
sep.to_csv(OUT_DIR / "xss_vs_bruteforce_separability.csv", index=False)
print(f"\nXSS vs BRUTE FORCE: best single-feature separability "
      f"(test flows: {int(yp.sum())} XSS, {int((1 - yp).sum())} Brute Force)")
print(sep.head(10).to_string(index=False))
print(f"Median separation over all features: {sep['separation'].median():.3f}")

# %% 3. Local explanations: alert cards for individual flows
examples = [
    ("PortScan", None),
    ("Bot", None),
    ("DoS slowloris", None),
    ("Web Attack - XSS", "Web Attack - Brute Force"),   # a misclassified flow
]

card_rows = []
for n, (true_cls, pred_cls) in enumerate(examples, start=1):
    want = pred_cls or true_cls
    m = ((ys == true_cls) & (ps == want)).to_numpy()
    if not m.any():
        print(f"\n(No example found for true={true_cls}, predicted={want})")
        continue

    pos = int(np.where(m)[0][0])
    k = classes.index(want)
    contrib = pd.Series(sv[pos, :, k], index=feats)
    values = Xs.iloc[pos]
    prob = float(model.predict_proba(Xs.iloc[[pos]])[0, k])

    order = contrib.abs().sort_values(ascending=False).head(TOP_N).index
    print(f"\n{'=' * 70}")
    print(f"ALERT CARD {n}: predicted {want}  (confidence {prob:.0%}; true label: {true_cls})")
    print("Top reasons (positive = pushes toward this prediction):")
    for rank, fname in enumerate(order, start=1):
        print(f"  {rank}. {fname} = {values[fname]:.4g}   (SHAP {contrib[fname]:+.3f})")
        card_rows.append({
            "card": n, "true_label": true_cls, "predicted": want,
            "confidence": round(prob, 3), "rank": rank, "feature": fname,
            "value": float(values[fname]), "shap": round(float(contrib[fname]), 4),
        })

    shown = contrib.loc[order].iloc[::-1]
    labels = [f"{f} = {values[f]:.3g}" for f in shown.index]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in shown.values]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, shown.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Contribution to '{want}' (red = toward, blue = against)")
    ax.set_title(f"Alert {n}: predicted {want} ({prob:.0%}), true {true_cls}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"explain_alert_{n}.png", dpi=150)
    plt.close()

pd.DataFrame(card_rows).to_csv(OUT_DIR / "explain_alert_cards.csv", index=False)
print("\nSaved SHAP tables, plots and alert cards to", OUT_DIR)