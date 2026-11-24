# %% [markdown]
# # Feature selection on CIC-IDS2017
# 1. Correlation filter: remove near-duplicate features
# 2. Mutual information ranking
# 3. Random Forest importance ranking
# 4. Combine rankings, then test how many features we really need
# All rankings are computed on TRAINING data only (no test-set leakage).
 
# %% Imports and settings
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
 
OUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_TRAIN_PER_CLASS = 50_000    # same cap as scripts 02 and 03
CORR_THRESHOLD = 0.95           # features correlated above this are redundant
STAT_SAMPLE = 100_000           # rows used for correlation and mutual information
K_VALUES = [5, 10, 20, 30]      # feature-set sizes to test
SWEEP_TREES = 50                # fewer trees in the sweep, for speed
 
# %% Load data and split (identical to scripts 02 and 03)
df = pd.read_parquet(OUT_DIR / "cicids2017_clean.parquet")
meta_cols = ["Label", "source_file", "is_attack"]
features = [c for c in df.columns if c not in meta_cols]
features = [c for c in features if df[c].nunique() > 1]
print("Features available:", len(features))
 
X = df[features].astype("float32")
y = df["Label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
 
rng = np.random.RandomState(RANDOM_STATE)
keep = []
for cls, grp in y_train.groupby(y_train):
    ids = grp.index.values
    if len(ids) > MAX_TRAIN_PER_CLASS:
        ids = rng.choice(ids, MAX_TRAIN_PER_CLASS, replace=False)
    keep.append(ids)
keep = np.concatenate(keep)
X_train, y_train = X_train.loc[keep], y_train.loc[keep]
print("Train:", X_train.shape, "Test:", X_test.shape)
 
# %% Method 1: correlation filter (Spearman, on a sample)
sample_idx = X_train.sample(
    min(STAT_SAMPLE, len(X_train)), random_state=RANDOM_STATE
).index
 
corr = X_train.loc[sample_idx].corr(method="spearman").abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
corr_drop = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
corr_keep = [c for c in features if c not in corr_drop]
 
print(f"\nCorrelation filter (>{CORR_THRESHOLD}): dropped {len(corr_drop)}, kept {len(corr_keep)}")
pairs = (
    upper.stack()
    .loc[lambda s: s > CORR_THRESHOLD]
    .sort_values(ascending=False)
    .head(10)
)
print("\nMost correlated pairs:")
print(pairs.round(3))
 
# %% Method 2: mutual information (on the same sample)
t0 = time.time()
mi = mutual_info_classif(
    X_train.loc[sample_idx],
    y_train.loc[sample_idx],
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
mi_scores = pd.Series(mi, index=features).sort_values(ascending=False)
print(f"\nMutual information done in {time.time() - t0:.0f}s")
print(mi_scores.head(15).round(4))
 
# %% Method 3: Random Forest importance (full training set)
t0 = time.time()
rf_full = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
).fit(X_train, y_train)
rf_scores = pd.Series(rf_full.feature_importances_, index=features).sort_values(
    ascending=False
)
print(f"\nRandom Forest importance done in {time.time() - t0:.0f}s")
print(rf_scores.head(15).round(4))
 
# %% Combine into one ranking
ranking = pd.DataFrame({
    "mi_score": mi_scores,
    "rf_score": rf_scores,
})
ranking["mi_rank"] = ranking["mi_score"].rank(ascending=False)
ranking["rf_rank"] = ranking["rf_score"].rank(ascending=False)
ranking["avg_rank"] = ranking[["mi_rank", "rf_rank"]].mean(axis=1)
ranking["dropped_by_corr"] = ranking.index.isin(corr_drop)
ranking = ranking.sort_values("avg_rank")
ranking.round(4).to_csv(OUT_DIR / "feature_ranking.csv")
 
print("\nTop 20 by combined rank:")
print(ranking.head(20).round(4))
 
# %% Plot: top 20 features by RF importance
top = rf_scores.head(20).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
top.plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Random Forest importance")
ax.set_title("Top 20 features (Random Forest importance)")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance_top20.png", dpi=150)
plt.close()
 
# %% Evaluate a feature set
per_class_f1 = {}
 
 
def evaluate(cols, tag):
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=SWEEP_TREES, n_jobs=-1, random_state=RANDOM_STATE
    ).fit(X_train[cols], y_train)
    pred = model.predict(X_test[cols])
    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    per_class_f1[tag] = {c: rep[c]["f1-score"] for c in model.classes_}
    row = {
        "config": tag,
        "n_features": len(cols),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "seconds": round(time.time() - t0, 1),
    }
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return row
 
 
# %% Sweep: top-k features under each ranking
orderings = {
    "rf_importance": list(rf_scores.index),
    "mutual_info": list(mi_scores.index),
    "combined_rank": list(ranking.index),
}
 
results = [evaluate(features, "all_features")]
results.append(evaluate(corr_keep, "corr_filtered"))
 
for method, order in orderings.items():
    for k in K_VALUES:
        results.append(evaluate(order[:k], f"{method}_top{k}"))
 
# %% Summary
summary = pd.DataFrame(results).set_index("config")
summary["macro_f1"] = summary["macro_f1"].round(4)
summary["weighted_f1"] = summary["weighted_f1"].round(4)
print("\nSUMMARY")
print(summary)
summary.to_csv(OUT_DIR / "feature_selection_results.csv")
 
f1_table = pd.DataFrame(per_class_f1).round(3)
print("\nPER-CLASS F1")
print(f1_table)
f1_table.to_csv(OUT_DIR / "feature_selection_per_class_f1.csv")
 
# %% Plot: macro-F1 versus number of features
fig, ax = plt.subplots(figsize=(9, 6))
for method in orderings:
    sub = summary[summary.index.str.startswith(method)]
    ax.plot(sub["n_features"], sub["macro_f1"], marker="o", label=method)
ax.axhline(
    summary.loc["all_features", "macro_f1"],
    color="gray", linestyle="--", label=f"all {len(features)} features",
)
ax.set_xlabel("Number of features used")
ax.set_ylabel("Macro-F1 (test set)")
ax.set_title("How many features do we really need?")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_selection_curve.png", dpi=150)
plt.close()
print("Saved tables and plots to", OUT_DIR)
 