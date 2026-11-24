# %% [markdown]
# # Train and save the final model for the dashboard
# Final model: Random Forest (100 trees), no class weights, no resampling,
# correlation-filtered features. Also writes a demo file of held-out flows.
#
# Requires: pip install joblib

# %% Imports and settings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_TRAIN_PER_CLASS = 50_000
CORR_THRESHOLD = 0.95
STAT_SAMPLE = 100_000
N_TREES = 100
DEMO_BENIGN = 4000        # benign flows in the demo file
DEMO_PER_ATTACK = 100     # max flows per attack class in the demo file

# %% Load data and split (same procedure as scripts 05 and 06)
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

# %% Correlation filter and training
sample = X_tr.sample(min(STAT_SAMPLE, len(X_tr)), random_state=RANDOM_STATE)
corr = sample.corr(method="spearman").abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
feats = [c for c in X_tr.columns if c not in drop]
print(f"Training on {len(X_tr)} flows with {len(feats)} features")

model = RandomForestClassifier(
    n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE
).fit(X_tr[feats], y_tr)

# %% Save the model bundle
bundle = {
    "model": model,
    "features": feats,
    "classes": [str(c) for c in model.classes_],
}
path = OUT_DIR / "final_model.joblib"
joblib.dump(bundle, path, compress=3)
print(f"Saved {path} ({path.stat().st_size / 1e6:.0f} MB)")

# %% Demo file: held-out TEST flows only (the model never trained on these)
rng = np.random.RandomState(RANDOM_STATE)
pick = []
for cls, grp in y_te.groupby(y_te):
    ids = grp.index.values
    n = DEMO_BENIGN if cls == "BENIGN" else DEMO_PER_ATTACK
    if len(ids) > n:
        ids = rng.choice(ids, n, replace=False)
    pick.append(ids)
pick = rng.permutation(np.concatenate(pick))

demo = X_te.loc[pick, all_features].copy()
demo["Label"] = y_te.loc[pick].values
demo.to_csv(OUT_DIR / "demo_flows.csv", index=False)
print(f"Saved demo_flows.csv: {len(demo)} flows")
print(demo["Label"].value_counts())