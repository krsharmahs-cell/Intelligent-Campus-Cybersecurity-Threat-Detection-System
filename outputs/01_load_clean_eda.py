# %% [markdown]
# # CIC-IDS2017: load, clean, and inspect class imbalance
# Run cell by cell in VS Code (click "Run Cell" above each `# %%` line).

# %% Imports and paths
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# %% Load all CSVs
files = sorted(DATA_DIR.rglob("*.csv"))
assert files, f"No CSV files found in {DATA_DIR.resolve()}"

frames = []
for f in files:
    # latin-1 avoids a UnicodeDecodeError in the Thursday files
    part = pd.read_csv(f, encoding="latin-1", low_memory=False)
    part.columns = part.columns.str.strip()  # column names have leading spaces
    part["source_file"] = f.name             # keep track of the day/session
    print(f"{f.name:55s} {part.shape}")
    frames.append(part)

df = pd.concat(frames, ignore_index=True)
print("\nCombined shape:", df.shape)

# %% Basic inspection
print(df.dtypes.value_counts())
print("\nFirst columns:", list(df.columns[:8]))
print("Label column present:", "Label" in df.columns)

# %% Clean the labels
# Web attack labels contain a broken character; replace it with a dash
df["Label"] = (
    df["Label"]
    .astype(str)
    .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
    .str.strip()
)
print(df["Label"].unique())

# %% Clean the features
before = len(df)

# 'Fwd Header Length' appears twice in the original data (pandas names the copy ".1")
dup_cols = [c for c in df.columns if c.endswith(".1")]
if dup_cols:
    print("Dropping duplicate columns:", dup_cols)
    df = df.drop(columns=dup_cols)

# inf values (mostly in Flow Bytes/s and Flow Packets/s) -> NaN
feature_cols = [c for c in df.columns if c not in ("Label", "source_file")]
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
n_inf = np.isinf(df[feature_cols]).sum().sum()
df = df.replace([np.inf, -np.inf], np.nan)
print(f"inf values replaced: {n_inf}")
print(f"NaN values: {df[feature_cols].isna().sum().sum()}")

# Drop rows with NaN
df = df.dropna(subset=feature_cols)
print(f"Rows removed (NaN/inf): {before - len(df)}")

# Drop exact duplicates (important: duplicates cause train/test leakage)
before_dup = len(df)
df = df.drop_duplicates(subset=feature_cols + ["Label"])
print(f"Rows removed (duplicates): {before_dup - len(df)}")
print("Clean shape:", df.shape)

# %% Constant columns (no information)
const_cols = [c for c in feature_cols if df[c].nunique() <= 1]
print("Constant columns:", const_cols)

# %% Class distribution (your class imbalance evidence)
counts = df["Label"].value_counts()
summary = pd.DataFrame({
    "count": counts,
    "percent": (counts / counts.sum() * 100).round(4),
})
print(summary)
print(f"\nImbalance ratio (largest/smallest): {counts.max() / counts.min():,.0f} : 1")

# %% Plot class distribution (log scale so rare classes are visible)
fig, ax = plt.subplots(figsize=(10, 6))
counts.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_xscale("log")
ax.set_xlabel("Number of flows (log scale)")
ax.set_title("CIC-IDS2017 class distribution (after cleaning)")
for i, v in enumerate(counts.sort_values()):
    ax.text(v, i, f" {v:,}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
plt.show()




# %% Binary view: benign vs malicious
df["is_attack"] = (df["Label"] != "BENIGN").astype(int)
print(df["is_attack"].value_counts(normalize=True).round(4))

# %% Attacks per day (useful for time-based splits later)
print(pd.crosstab(df["source_file"], df["Label"]).T)

# %% Save cleaned data for the next steps
df.to_parquet(OUT_DIR / "cicids2017_clean.parquet", index=False)
print("Saved:", OUT_DIR / "cicids2017_clean.parquet")