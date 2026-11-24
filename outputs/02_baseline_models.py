# %% [markdown]
# # Baseline models on CIC-IDS2017
# Compares Logistic Regression and Random Forest, each with and without class weights.
# Run from the project root (the folder that contains outputs/).
 
# %% Imports and settings
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
 
OUT_DIR = Path("outputs")
RANDOM_STATE = 42
TASK = "multiclass"          # "multiclass" (15 labels) or "binary" (benign vs attack)
TEST_SIZE = 0.30
# Cap on training rows per class, to keep runtime reasonable on a laptop.
# The TEST set is never capped, so evaluation reflects the real distribution.
# Set to None to train on everything (slow).
MAX_TRAIN_PER_CLASS = 50_000
 
# %% Load the cleaned data
df = pd.read_parquet(OUT_DIR / "cicids2017_clean.parquet")
print("Loaded:", df.shape)
 
target_col = "Label" if TASK == "multiclass" else "is_attack"
meta_cols = ["Label", "source_file", "is_attack"]
features = [c for c in df.columns if c not in meta_cols]
 
# Remove constant columns (no information)
constant = [c for c in features if df[c].nunique() <= 1]
features = [c for c in features if c not in constant]
print(f"Using {len(features)} features (dropped {len(constant)} constant columns)")
 
X = df[features].astype("float32")
y = df[target_col]
 
# %% Stratified train/test split (BEFORE any scaling or resampling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)
 
# %% Cap the training set per class (test set untouched)
if MAX_TRAIN_PER_CLASS:
    rng = np.random.RandomState(RANDOM_STATE)
    keep = []
    for cls, grp in y_train.groupby(y_train):
        ids = grp.index.values
        if len(ids) > MAX_TRAIN_PER_CLASS:
            ids = rng.choice(ids, MAX_TRAIN_PER_CLASS, replace=False)
        keep.append(ids)
    keep = np.concatenate(keep)
    X_train, y_train = X_train.loc[keep], y_train.loc[keep]
 
print("Training rows used:", len(X_train))
print("\nTraining class counts:")
print(y_train.value_counts())
print("\nTest class counts:")
print(y_test.value_counts())
 
# %% Define models (scaler lives inside the pipeline, so it is fitted on train only)
def make_model(name, class_weight):
    if name == "LogReg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, class_weight=class_weight),
        )
    return RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
 
 
def macro_pr_auc(y_true, proba, classes):
    """Average precision per class, then averaged (macro)."""
    if len(classes) == 2:
        return average_precision_score(y_true == classes[1], proba[:, 1])
    y_bin = label_binarize(y_true, classes=classes)
    scores = []
    for i in range(len(classes)):
        if y_bin[:, i].sum() > 0:  # class must appear in the test set
            scores.append(average_precision_score(y_bin[:, i], proba[:, i]))
    return float(np.mean(scores))
 
 
# %% Train and evaluate all four configurations
configs = [
    ("LogReg", None),
    ("LogReg", "balanced"),
    ("RandomForest", None),
    ("RandomForest", "balanced"),
]
 
results = []
per_class_f1 = {}
fitted = {}
 
for name, cw in configs:
    tag = f"{name} | class_weight={cw}"
    print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}")
 
    model = make_model(name, cw)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
 
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.classes_
 
    row = {
        "model": tag,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "macro_pr_auc": macro_pr_auc(y_test, proba, classes),
        "train_seconds": round(train_time, 1),
    }
    results.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
 
    report = classification_report(
        y_test, pred, digits=3, zero_division=0, output_dict=True
    )
    per_class_f1[tag] = {
        str(c): report[str(c)]["f1-score"] for c in classes
    }
    print(classification_report(y_test, pred, digits=3, zero_division=0))
 
    fitted[tag] = (model, pred)
 
# %% Summary tables
summary = pd.DataFrame(results).set_index("model").round(4)
print("\nSUMMARY")
print(summary)
summary.to_csv(OUT_DIR / f"baseline_summary_{TASK}.csv")
 
per_class = pd.DataFrame(per_class_f1).round(3)
print("\nPER-CLASS F1")
print(per_class)
per_class.to_csv(OUT_DIR / f"baseline_per_class_f1_{TASK}.csv")
 
# %% Confusion matrix for the best model (by macro-F1)
best_tag = summary["macro_f1"].idxmax()
best_model, best_pred = fitted[best_tag]
labels = list(best_model.classes_)
cm = confusion_matrix(y_test, best_pred, labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)  # each row sums to 1 (= recall)
 
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Normalized confusion matrix: {best_tag}")
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm_norm[i, j] >= 0.01:
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_DIR / f"confusion_matrix_{TASK}.png", dpi=150)
plt.close()
print("Best model:", best_tag)
print("Saved confusion matrix and CSV summaries to", OUT_DIR)
 