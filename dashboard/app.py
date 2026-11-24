"""Network Threat Triage dashboard.

Run from the project root:   streamlit run app.py
Needs outputs/final_model.joblib and outputs/demo_flows.csv (made by 07_train_final_model.py).
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path("outputs/final_model.joblib")
DEMO_PATH = Path("outputs/demo_flows.csv")
WEB = ["Web Attack - Brute Force", "Web Attack - Sql Injection", "Web Attack - XSS"]
FAMILY = "Web Attack (family)"
FACT_COLS = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Bytes/s"]

st.set_page_config(page_title="Network Threat Triage", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle, shap.TreeExplainer(bundle["model"])


def prepare(raw, feats):
    """Clean an uploaded table the same way the training data was cleaned."""
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    labels = None
    if "Label" in df.columns:
        labels = (
            df["Label"].astype(str)
            .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
            .str.strip()
        )
    missing = [f for f in feats if f not in df.columns]
    if missing:
        return None, None, missing, 0
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1)
    dropped = int((~ok).sum())
    X = X[ok].astype("float32").reset_index(drop=True)
    if labels is not None:
        labels = labels[ok].reset_index(drop=True)
    return X, labels, [], dropped


@st.cache_data(show_spinner="Scoring flows...")
def score(X, _model):
    return _model.predict_proba(X)


loaded = load_model()
if loaded is None:
    st.error("Model file not found. Run `python 07_train_final_model.py` first.")
    st.stop()
bundle, explainer = loaded
model, feats, classes = bundle["model"], bundle["features"], bundle["classes"]
benign_i = classes.index("BENIGN")
web_i = [classes.index(w) for w in WEB if w in classes]

# ----------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
source = st.sidebar.radio("Data source", ["Demo sample (held-out flows)", "Upload CSV"])
threshold = st.sidebar.slider(
    "Alert threshold (threat score)", 0.05, 0.95, 0.50, 0.05,
    help="Lower = catch more attacks but raise more false alarms. "
         "Higher = fewer false alarms but more missed attacks.",
)
group_web = st.sidebar.checkbox(
    "Group web attacks into one family", value=True,
    help="The model separates web attacks from normal traffic very well, "
         "but cannot reliably tell Brute Force, XSS and SQL Injection apart.",
)

if source.startswith("Demo"):
    if not DEMO_PATH.exists():
        st.error("Demo file not found. Run `python 07_train_final_model.py` first.")
        st.stop()
    raw = pd.read_csv(DEMO_PATH)
    st.sidebar.caption("Demo flows come from the held-out test split, so the model "
                       "never trained on them. The sample contains more attacks "
                       "than real traffic would.")
else:
    up = st.sidebar.file_uploader("CSV of network flows (CICFlowMeter columns)", type="csv")
    if up is None:
        st.title("Network Threat Triage")
        st.info("Upload a CSV with the CIC-IDS2017 flow columns, or switch to the demo sample.")
        st.stop()
    raw = pd.read_csv(up, low_memory=False, encoding="latin-1")

X, labels, missing, dropped = prepare(raw, feats)
if missing:
    st.error(f"The file is missing {len(missing)} required columns, for example: {missing[:5]}")
    st.stop()
if len(X) == 0:
    st.error("No usable rows after cleaning (all rows had missing or infinite values).")
    st.stop()

# ----------------------------------------------------------------- scoring
proba = score(X, model)
threat = 1.0 - proba[:, benign_i]
masked = proba.copy()
masked[:, benign_i] = -1.0
type_i = masked.argmax(axis=1)
fine_type = np.array(classes)[type_i]
type_prob = proba[np.arange(len(proba)), type_i]

disp_type = fine_type.copy()
if group_web:
    is_web = np.isin(fine_type, WEB)
    disp_type[is_web] = FAMILY
    type_prob = np.where(is_web, proba[:, web_i].sum(axis=1), type_prob)

res = pd.DataFrame({
    "Row": np.arange(len(X)),
    "Threat score": threat.round(3),
    "Predicted type": disp_type,
    "Type probability": type_prob.round(3),
})
for c in FACT_COLS:
    if c in X.columns:
        res[c] = X[c].values
if labels is not None:
    true_disp = labels.where(~labels.isin(WEB), FAMILY) if group_web else labels
    res["True label"] = true_disp.values
flagged = threat >= threshold

# ----------------------------------------------------------------- header
st.title("Network Threat Triage")
st.caption("Machine-learning triage of network flows with per-alert explanations. "
           "Threat score = 1 minus the model's probability that the flow is benign.")
if dropped:
    st.caption(f"{dropped:,} rows were ignored because they had missing or infinite values.")

c1, c2, c3 = st.columns(3)
c1.metric("Flows analysed", f"{len(X):,}")
c2.metric("Alerts raised", f"{int(flagged.sum()):,}")
c3.metric("Share flagged", f"{flagged.mean():.1%}")

if labels is not None:
    is_attack = (labels != "BENIGN").to_numpy()
    tp = int((flagged & is_attack).sum())
    prec = tp / flagged.sum() if flagged.sum() else float("nan")
    rec = tp / is_attack.sum() if is_attack.sum() else float("nan")
    fa = float((flagged & ~is_attack).sum() / max((~is_attack).sum(), 1))
    d1, d2, d3 = st.columns(3)
    d1.metric("Alert precision", "n/a" if np.isnan(prec) else f"{prec:.1%}")
    d2.metric("Attack recall", "n/a" if np.isnan(rec) else f"{rec:.1%}")
    d3.metric("False alarm rate (benign)", f"{fa:.2%}")
    st.caption("These three numbers use the true labels in the file and change as you move the threshold.")

tab_alerts, tab_over, tab_about = st.tabs(["Alert queue", "Overview", "About and limits"])

# ----------------------------------------------------------------- alerts
with tab_alerts:
    alerts = res[flagged].sort_values("Threat score", ascending=False)
    if alerts.empty:
        st.success("No flows exceed the current threshold.")
    else:
        st.subheader(f"{len(alerts):,} flagged flows (highest threat first)")
        st.dataframe(alerts.head(500), hide_index=True)

        top = alerts.head(200)
        options = top["Row"].tolist()
        lookup = top.set_index("Row")
        pick = st.selectbox(
            "Choose a flagged flow to explain",
            options,
            format_func=lambda r: (
                f"Row {r} | {lookup.loc[r, 'Predicted type']} | "
                f"score {lookup.loc[r, 'Threat score']:.2f}"
            ),
        )

        row = X.iloc[[pick]]
        sv = explainer.shap_values(row, check_additivity=False)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        sv = np.asarray(sv)

        ft = fine_type[pick]
        if group_web and ft in WEB:
            contrib = pd.Series(sv[0][:, web_i].sum(axis=1), index=feats)
            target = FAMILY
        else:
            contrib = pd.Series(sv[0][:, classes.index(ft)], index=feats)
            target = ft

        order = contrib.abs().sort_values(ascending=False).head(8).index
        vals = row.iloc[0]

        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"### Verdict: {lookup.loc[pick, 'Predicted type']}")
            st.progress(float(min(max(threat[pick], 0.0), 1.0)),
                        text=f"Threat score {threat[pick]:.0%}")
            st.write(f"Probability of this type: **{lookup.loc[pick, 'Type probability']:.0%}**")
            if labels is not None:
                st.write(f"True label in file: **{lookup.loc[pick, 'True label']}**")

            explain = pd.DataFrame({
                "Feature": order,
                "Value": [float(vals[f]) for f in order],
                "Contribution": [round(float(contrib[f]), 3) for f in order],
                "Effect": ["toward" if contrib[f] > 0 else "against" for f in order],
            })
            st.dataframe(explain, hide_index=True)

            if ft in WEB and not group_web:
                st.warning("The sub-type of a web attack is unreliable: the model "
                           "often confuses Brute Force, XSS and SQL Injection, even "
                           "when it reports high confidence. Treat this as a web attack.")
            if "Destination Port" in list(order[:3]):
                st.info("Destination Port is one of the top reasons. In the lab data "
                        "each attack targeted particular services, but real attackers "
                        "can use any port.")
        with right:
            shown = contrib.loc[order].iloc[::-1]
            labels_txt = [f"{f} = {vals[f]:.3g}" for f in shown.index]
            colors = ["#c0392b" if v > 0 else "#2471a3" for v in shown.values]
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.barh(labels_txt, shown.values, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"Effect on probability of '{target}' (red = toward, blue = against)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ----------------------------------------------------------------- overview
with tab_over:
    left, right = st.columns(2)
    with left:
        st.subheader("Alerts by predicted type")
        if flagged.any():
            st.bar_chart(res[flagged]["Predicted type"].value_counts())
        else:
            st.write("No alerts at this threshold.")
    with right:
        st.subheader("Distribution of threat scores")
        counts, edges = np.histogram(threat, bins=20, range=(0, 1))
        hist = pd.Series(counts, index=[f"{e:.2f}" for e in edges[:-1]])
        st.bar_chart(hist)
        st.caption("Most flows sit near 0 (clearly benign); the mass near 1 is clearly malicious. "
                   "Flows in between are where the threshold choice matters.")

    if labels is not None:
        st.subheader("Flagged share by true label at the current threshold")
        by = (
            pd.DataFrame({"True label": true_disp.values, "flagged": flagged})
            .groupby("True label")["flagged"].agg(["size", "mean"])
            .rename(columns={"size": "Flows", "mean": "Share flagged"})
            .sort_values("Share flagged", ascending=False)
        )
        by["Share flagged"] = by["Share flagged"].round(3)
        st.dataframe(by)
        st.caption("For attack classes this is recall (higher is better). "
                   "For BENIGN it is the false alarm rate (lower is better).")

# ----------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**How it works.** A Random Forest classifier trained on CIC-IDS2017 flow statistics
scores each flow. The threat score is one minus the probability of BENIGN. Explanations
come from SHAP: each bar shows how much a feature moved the probability of the predicted
type, relative to an average flow.

**What the evaluation showed (see the project report for details).**
- On held-out flows from the same lab data, macro-F1 was about 0.89 (5 random splits),
  with roughly 0.15% of benign flows falsely flagged.
- Brute Force, XSS and SQL Injection cannot be reliably told apart from flow statistics
  alone, but together they are detected as a web-attack family very well.
- Adding noise to the features lowers attack detection while false alarms stay flat,
  so degraded input can cause quiet misses.
- Attack types the model never saw in training were mostly not flagged at all.

**Limits to keep in mind.**
- The data comes from a simulated lab network, so real networks may behave differently.
- Some strong features (destination port, TCP window sizes) may reflect how the lab
  attacks were run rather than how attacks look in general.
- This sample contains far more attacks than real traffic. On real traffic, even a
  0.15% false alarm rate would produce many false alerts in absolute terms.
- The tool supports an analyst; it does not replace one.
"""
    )