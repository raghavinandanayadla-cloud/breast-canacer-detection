import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title { font-size:2.1rem; font-weight:700; color:#c0392b; }
  .sub-title  { font-size:1rem; color:#666; margin-bottom:1.2rem; }
  .section-hdr{ font-size:1.25rem; font-weight:600; color:#2c3e50;
                border-bottom:2px solid #c0392b; padding-bottom:4px;
                margin-top:1.4rem; margin-bottom:.8rem; }
  .info-box   { background:#eaf4fb; border-radius:8px;
                padding:12px 16px; margin-bottom:1rem; font-size:.93rem; }
  .pred-box   { padding:18px 24px; border-radius:8px; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Breast Cancer Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dataset Overview",
         "📈 EDA & Visualization",
         "🤖 Model Training",
         "⚙️ Ensemble Methods",
         "🔬 Predict Diagnosis"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Dataset: Breast Cancer Wisconsin\nTask: Binary Classification\nClasses: Malignant / Benign")

# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        raw = load_breast_cancer()
        df  = pd.DataFrame(raw.data, columns=raw.feature_names)
        df.insert(0, "id", range(1, len(df) + 1))
        df.insert(1, "diagnosis", np.where(raw.target == 1, "B", "M"))
    return df

@st.cache_data
def preprocess(_df):
    df = _df.drop_duplicates().copy()
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    X = df.drop(drop_cols + ["diagnosis"], axis=1)
    y = df["diagnosis"]
    feat_names = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=50
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr_sc, y_tr)

    le = LabelEncoder()
    y_res_enc = le.fit_transform(y_res)
    y_te_enc  = le.transform(y_te)

    return (X_tr, X_te, y_tr, y_te,
            X_tr_sc, X_te_sc,
            X_res, y_res, y_res_enc, y_te_enc,
            scaler, le, feat_names)

# ── Plot helpers ───────────────────────────────────────────────────────────────
def plot_roc(y_true, y_prob, title, pos=1):
    yb = (pd.Series(y_true) == pos).astype(int) if pos != 1 else pd.Series(y_true)
    auc = roc_auc_score(yb, y_prob)
    fpr, tpr, _ = roc_curve(yb, y_prob)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig, auc

def plot_cm(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    plt.tight_layout()
    return fig

# ── Load data ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 Breast Cancer Wisconsin – ML Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore data · train models · compare performance · predict diagnoses</div>',
            unsafe_allow_html=True)

with st.expander("📂 Upload your own CSV (optional — defaults to sklearn built-in dataset)"):
    uploaded = st.file_uploader("CSV must contain a 'diagnosis' column (M/B)", type="csv")

df = load_data(uploaded)
(X_tr, X_te, y_tr, y_te,
 X_tr_sc, X_te_sc,
 X_res, y_res, y_res_enc, y_te_enc,
 scaler, le, feat_names) = preprocess(df)

# ══════════════════════════════════════════════════════════════════════════════
# 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples",    df.shape[0])
    c2.metric("Features",   len(feat_names))
    c3.metric("Malignant",  int((df["diagnosis"] == "M").sum()))
    c4.metric("Benign",     int((df["diagnosis"] == "B").sum()))

    st.markdown("#### Sample rows")
    st.dataframe(df.head(10), use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Column info")
        info = pd.DataFrame({
            "dtype":    df.dtypes,
            "non-null": df.notnull().sum(),
            "nulls":    df.isnull().sum(),
        })
        st.dataframe(info, use_container_width=True)
    with cb:
        st.markdown("#### Descriptive statistics")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

    st.markdown("#### Class distribution")
    vc = df["diagnosis"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Benign (B)", "Malignant (M)"], vc.reindex(["B", "M"]).values,
                  color=["#27ae60", "#c0392b"], edgecolor="white", width=0.5)
    for bar, v in zip(bars, vc.reindex(["B", "M"]).values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Count"); ax.set_title("Class Distribution"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    st.pyplot(fig)

    imb = vc.values[0] / vc.values[1]
    st.info(f"**Imbalance ratio:** {imb:.2f}  |  "
            f"Benign: {vc['B']/vc.sum()*100:.1f}%  |  "
            f"Malignant: {vc['M']/vc.sum()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & Visualization":
    st.markdown('<div class="section-hdr">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    num_df = df.select_dtypes(include=np.number)
    t1, t2, t3, t4 = st.tabs(["Distribution", "Correlation", "Box Plots", "Scatter"])

    with t1:
        feat = st.selectbox("Feature", feat_names)
        fig, ax = plt.subplots(figsize=(7, 4))
        for diag, col in [("M", "#c0392b"), ("B", "#27ae60")]:
            ax.hist(df[df["diagnosis"] == diag][feat], bins=30,
                    alpha=0.6, color=col, label=diag, edgecolor="white")
        ax.set(xlabel=feat, ylabel="Count", title=f"Distribution – {feat}")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); st.pyplot(fig)

    with t2:
        top = num_df.var().nlargest(15).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df[top].corr(), annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, ax=ax,
                    linewidths=.4, annot_kws={"size": 7})
        plt.tight_layout(); st.pyplot(fig)

    with t3:
        feat_b = st.selectbox("Feature for box plot", feat_names, key="bp")
        fig, ax = plt.subplots(figsize=(6, 4))
        df.boxplot(column=feat_b, by="diagnosis", ax=ax,
                   boxprops=dict(color="#c0392b"),
                   medianprops=dict(color="black", linewidth=2))
        ax.set(title=f"Box Plot – {feat_b}", xlabel="Diagnosis", ylabel=feat_b)
        plt.suptitle(""); plt.tight_layout(); st.pyplot(fig)

    with t4:
        cx = st.selectbox("X-axis", feat_names, index=0)
        cy = st.selectbox("Y-axis", feat_names, index=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        for diag, col, lab in [("M","#c0392b","Malignant"), ("B","#27ae60","Benign")]:
            sub = df[df["diagnosis"] == diag]
            ax.scatter(sub[cx], sub[cy], alpha=0.5, color=col, label=lab,
                       s=28, edgecolors="white", linewidths=0.3)
        ax.set(xlabel=cx, ylabel=cy, title="Scatter")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 3 — Model Training
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown('<div class="section-hdr">Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        <b>Phase 1</b> – raw features, no scaling/SMOTE &nbsp;|&nbsp;
        <b>Phase 2</b> – StandardScaler + SMOTE applied
    </div>""", unsafe_allow_html=True)

    mdl_name = st.selectbox("Model", [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "SVM – Linear", "SVM – RBF", "SVM – Polynomial", "KNN (k=5)"])
    phase = st.radio("Phase", ["Before Scaling", "After Scaling + SMOTE"], horizontal=True)

    @st.cache_data
    def run_model(name, phase):
        if phase == "Before Scaling":
            Xtr, Xte, ytr, yte = (
                X_tr.values, X_te.values, y_tr.values, y_te.values)
        else:
            Xtr, Xte, ytr, yte = X_res, X_te_sc, y_res.values, y_te.values

        defs = {
            "Logistic Regression":  LogisticRegression(random_state=50, solver="liblinear", max_iter=1000),
            "Decision Tree":        DecisionTreeClassifier(random_state=42),
            "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM – Linear":         LinearSVC(max_iter=5000, random_state=42),
            "SVM – RBF":            SVC(kernel="rbf", random_state=42, probability=True),
            "SVM – Polynomial":     SVC(kernel="poly", random_state=42, probability=True),
            "KNN (k=5)":            KNeighborsClassifier(n_neighbors=5),
        }
        m = defs[name]
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        yprob = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else None
        return yp.tolist(), (yprob.tolist() if yprob is not None else None)

    with st.spinner("Training…"):
        yp, yprob = run_model(mdl_name, phase)

    yp    = np.array(yp)
    yprob = np.array(yprob) if yprob is not None else None
    yte   = y_te.values

    acc  = accuracy_score(yte, yp)
    prec = precision_score(yte, yp, pos_label="M", zero_division=0)
    rec  = recall_score(yte, yp, pos_label="M", zero_division=0)
    f1   = f1_score(yte, yp, pos_label="M", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1-Score",  f"{f1:.3f}")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Confusion Matrix")
        st.pyplot(plot_cm(yte, yp, ["B","M"], mdl_name))
    with cr:
        if yprob is not None:
            st.markdown("#### ROC Curve")
            fig_roc, auc_val = plot_roc(yte, yprob, f"ROC – {mdl_name}", pos="M")
            st.pyplot(fig_roc)
            st.metric("ROC-AUC", f"{auc_val:.3f}")
        else:
            st.info("ROC not available for LinearSVC (no probability output).")

    st.markdown("#### Classification Report")
    st.dataframe(pd.DataFrame(
        classification_report(yte, yp, output_dict=True)).T.round(3),
        use_container_width=True)

    st.markdown('<div class="section-hdr">Comparison Table (from notebook results)</div>',
                unsafe_allow_html=True)
    cmp = pd.DataFrame({
        "Model":              ["Logistic Regression","Decision Tree","Random Forest",
                               "SVM (Linear)","SVM (RBF)","SVM (Poly)","KNN (k=5)"],
        "Accuracy Before":    [0.938, 0.938, 0.965, 0.938, 0.622, 0.614, 0.754],
        "Accuracy After":     [0.980, 0.960, 0.970, 0.960, 0.970, 0.900, 0.960],
    })
    cmp["Δ Improvement"] = ((cmp["Accuracy After"] - cmp["Accuracy Before"]) * 100).map(lambda x: f"+{x:.1f}%")
    st.dataframe(cmp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4 — Ensemble Methods
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Ensemble Methods":
    st.markdown('<div class="section-hdr">Ensemble Learning</div>', unsafe_allow_html=True)

    ens_name = st.selectbox("Ensemble method", [
        "Bagging", "AdaBoost", "XGBoost",
        "Soft Voting", "Hard Voting", "Stacking"])

    @st.cache_data
    def run_ensemble(name):
        lr  = LogisticRegression(random_state=50, solver="liblinear", max_iter=1000)
        rf  = RandomForestClassifier(n_estimators=100, random_state=42)
        svm = SVC(kernel="rbf", random_state=42, probability=True)

        use_enc = name == "XGBoost"
        ytr = y_res_enc if use_enc else y_res.values
        yte = y_te_enc  if use_enc else y_te.values

        ensembles = {
            "Bagging":     BaggingClassifier(
                               estimator=DecisionTreeClassifier(),
                               n_estimators=100, random_state=42),
            "AdaBoost":    AdaBoostClassifier(
                               estimator=DecisionTreeClassifier(max_depth=1),
                               n_estimators=100, random_state=42),
            "XGBoost":     xgb.XGBClassifier(
                               objective="binary:logistic",
                               eval_metric="logloss", random_state=42),
            "Soft Voting": VotingClassifier(
                               estimators=[("lr",lr),("rf",rf),("svm",svm)],
                               voting="soft"),
            "Hard Voting": VotingClassifier(
                               estimators=[("lr",lr),("rf",rf),("svm",svm)],
                               voting="hard"),
            "Stacking":    StackingClassifier(
                               estimators=[
                                   ("lr", LogisticRegression(max_iter=1000)),
                                   ("rf", RandomForestClassifier(n_estimators=50)),
                                   ("svm", SVC(probability=True))],
                               final_estimator=LogisticRegression(max_iter=1000),
                               cv=5),
        }
        m = ensembles[name]
        m.fit(X_res, ytr)
        yp    = m.predict(X_te_sc)
        yprob = m.predict_proba(X_te_sc)[:, 1] if hasattr(m, "predict_proba") and name != "Hard Voting" else None
        return yp.tolist(), (yprob.tolist() if yprob is not None else None), yte.tolist(), use_enc

    with st.spinner(f"Training {ens_name}…"):
        yp_e, yprob_e, yte_e, use_enc = run_ensemble(ens_name)

    yp_e    = np.array(yp_e)
    yprob_e = np.array(yprob_e) if yprob_e is not None else None
    yte_e   = np.array(yte_e)
    pos     = 1 if use_enc else "M"

    acc  = accuracy_score(yte_e, yp_e)
    prec = precision_score(yte_e, yp_e, pos_label=pos, zero_division=0)
    rec  = recall_score(yte_e, yp_e, pos_label=pos, zero_division=0)
    f1   = f1_score(yte_e, yp_e, pos_label=pos, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1-Score",  f"{f1:.3f}")

    lab_map = (["0","1"] if use_enc else ["B","M"])
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Confusion Matrix")
        st.pyplot(plot_cm(yte_e, yp_e, lab_map, ens_name))
    with cr:
        if yprob_e is not None:
            st.markdown("#### ROC Curve")
            yb = yte_e if use_enc else (yte_e == "M").astype(int)
            auc_e = roc_auc_score(yb, yprob_e)
            fpr, tpr, _ = roc_curve(yb, yprob_e)
            fig, ax = plt.subplots(figsize=(5, 3.8))
            ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc_e:.3f}")
            ax.plot([0,1],[0,1],"--",color="grey",lw=1)
            ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC – {ens_name}")
            ax.legend(); ax.grid(alpha=.3)
            plt.tight_layout()
            st.pyplot(fig)
            st.metric("ROC-AUC", f"{auc_e:.3f}")
        else:
            st.info("Probability output not available for Hard Voting.")

    st.markdown("#### Classification Report")
    st.dataframe(pd.DataFrame(
        classification_report(yte_e, yp_e, output_dict=True)).T.round(3),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict Diagnosis":
    st.markdown('<div class="section-hdr">Predict New Diagnosis</div>', unsafe_allow_html=True)
    st.write("Adjust the sliders / inputs to match patient measurements, then click **Predict**.")

    pred_mdl = st.selectbox("Model", [
        "Logistic Regression", "Random Forest", "SVM – RBF", "XGBoost"])

    @st.cache_data
    def get_pred_model(name):
        use_enc = name == "XGBoost"
        ytr = y_res_enc if use_enc else y_res.values
        models = {
            "Logistic Regression": LogisticRegression(random_state=50, solver="liblinear", max_iter=1000),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM – RBF":           SVC(kernel="rbf", random_state=42, probability=True),
            "XGBoost":             xgb.XGBClassifier(
                                       objective="binary:logistic",
                                       eval_metric="logloss", random_state=42),
        }
        m = models[name]
        m.fit(X_res, ytr)
        return m, use_enc

    pm, use_enc = get_pred_model(pred_mdl)

    num_df  = df.select_dtypes(include=np.number)
    medians = num_df[feat_names].median()
    mins    = num_df[feat_names].min()
    maxs    = num_df[feat_names].max()

    st.markdown("#### Feature Values")
    cols_per_row = 3
    chunks = [feat_names[i:i+cols_per_row] for i in range(0, len(feat_names), cols_per_row)]
    inp = {}
    for chunk in chunks:
        cols = st.columns(cols_per_row)
        for col, f in zip(cols, chunk):
            inp[f] = col.number_input(
                f, value=float(round(medians[f], 4)),
                min_value=float(mins[f]),
                max_value=float(maxs[f] * 2),
                format="%.4f", key=f"i_{f}")

    if st.button("🔬 Predict Diagnosis", use_container_width=True):
        arr    = np.array([[inp[f] for f in feat_names]])
        arr_sc = scaler.transform(arr)
        pred   = pm.predict(arr_sc)[0]
        prob   = pm.predict_proba(arr_sc)[0]

        if use_enc:
            label = "Malignant (M)" if pred == 1 else "Benign (B)"
            conf  = prob[pred]
        else:
            label = "Malignant (M)" if pred == "M" else "Benign (B)"
            conf  = prob[list(pm.classes_).index(pred)]

        color = "#c0392b" if "Malignant" in label else "#27ae60"
        st.markdown(f"""
        <div class="pred-box" style="background:{color}18;border-left:6px solid {color};">
          <h2 style="color:{color};margin:0">Prediction: {label}</h2>
          <p style="margin:6px 0 0">Confidence: <b>{conf*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([""], [conf],            color=color, height=0.5)
        ax.barh([""], [1 - conf], left=[conf], color="#e0e0e0", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0,.25,.5,.75,1])
        ax.set_xticklabels(["0%","25%","50%","75%","100%"])
        ax.set_title("Confidence"); ax.grid(axis="x", alpha=.3)
        plt.tight_layout(); st.pyplot(fig)

        st.markdown("#### Class Probabilities")
        if use_enc:
            prob_df = pd.DataFrame({"Class": ["Benign","Malignant"], "Probability": [prob[0], prob[1]]})
        else:
            cls_map = {c: "Malignant" if c == "M" else "Benign" for c in pm.classes_}
            prob_df = pd.DataFrame({"Class": [cls_map[c] for c in pm.classes_], "Probability": prob})
        st.dataframe(prob_df.round(4), use_container_width=True, hide_index=True)
