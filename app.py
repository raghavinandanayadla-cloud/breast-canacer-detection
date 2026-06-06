import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

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

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#c0392b; }
    .sub-title   { font-size:1.1rem; color:#555; margin-bottom:1rem; }
    .metric-box  { background:#f8f9fa; border-left:4px solid #c0392b;
                   padding:12px 18px; border-radius:6px; margin-bottom:8px; }
    .section-hdr { font-size:1.3rem; font-weight:600; color:#2c3e50;
                   border-bottom:2px solid #c0392b; padding-bottom:4px; margin-top:1.5rem; }
    .info-box    { background:#eaf4fb; border-radius:8px;
                   padding:14px 18px; margin-bottom:1rem; }
    .stButton>button { background:#c0392b; color:white;
                       border:none; border-radius:6px; padding:8px 22px; }
    .stButton>button:hover { background:#a93226; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.markdown("## 🩺 Breast Cancer Classifier")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["📊 Dataset Overview",
         "📈 EDA & Visualization",
         "🤖 Model Training",
         "⚙️ Ensemble Methods",
         "🔬 Predict Diagnosis"]
    )
    st.markdown("---")
    st.markdown("**Dataset:** Breast Cancer Wisconsin")
    st.markdown("**Task:** Binary Classification")
    st.markdown("**Classes:** Malignant / Benign")

# ─── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df.insert(1, "diagnosis", np.where(data.target == 1, "B", "M"))
        df.insert(0, "id", range(1, len(df) + 1))
    return df

@st.cache_data
def preprocess(df):
    df = df.drop_duplicates()
    drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    X = df.drop(drop_cols + ["diagnosis"], axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=50
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    le = LabelEncoder()
    y_resampled_enc = le.fit_transform(y_resampled)
    y_test_enc      = le.transform(y_test)

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            X_resampled, y_resampled, y_resampled_enc, y_test_enc,
            scaler, le, X.columns.tolist())

# ─── Helper: Metrics ───────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob=None, pos_label="M"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    auc  = None
    if y_prob is not None:
        y_bin = (pd.Series(y_true) == pos_label).astype(int)
        auc   = roc_auc_score(y_bin, y_prob)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc}

def plot_roc(y_true, y_prob, title, pos_label="M"):
    y_bin  = (pd.Series(y_true) == pos_label).astype(int)
    auc    = roc_auc_score(y_bin, y_prob)
    fpr, tpr, _ = roc_curve(y_bin, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="grey")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig

def plot_cm(y_true, y_pred, title):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                xticklabels=["Benign","Malignant"],
                yticklabels=["Benign","Malignant"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    plt.tight_layout()
    return fig

# ─── File Uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 Breast Cancer Wisconsin – ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore data, train models, compare performance, and predict diagnoses.</div>', unsafe_allow_html=True)

with st.expander("📂 Upload your own dataset (optional)", expanded=False):
    uploaded = st.file_uploader("Upload data.csv (must contain 'diagnosis' column)", type="csv")

df = load_data(uploaded)
(X_train, X_test, y_train, y_test,
 X_train_scaled, X_test_scaled,
 X_resampled, y_resampled, y_resampled_enc, y_test_enc,
 scaler, le, feature_names) = preprocess(df)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Malignant", int((df["diagnosis"] == "M").sum()))
    col4.metric("Benign",    int((df["diagnosis"] == "B").sum()))

    st.markdown("#### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Data Types & Null Values")
        info_df = pd.DataFrame({
            "dtype": df.dtypes,
            "non-null": df.notnull().sum(),
            "null": df.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True)
    with col_b:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

    st.markdown("#### Class Distribution")
    vc = df["diagnosis"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Benign (B)", "Malignant (M)"], vc.values,
                  color=["#2ecc71", "#c0392b"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontweight="bold")
    ax.set_ylabel("Count"); ax.set_title("Class Distribution"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    st.pyplot(fig)

    imb = vc.iloc[0] / vc.iloc[1]
    st.info(f"**Class Imbalance Ratio:** {imb:.2f}  |  "
            f"Benign: {vc.values[0]/vc.sum()*100:.1f}%  |  "
            f"Malignant: {vc.values[1]/vc.sum()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA & Visualization
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & Visualization":
    st.markdown('<div class="section-hdr">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=np.number)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Box Plots", "Feature Comparison"])

    with tab1:
        feat = st.selectbox("Select feature", feature_names, key="dist_feat")
        fig, ax = plt.subplots(figsize=(7, 4))
        for diag, color in [("M", "#c0392b"), ("B", "#2ecc71")]:
            subset = df[df["diagnosis"] == diag][feat]
            ax.hist(subset, bins=30, alpha=0.6, label=diag, color=color, edgecolor="white")
        ax.set(xlabel=feat, ylabel="Count", title=f"Distribution of {feat}")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.markdown("**Correlation Heatmap** (Top 15 features by variance)")
        top_feats = numeric_df.var().nlargest(15).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df[top_feats].corr(), annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, ax=ax, linewidths=.5,
                    annot_kws={"size": 7})
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        feat_box = st.selectbox("Select feature for box plot", feature_names, key="box_feat")
        fig, ax = plt.subplots(figsize=(6, 4))
        df.boxplot(column=feat_box, by="diagnosis", ax=ax,
                   boxprops=dict(color="#c0392b"),
                   medianprops=dict(color="black", linewidth=2))
        ax.set(title=f"Box Plot – {feat_box}", xlabel="Diagnosis", ylabel=feat_box)
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        col_x = st.selectbox("X-axis feature", feature_names, index=0)
        col_y = st.selectbox("Y-axis feature", feature_names, index=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        for diag, color in [("M", "#c0392b"), ("B", "#2ecc71")]:
            subset = df[df["diagnosis"] == diag]
            ax.scatter(subset[col_x], subset[col_y], alpha=0.5,
                       label=f"{'Malignant' if diag=='M' else 'Benign'}",
                       color=color, s=30, edgecolors="white", linewidths=0.3)
        ax.set(xlabel=col_x, ylabel=col_y, title="Feature Scatter Plot")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Training
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown('<div class="section-hdr">Model Training & Evaluation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Models are trained in two phases:<br>
    &nbsp;1. <b>Before Scaling</b> – raw features, original class distribution<br>
    &nbsp;2. <b>After Scaling + SMOTE</b> – StandardScaler + synthetic over-sampling
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox("Choose a model to inspect",
        ["Logistic Regression", "Decision Tree", "Random Forest",
         "SVM – Linear", "SVM – RBF", "SVM – Polynomial", "KNN"])

    phase = st.radio("Training phase", ["Before Scaling", "After Scaling + SMOTE"], horizontal=True)

    @st.cache_resource
    def train_model(name, phase):
        if phase == "Before Scaling":
            Xtr, Xte, ytr, yte = X_train, X_test, y_train, y_test
        else:
            Xtr, Xte, ytr, yte = X_resampled, X_test_scaled, y_resampled, y_test

        models = {
            "Logistic Regression": LogisticRegression(random_state=50, solver="liblinear", max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM – Linear":        LinearSVC(max_iter=5000, random_state=42),
            "SVM – RBF":           SVC(kernel="rbf", random_state=42, probability=True),
            "SVM – Polynomial":    SVC(kernel="poly", random_state=42, probability=True),
            "KNN":                 KNeighborsClassifier(n_neighbors=5),
        }
        m = models[name]
        m.fit(Xtr, ytr)
        y_pred = m.predict(Xte)
        y_prob = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else None
        return m, y_pred, y_prob

    with st.spinner("Training model…"):
        model, y_pred, y_prob = train_model(model_choice, phase)

    metrics = get_metrics(y_test, y_pred, y_prob)

    st.markdown("#### Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (k, v) in zip([c1,c2,c3,c4,c5], metrics.items()):
        if v is not None:
            col.metric(k, f"{v:.3f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Confusion Matrix")
        st.pyplot(plot_cm(y_test, y_pred, f"{model_choice} – {phase}"))
    with col_right:
        if y_prob is not None:
            st.markdown("#### ROC Curve")
            st.pyplot(plot_roc(y_test, y_prob, f"ROC – {model_choice}"))
        else:
            st.info("ROC curve not available for LinearSVC (no probability output).")

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Model Comparison Table</div>', unsafe_allow_html=True)
    comparison = {
        "Model":              ["Logistic Regression","Decision Tree","Random Forest",
                               "SVM (Linear)","SVM (RBF)","SVM (Poly)","KNN (k=5)"],
        "Accuracy (Before)":  [0.938, 0.938, 0.965, 0.938, 0.622, 0.614, 0.754],
        "Accuracy (After)":   [0.980, 0.960, 0.970, 0.960, 0.970, 0.900, 0.960],
    }
    comp_df = pd.DataFrame(comparison)
    comp_df["Improvement"] = (comp_df["Accuracy (After)"] - comp_df["Accuracy (Before)"]).map(lambda x: f"+{x*100:.1f}%")
    st.dataframe(comp_df.style.highlight_max(subset=["Accuracy (After)"], color="#c8f7c5"),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Ensemble Methods
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Ensemble Methods":
    st.markdown('<div class="section-hdr">Ensemble Learning</div>', unsafe_allow_html=True)

    ens_choice = st.selectbox("Choose an ensemble method",
        ["Bagging", "AdaBoost", "XGBoost", "Soft Voting", "Hard Voting", "Stacking"])

    @st.cache_resource
    def train_ensemble(name):
        lr  = LogisticRegression(random_state=50, solver="liblinear", max_iter=1000)
        rf  = RandomForestClassifier(n_estimators=100, random_state=42)
        svm = SVC(kernel="rbf", random_state=42, probability=True)

        ensembles = {
            "Bagging":      BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
            "AdaBoost":     AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42),
            "XGBoost":      xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, use_label_encoder=False),
            "Soft Voting":  VotingClassifier(estimators=[("lr",lr),("rf",rf),("svm",svm)], voting="soft"),
            "Hard Voting":  VotingClassifier(estimators=[("lr",lr),("rf",rf),("svm",svm)], voting="hard"),
            "Stacking":     StackingClassifier(
                                estimators=[("lr",LogisticRegression(max_iter=1000)),
                                            ("rf",RandomForestClassifier()),
                                            ("svm",SVC(probability=True))],
                                final_estimator=LogisticRegression(max_iter=1000), cv=5),
        }
        Xtr = X_resampled
        ytr = y_resampled_enc if name in ["XGBoost"] else y_resampled
        yte = y_test_enc      if name in ["XGBoost"] else y_test

        m = ensembles[name]
        m.fit(Xtr, ytr)
        y_pred = m.predict(X_test_scaled)
        y_prob = m.predict_proba(X_test_scaled)[:, 1] if hasattr(m, "predict_proba") and name != "Hard Voting" else None

        if name == "XGBoost":
            y_true_eval = y_test_enc
        else:
            y_true_eval = y_test

        return m, y_pred, y_prob, y_true_eval

    with st.spinner(f"Training {ens_choice}…"):
        ens_model, y_pred_ens, y_prob_ens, y_true_ens = train_ensemble(ens_choice)

    # For XGBoost results need numeric pos_label
    pos = 1 if ens_choice == "XGBoost" else "M"
    acc  = accuracy_score(y_true_ens, y_pred_ens)
    prec = precision_score(y_true_ens, y_pred_ens, pos_label=pos, zero_division=0)
    rec  = recall_score(y_true_ens, y_pred_ens, pos_label=pos, zero_division=0)
    f1   = f1_score(y_true_ens, y_pred_ens, pos_label=pos, zero_division=0)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1-Score",  f"{f1:.3f}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Confusion Matrix")
        st.pyplot(plot_cm(y_true_ens, y_pred_ens, f"{ens_choice} – Confusion Matrix"))
    with col_r:
        if y_prob_ens is not None:
            st.markdown("#### ROC Curve")
            yb = (pd.Series(y_true_ens) == pos).astype(int)
            auc_val = roc_auc_score(yb, y_prob_ens)
            st.pyplot(plot_roc(y_true_ens, y_prob_ens, f"ROC – {ens_choice}", pos_label=pos))
            st.metric("ROC-AUC", f"{auc_val:.3f}")
        else:
            st.info("Probability output not available for Hard Voting.")

    st.markdown("#### Classification Report")
    report_ens = classification_report(y_true_ens, y_pred_ens, output_dict=True)
    st.dataframe(pd.DataFrame(report_ens).T.round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Predict Diagnosis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Predict Diagnosis":
    st.markdown('<div class="section-hdr">Predict New Diagnosis</div>', unsafe_allow_html=True)
    st.markdown("Enter feature values below to predict whether a tumour is **Malignant** or **Benign**.")

    pred_model_name = st.selectbox("Select model for prediction",
        ["Logistic Regression", "Random Forest", "SVM – RBF", "XGBoost"])

    @st.cache_resource
    def get_pred_model(name):
        if name == "XGBoost":
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                   random_state=42, use_label_encoder=False)
            m.fit(X_resampled, y_resampled_enc)
        else:
            models = {
                "Logistic Regression": LogisticRegression(random_state=50, solver="liblinear", max_iter=1000),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "SVM – RBF":           SVC(kernel="rbf", random_state=42, probability=True),
            }
            m = models[name]
            m.fit(X_resampled, y_resampled)
        return m

    pred_model = get_pred_model(pred_model_name)

    st.markdown("#### Enter Feature Values")
    # Use median values from dataset as defaults
    numeric_df = df.select_dtypes(include=np.number)
    medians    = numeric_df[feature_names].median()

    input_vals = {}
    cols_per_row = 3
    feat_chunks  = [feature_names[i:i+cols_per_row] for i in range(0, len(feature_names), cols_per_row)]
    for chunk in feat_chunks:
        cols = st.columns(cols_per_row)
        for col, feat in zip(cols, chunk):
            input_vals[feat] = col.number_input(feat, value=float(round(medians[feat], 4)),
                                                 format="%.4f", key=f"inp_{feat}")

    if st.button("🔬 Predict Diagnosis"):
        input_array = np.array([[input_vals[f] for f in feature_names]])
        input_scaled = scaler.transform(input_array)

        pred_raw = pred_model.predict(input_scaled)[0]
        if pred_model_name == "XGBoost":
            label = "Malignant (M)" if pred_raw == 1 else "Benign (B)"
            prob_val = pred_model.predict_proba(input_scaled)[0][pred_raw]
        else:
            label    = "Malignant (M)" if pred_raw == "M" else "Benign (B)"
            prob_val = pred_model.predict_proba(input_scaled)[0].max() if hasattr(pred_model, "predict_proba") else None

        colour = "#c0392b" if "Malignant" in label else "#27ae60"
        st.markdown(f"""
        <div style="background:{colour}22; border-left:6px solid {colour};
                    padding:18px 24px; border-radius:8px; margin-top:1rem;">
            <h2 style="color:{colour}; margin:0;">Prediction: {label}</h2>
            {'<p style="margin:6px 0 0;">Confidence: <b>' + f"{prob_val*100:.1f}%" + '</b></p>' if prob_val else ''}
        </div>
        """, unsafe_allow_html=True)

        if prob_val is not None:
            fig, ax = plt.subplots(figsize=(5, 1.5))
            ax.barh(["Confidence"], [prob_val], color=colour, height=0.4)
            ax.barh(["Confidence"], [1 - prob_val], left=[prob_val], color="#eee", height=0.4)
            ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"])
            ax.set_title("Prediction Confidence"); ax.grid(axis="x", alpha=.3)
            plt.tight_layout()
            st.pyplot(fig)
