"""
Interactive model evaluation app.
Run locally: streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGET = "diagnosis"
POSITIVE_CLASS = "Malignant"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.set_page_config(page_title="AIML Classification Lab", page_icon="🧪",
                   layout="wide")
st.title("🧪 Breast Cancer Classification — Model Evaluation")
st.caption("AIML Assignment 2 | Five trained classifiers | UCI Wisconsin Diagnostic dataset")

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python train_models.py` before starting the app."
            )
        loaded[name] = joblib.load(path)
    return loaded

@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "test_data.csv")

try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Experiment settings")
    selected_model = st.selectbox("Choose a classification model",
                                  list(MODEL_FILES.keys()))
    st.markdown("---")
    st.write("**Positive class:** Malignant")
    st.write("**Evaluation:** Uploaded held-out test data")
    st.info("The app loads saved models. It does not retrain on uploaded data.")

st.subheader("1. Upload test data")
uploaded_file = st.file_uploader("Upload a CSV containing all 30 features and the diagnosis target",
                                 type=["csv"])

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Load included sample test data", use_container_width=True):
        st.session_state["use_sample"] = True
with col_b:
    if st.button("Clear sample selection", use_container_width=True):
        st.session_state["use_sample"] = False

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif st.session_state.get("use_sample", False):
    test_df = load_sample()
else:
    st.info("Upload a test CSV, or click **Load included sample test data** to try the app.")
    st.download_button("Download sample test_data.csv",
                       data=(ROOT / "test_data.csv").read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    st.stop()

if TARGET not in test_df.columns:
    st.error(f"The CSV must include the target column named `{TARGET}`.")
    st.stop()

model = models[selected_model]
feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
if feature_names is None:
    try:
        feature_names = json.loads((ROOT / "feature_columns.json").read_text())
    except Exception:
        st.error("Could not determine the expected feature columns.")
        st.stop()

missing = [c for c in feature_names if c not in test_df.columns]
extra = [c for c in test_df.columns if c not in feature_names and c != TARGET]
if missing:
    st.error("Missing required feature columns: " + ", ".join(missing))
    st.stop()
if test_df[feature_names].isnull().any().any():
    st.error("The uploaded feature data contains missing values. Please clean the CSV first.")
    st.stop()

X_test = test_df[feature_names]
predictions = model.predict(X_test)
has_labels = test_df[TARGET].notna().all() and set(test_df[TARGET].unique()).issubset(
    {"Malignant", "Benign"}
)

st.subheader("2. Test data preview")
st.write(f"Rows: **{len(test_df):,}** | Features used: **{len(feature_names)}**")
st.dataframe(test_df.head(10), use_container_width=True)

st.subheader(f"3. Results — {selected_model}")
pred_df = pd.DataFrame({"Predicted diagnosis": predictions})
if hasattr(model, "predict_proba"):
    class_order = list(model.classes_)
    positive_index = class_order.index(POSITIVE_CLASS)
    pred_df["Malignant probability"] = model.predict_proba(X_test)[:, positive_index]
st.dataframe(pred_df.head(20), use_container_width=True)

if has_labels:
    y_true = test_df[TARGET]
    class_order = list(model.classes_)
    positive_index = class_order.index(POSITIVE_CLASS)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "AUC": roc_auc_score((y_true == POSITIVE_CLASS).astype(int), probabilities),
        "Precision": precision_score(y_true, predictions, pos_label=POSITIVE_CLASS,
                                     zero_division=0),
        "Recall": recall_score(y_true, predictions, pos_label=POSITIVE_CLASS,
                               zero_division=0),
        "F1 Score": f1_score(y_true, predictions, pos_label=POSITIVE_CLASS,
                             zero_division=0),
        "MCC": matthews_corrcoef(y_true, predictions)
    }
    metric_cols = st.columns(6)
    for col, (label, value) in zip(metric_cols, metrics.items()):
        col.metric(label, f"{value:.4f}")

    st.subheader("4. Confusion matrix")
    cm = confusion_matrix(y_true, predictions, labels=["Benign", "Malignant"])
    cm_df = pd.DataFrame(cm, index=["Actual Benign", "Actual Malignant"],
                         columns=["Predicted Benign", "Predicted Malignant"])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("5. Classification report")
    report = classification_report(y_true, predictions, labels=["Benign", "Malignant"],
                                   zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
else:
    st.warning("No valid ground-truth labels found. Predictions are shown, but evaluation metrics "
               "cannot be calculated. Include diagnosis values (Benign/Malignant) to evaluate.")

if extra:
    with st.expander("Additional columns ignored"):
        st.write(", ".join(extra))

st.markdown("---")
st.caption("Educational demonstration only. This model is not a medical diagnostic tool.")
