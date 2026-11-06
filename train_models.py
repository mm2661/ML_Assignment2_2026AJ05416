"""
Train five classifiers for the AIML Assignment 2.
Run: python train_models.py
Dataset: dataset.csv (Wisconsin Diagnostic Breast Cancer dataset)
"""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "dataset.csv")
TARGET = "diagnosis"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Fixed, stratified split so that all models are evaluated on identical observations.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Keep the positive class consistent for precision, recall, F1 and ROC-AUC.
POSITIVE_CLASS = "Malignant"

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced",
                                          random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance"))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(C=1.0, kernel="rbf", probability=True,
                           class_weight="balanced", random_state=42))
    ])
}

results = []
for display_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Obtain the probability column corresponding to Malignant.
    class_order = list(model.classes_)
    positive_index = class_order.index(POSITIVE_CLASS)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    scores = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score((y_test == POSITIVE_CLASS).astype(int), probabilities),
        "Precision": precision_score(y_test, predictions, pos_label=POSITIVE_CLASS,
                                     zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label=POSITIVE_CLASS,
                               zero_division=0),
        "F1": f1_score(y_test, predictions, pos_label=POSITIVE_CLASS,
                       zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }
    results.append(scores)
    filename = display_name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)
    print(f"Saved {display_name}: {filename}")

# Save the exact held-out test observations for the Streamlit demonstration.
test_output = X_test.copy()
test_output[TARGET] = y_test
test_output.to_csv(ROOT / "test_data.csv", index=False)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(ROOT / "model_metrics.csv", index=False)
with open(ROOT / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(ROOT / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=2)

print("\nEvaluation on the held-out test set (Malignant is the positive class):")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nBest model by F1:", metrics_df.sort_values("F1", ascending=False).iloc[0]["Model"])
