# AIML Assignment 2 — Breast Cancer Classification

## a. Problem statement
Build and compare five supervised classification algorithms to predict whether a breast tumour observation is **Malignant** or **Benign**. The project reports Accuracy, ROC-AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC), and exposes saved models through a Streamlit application.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569
- **Input features:** 30
- **Target:** `diagnosis` (`Malignant` / `Benign`)
- **File:** `dataset.csv`
- **Split:** 80% training / 20% test, stratified, `random_state=42`.
- **Positive class for Precision, Recall, F1 and AUC:** `Malignant`.

The included `test_data.csv` contains the held-out test observations and their ground-truth target. The Streamlit app evaluates uploaded test data with the saved models and does not retrain them.

## c. GitHub Repository Link
**[](https://github.com/mm2661/ML_Assignment2_2026AJ05416)**

## d. Models used and evaluation metrics

All models use the same stratified hold-out test split. StandardScaler is included in the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree is trained without scaling.

| Model                  |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9737 | 0.9954 |      0.9756 |   0.9524 | 0.9639 | 0.9433 |
| Decision Tree          |     0.9211 | 0.9001 |      0.9024 |   0.8810 | 0.8916 | 0.8297 |
| K-Nearest Neighbors    |     0.9561 | 0.9835 |      0.9744 |   0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes   |     0.9211 | 0.9891 |      0.9231 |   0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine |     0.9825 | 0.9954 |      0.9762 |   0.9762 | 0.9762 | 0.9623 |

### Observations about model performance

| ML Model | Observation about performance |
|---|---|
| Logistic Regression | Achieved 97.37% accuracy and 0.9954 AUC. Its malignant-class recall is 95.24%, showing strong detection performance. The F1-score of 0.9639 and MCC of 0.9433 indicate a strong and balanced classifier. |
| Decision Tree | Achieved 92.11% accuracy. Its AUC (0.9001), recall (88.10%), F1-score (0.8916) and MCC (0.8297) are the lowest among the five models in this experiment. The chosen tree constraints may limit complexity and generalisation. |
| K-Nearest Neighbors | Achieved 95.61% accuracy and 0.9835 AUC. Precision is high at 97.44%, while recall is 90.48%. This means its malignant predictions are usually correct, but it misses more malignant cases than Logistic Regression and SVM. |
| Gaussian Naive Bayes | Achieved 92.11% accuracy and a relatively high AUC of 0.9891. However, recall is 85.71% and F1 is 0.8889, so its thresholded predictions miss more malignant observations than the other models. |
| Support Vector Machine | Achieved the highest accuracy (98.25%), recall (97.62%), F1-score (0.9762) and MCC (0.9623), while sharing the highest AUC (0.9954) with Logistic Regression. It provides the strongest overall balance on this test split. |

**Overall winner: Support Vector Machine (SVM).** It performs best on four of the six reported metrics and ties for the best AUC. Its 97.62% recall means it identifies most malignant observations in this held-out test set. Logistic Regression is a close alternative, with the same AUC and strong performance across the metrics.

### Metric interpretation
- **Accuracy:** overall fraction of correct predictions.
- **AUC:** ability to rank malignant observations above benign observations.
- **Precision:** among observations predicted malignant, fraction actually malignant.
- **Recall:** among actual malignant observations, fraction detected.
- **F1:** harmonic mean of malignant-class precision and recall.
- **MCC:** balanced correlation between predictions and true labels; 1 is perfect, 0 is no better than chance, and -1 is total disagreement.

These are academic results from one fixed 80:20 split, not clinical performance estimates. A different split or dataset may produce different rankings.

## Project structure
```text
aiml_assignment_2/
├── app.py
├── train_models.py
├── dataset.csv
├── test_data.csv
├── model_metrics.csv
├── model_metrics.json
├── feature_columns.json
├── requirements.txt
├── README.md
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

