import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


def load_data(filepath="data/breast_cancer_cleaned.csv"):
    """Load cleaned CSV from previous task or fallback to sklearn dataset."""
    if os.path.exists(filepath):
        print(f"Loading cleaned dataset from: {filepath}")
        df = pd.read_csv(filepath)
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        print("Cleaned CSV not found locally. Loading dataset directly from scikit-learn...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    return X, y


def get_models():
    """Define candidate classification models wrapped in Pipelines where scaling is needed."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
    }
    return models


def evaluate_models(X, y):
    """Run Stratified K-Fold Cross-Validation and Test Set Evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = []
    trained_models = {}

    print("\n--- Running 5-Fold Stratified Cross-Validation & Test Evaluation ---")
    for name, model in models.items():
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring_metrics, n_jobs=-1
        )

        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "CV Accuracy (Mean ± Std)": f"{cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}",
            "CV F1-Score": round(cv_results["test_f1"].mean(), 4),
            "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Test Precision": round(precision_score(y_test, y_pred), 4),
            "Test Recall": round(recall_score(y_test, y_pred), 4),
            "Test F1-Score": round(f1_score(y_test, y_pred), 4),
            "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        })

    results_df = pd.DataFrame(results).sort_values(by="Test F1-Score", ascending=False)
    return results_df, trained_models, X_test, y_test


def save_visualizations(results_df, best_model, best_model_name, X_test, y_test):
    """Generate and save model comparison bar chart and confusion matrix."""
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 5))
    metrics_to_plot = results_df.melt(
        id_vars="Model",
        value_vars=["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=metrics_to_plot, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.ylim(0.85, 1.02)
    plt.title("Model Performance Comparison on Held-Out Test Set")
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/model_comparison_chart.png", dpi=300)
    plt.close()

    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
    )
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("reports/best_model_confusion_matrix.png", dpi=300)
    plt.close()

    print("\nSaved charts to 'reports/model_comparison_chart.png' and 'reports/best_model_confusion_matrix.png'.")


if __name__ == "__main__":
    X, y = load_data()
    results_df, trained_models, X_test, y_test = evaluate_models(X, y)

    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print(f"\nSelected Best Model: {best_model_name}")
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["Malignant (0)", "Benign (1)"]))

    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_metrics_summary.csv", index=False)
    save_visualizations(results_df, best_model, best_model_name, X_test, y_test)
