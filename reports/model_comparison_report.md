# Machine Learning Model Selection and Evaluation Report

## 1. Objective & Problem Framing
The goal of this task is to evaluate and compare multiple supervised classification algorithms on the **Breast Cancer Wisconsin Diagnostic Dataset** to accurately distinguish between malignant (`0`) and benign (`1`) tumors. Because medical diagnostic tasks carry asymmetrical error costs—missing a malignant tumor (False Negative) is far riskier than flagging a benign case for secondary screening (False Positive)—model evaluation prioritizes **Recall**, **F1-Score**, and **ROC-AUC** alongside overall **Accuracy**.

---

## 2. Methodology & Overfitting Prevention
1. **Stratified Train-Test Split (80/20):** Preserves the exact class distribution (~63% Benign, ~37% Malignant) in both training and test sets.
2. **Leakage-Free Pipelines:** Feature scaling (`StandardScaler`) is encapsulated inside `scikit-learn` `Pipeline` objects for scale-sensitive models (Logistic Regression and SVM) so scaling statistics are computed strictly on training folds.
3. **5-Fold Stratified Cross-Validation:** Evaluates generalization stability across five distinct data splits before touching the held-out test set, ensuring high test performance is not due to lucky data partitioning or overfitting.

---

## 3. Candidate Models Evaluated
- **Logistic Regression:** Linear baseline classifier; fast, interpretable, and effective when classes are nearly linearly separable.
- **Decision Tree (`max_depth=5`):** Non-linear rule-based model; easy to visualize, though prone to higher variance without depth constraints.
- **Random Forest (`n_estimators=100`):** Bagging ensemble of decision trees that reduces variance and resists overfitting.
- **Gradient Boosting (`n_estimators=100`):** Sequential boosting ensemble that iteratively corrects residual classification errors.
- **Support Vector Machine (RBF Kernel):** Finds a non-linear maximum-margin hyperplane in high-dimensional feature space.

---

## 4. Quantitative Performance Comparison

| Model | CV Accuracy (Mean ± Std) | CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9758 ± 0.0082 | 0.9810 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9957 |
| **Support Vector Machine (RBF)** | 0.9714 ± 0.0112 | 0.9773 | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9964 |
| **Gradient Boosting** | 0.9538 ± 0.0189 | 0.9631 | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9891 |
| **Random Forest** | 0.9626 ± 0.0179 | 0.9699 | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9942 |
| **Decision Tree** | 0.9231 ± 0.0120 | 0.9372 | 0.9211 | 0.9565 | 0.9167 | 0.9362 | 0.9163 |

---

## 5. Key Findings & Model Justification

### Winner: Logistic Regression
1. **Superior Generalization & Low Variance:** Logistic Regression achieved top cross-validation stability and held-out test F1-Score (**0.9861**), with minimal gap between CV and test scores—confirming strong generalization without overfitting.
2. **Why Linear/Margin Models Outperform Trees Here:** The 30 continuous geometric cell-nucleus features exhibit strong linear and smooth boundary separability after standardization. Axis-aligned splits in single Decision Trees struggle to approximate diagonal boundaries efficiently, resulting in lower accuracy and higher fold-to-fold variance.
3. **Clinical Interpretability:** Unlike black-box ensembles, Logistic Regression provides explicit feature weights (log-odds coefficients) and well-calibrated probability estimates (`predict_proba`), allowing clinicians to inspect which tumor characteristics drive a prediction and adjust decision thresholds if higher sensitivity is required.
