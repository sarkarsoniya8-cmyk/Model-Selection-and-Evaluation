# Model Selection and Evaluation

Internship task comparing multiple classification algorithms on the Breast Cancer Wisconsin Diagnostic Dataset using 5-Fold Stratified Cross-Validation and held-out test evaluation.

## Repository Structure
- `data/breast_cancer_cleaned.csv`: Cleaned dataset with IQR outlier capping.
- `src/model_selection.py`: End-to-end training, cross-validation, evaluation, and chart generation script.
- `reports/model_comparison_report.md`: Full comparative analysis and model selection justification.
- `reports/model_metrics_summary.csv`: Computed CV and test metrics table.
- `reports/model_comparison_chart.png`: Bar chart comparing model test metrics.
- `reports/best_model_confusion_matrix.png`: Confusion matrix for the winning classifier.

## How to Run
```bash
pip install -r requirements.txt
python src/model_selection.py
```
