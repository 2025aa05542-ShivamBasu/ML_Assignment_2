# ML Assignment 2

> Note: This same README content should be included in the submitted PDF file.

## a. Problem statement

Build and evaluate a multi-class obesity-level classification system using machine learning.
The task is to train and compare six models on the Obesity dataset and analyze their performance
using standard classification metrics.

## b. Dataset description [1 mark]

- **Dataset name:** ObesityDataSet_raw_and_data_sinthetic.csv
- **Default path in project:** `data/ObesityDataSet_raw_and_data_sinthetic.csv`
- **Rows:** 2111
- **Columns:** 17 (16 input features + 1 target)
- **Target variable:** `NObeyesdad` (7 classes)
- **Feature types:** demographic, physical, lifestyle, food-consumption, and activity-related attributes.

If needed, place/copy the dataset with:

```bash
python scripts/fetch_dataset.py --local <path_to_csv>
```

## c. Models used [6 marks - 1 mark for all the metrics for each model]

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8132 | 0.9641 | 0.8062 | 0.8085 | 0.8057 | 0.7824 |
| Decision Tree | 0.9054 | 0.9437 | 0.9060 | 0.9032 | 0.9041 | 0.8897 |
| kNN | 0.8061 | 0.9467 | 0.8122 | 0.8005 | 0.7947 | 0.7772 |
| Naive Bayes | 0.6028 | 0.9074 | 0.6488 | 0.5982 | 0.5778 | 0.5474 |
| Random Forest (Ensemble) | 0.9574 | 0.9972 | 0.9606 | 0.9561 | 0.9571 | 0.9507 |
| XGBoost (Ensemble) | 0.9504 | 0.9968 | 0.9509 | 0.9487 | 0.9493 | 0.9422 |

### Observations on model performance [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with good AUC, but lower overall accuracy than tree ensembles; likely limited by linear decision boundaries and convergence at `max_iter=1000`. |
| Decision Tree | Good performance and interpretability; captures non-linear patterns better than linear models, but is generally more prone to overfitting than ensembles. |
| kNN | Competitive but slightly below Logistic Regression/Decision Tree; sensitive to feature scaling and local neighborhood structure in multi-class data. |
| Naive Bayes | Lowest performer across all metrics; conditional-independence assumption is likely too strong for this dataset’s correlated lifestyle/physical features. |
| Random Forest (Ensemble) | Best overall model in this experiment with highest Accuracy, AUC, F1, and MCC; robust generalization due to ensemble averaging. |
| XGBoost (Ensemble) | Very close second to Random Forest; excellent discrimination and balanced metrics, with strong performance on complex non-linear relationships. |

