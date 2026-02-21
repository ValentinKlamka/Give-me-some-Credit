# Credit Default Prediction – Give Me Some Credit

## Project Overview

This project builds a machine learning model to predict the **probability of serious delinquency within the next two years** using the Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.

The objective is to develop a robust credit risk model with strong discriminatory power (AUC / Gini), similar to approaches used in retail banking risk modeling.

---

## Dataset

The dataset contains **anonymized borrower information** including:

- **Revolving credit utilization**
- **Debt ratio**
- **Age**
- **Monthly income**
- **Number of past due events** (30–59, 60–89, 90+ days)
- **Number of open credit lines and loans**
- **Real estate loans**
- **Dependents**

### Target Variable

- **`SeriousDlqin2yrs`**
  - `1` = serious delinquency within 2 years
  - `0` = no delinquency

> **Note:** The dataset is **highly imbalanced**, with a relatively low default rate — typical for retail credit risk data.

---

## Data Preprocessing

The following preprocessing steps were applied:

| Step | Description |
|------|-------------|
| **Index removal** | Removed index column (`Unnamed: 0`) |
| **Age filtering** | Filtered out invalid ages (`age > 0`) |
| **Outlier capping** | Capped extreme values in `RevolvingUtilizationOfUnsecuredLines` (clipped at upper bound of 5) |
| **Missing values** | Handled natively by LightGBM |
| **Data splitting** | Stratified K-Fold cross-validation to preserve class distribution |

---

## Model

### Model Architecture

**Algorithm:** `LightGBM (LGBMClassifier)`

### Why LightGBM?

- Handles missing values natively  
- Robust to multicollinearity  
- Efficient for tabular data  
- Strong performance in credit risk problems  
- Widely used in industry  

### Training Strategy

```python
Cross-validation: StratifiedKFold
Early stopping: Based on validation AUC
```

The final model was trained using **LightGBM** with stratified cross-validation and early stopping.

---

## Results

### Final Performance

| Metric | Score |
|--------|-------|
| **Public Leaderboard AUC** | 0.86206 |
| **Corresponding Gini** | ~0.72 |

### Key Takeaways

- The model demonstrates **strong discriminatory power** in predicting serious delinquency.

- Cross-validation results were **consistent with leaderboard performance**, indicating stable generalization and limited overfitting.

- The model successfully captures key **credit risk patterns** in the dataset and achieves competitive performance for this benchmark problem.

---

## Future Improvements

- Feature engineering (e.g., interaction terms, binning)
- Hyperparameter tuning with Optuna/GridSearch
- Ensemble methods (stacking, blending)
- SMOTE or other resampling techniques for class imbalance
- Model interpretation (SHAP values, feature importance)

---

## License

This project is for educational purposes based on the Kaggle competition dataset.

---

## Author

**Valen**  
Contact: [Your Email/GitHub]

---

If you find this project helpful, please consider giving it a star!

