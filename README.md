# Credit Card Customer Churn Prediction

A machine learning project that predicts which credit card customers are likely to leave (churn) a bank. Several classifiers are compared with cross-validation, and the best performer, a **CatBoost** classifier, is trained and evaluated with a focus on how well it catches churners.

Because only a small share of customers churn, the project goes beyond accuracy and evaluates the model with a **precision-recall curve** and **average precision** for the churn class.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Possible Improvements](#possible-improvements)
- [Acknowledgements](#acknowledgements)

## Overview

| | |
|---|---|
| **Task** | Binary classification: will a customer churn (`Attrited Customer`) or stay (`Existing Customer`)? |
| **Challenge** | Imbalanced classes: only about 16% of customers churn |
| **Models compared** | XGBoost, Decision Tree, Random Forest, CatBoost |
| **Final model** | `CatBoostClassifier` |
| **Evaluation** | Cross-validation, accuracy, precision-recall curve, average precision |
| **Notebook** | `credit_card_churn.ipynb` |

Losing customers is expensive for a bank, and it is much cheaper to keep an existing customer than to win a new one. A model that flags customers at risk of churning lets the bank step in early with offers or outreach.

## Dataset

**Source:** [Credit Card Customers on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (file: `BankChurners.csv`)

- **Customers:** 10,127
- **Churn rate:** about 16% (roughly 1,627 churned vs. 8,500 existing customers)
- **Target:** `Attrition_Flag` (`Existing Customer` or `Attrited Customer`)

### Features

| Group | Columns |
|---|---|
| **Identifier** | `CLIENTNUM` (a customer ID, not a predictive feature) |
| **Demographics** | `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category` |
| **Account details** | `Card_Category`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon` |
| **Credit and spending behavior** | `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio` |

> The Kaggle version of the file also ends with two pre-computed `Naive_Bayes_Classifier_...` columns. They are not real features and should be excluded from modeling.

To run the notebook, download `BankChurners.csv` from the Kaggle link above and place it in the same folder as the notebook. Please check the dataset's page on Kaggle for its license and attribution terms.

## Approach

1. **Load the data** from `BankChurners.csv` with pandas and work on a copy of the original dataframe.
2. **Explore the data** with matplotlib and seaborn (class balance, feature distributions, and relationships with churn).
3. **Preprocess the features** with a scikit-learn `ColumnTransformer` inside a `Pipeline`:
   - missing values handled with `SimpleImputer`
   - numeric features scaled with `StandardScaler`
   - categorical features encoded with `OneHotEncoder` and `OrdinalEncoder`
4. **Split the data** into training and testing sets with `train_test_split`.
5. **Compare four classifiers** (XGBoost, Decision Tree, Random Forest, CatBoost) using `cross_val_score` and visualize the cross-validation scores in a box plot.
6. **Train the final model**, a CatBoost classifier, and check its training and testing scores.
7. **Evaluate the churn class** with a precision-recall curve and average precision, compared against a random-guessing baseline.

```python
cat = CatBoostClassifier(random_state=43, verbose=0)
cat.fit(x_train_sourced, y_train)

print('Training Score', cat.score(x_train_sourced, y_train))
print('Testing Score', cat.score(x_test_sourced, y_test))
```

## Results

### Final model (CatBoost)

| Metric | Training | Testing |
|---|---:|---:|
| Accuracy | 0.9958 | 0.9758 |

The model scores about **97.6% accuracy** on unseen data. The training score is a little higher than the test score, which is normal for a tree-based model, and the test score is still very strong.

> Accuracy alone can be misleading here: a model that predicted "existing customer" for everyone would already be right about 84% of the time. That is why the churn class is evaluated separately below.

### Model comparison

Cross-validation scores for the four classifiers:

![Cross-validation comparison of models](images/model_comparison.png)

| Model | Mean CV score |
|---|---|
| XGBoost | _add value_ |
| Decision Tree | _add value_ |
| Random Forest | _add value_ |
| CatBoost | _add value_ |

The single Decision Tree scored lowest of the four, as expected, since the ensemble models combine many trees.

### Precision-recall curve (churn class)

The churn class is treated as the positive class, and the dashed line shows the baseline of random guessing, which equals the churn rate (0.16).

![Precision-Recall Curve for the churn class](images/precision_recall_curve.png)

| Metric | Value |
|---|---|
| Average precision (churn class) | _add value_ |
| Random-guessing baseline | 0.16 |

How to read the curve:

- **Recall** is the share of churners the model catches. **Precision** is the share of flagged customers who really would have churned.
- Precision stays close to 1.0 across most of the recall range, so the model can catch most churners while almost every customer it flags is a genuine churn risk.
- Precision only drops sharply when the model is pushed to catch nearly all churners (recall close to 1.0).
- The curve sits far above the 0.16 baseline, so the model is much better than random guessing.

## Project Structure

```
credit-card-churn/
├── README.md
├── requirements.txt
├── .gitignore
├── credit_card_churn.ipynb    # main notebook
├── BankChurners.csv           # dataset (download from Kaggle)
└── images/
    ├── model_comparison.png
    └── precision_recall_curve.png
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
   cd YOUR-REPO
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download `BankChurners.csv` from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) and place it in the project folder, next to the notebook. It is loaded with a relative path:

   ```python
   df1 = pd.read_csv('BankChurners.csv')
   ```

4. Launch Jupyter and run all cells from top to bottom:

   ```bash
   jupyter notebook credit_card_churn.ipynb
   ```

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- catboost
- xgboost
- jupyter

Install everything with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost jupyter
```

## Possible Improvements

- Use **stratified** train/test splits and stratified cross-validation so the roughly 16% churn rate is preserved in every fold.
- Choose a **decision threshold** based on business cost: for example, accept lower precision to catch more churners if a retention offer is cheap.
- Report precision, recall, and F1 for the churn class at the chosen threshold, along with a confusion matrix.
- Tune the hyperparameters of CatBoost and XGBoost, or handle the imbalance with class weights.
- Add feature importance or SHAP plots to explain which behaviors (such as transaction counts or inactivity) signal churn.
- Check probability calibration if the predicted probabilities will be used to rank customers by risk.
- Try feature engineering, such as ratios between spending and credit limit.

## Acknowledgements

- Dataset: [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) on Kaggle by Sakshi Goyal.
- Models from [scikit-learn](https://scikit-learn.org/), [CatBoost](https://catboost.ai/), and [XGBoost](https://xgboost.ai/).
