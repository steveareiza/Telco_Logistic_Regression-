# Telco Customer Churn Prediction

## Project Overview
This project uses the Telco customer churn dataset to build an **end-to-end machine learning pipeline** that predicts the probability of customer churn. The goal is to provide actionable insights to reduce churn and optimize customer retention strategies.

---

## Data Source
This project uses the **IBM Telco Customer Churn** dataset publicly available on Kaggle:

IBM Telco Customer Churn Dataset  
https://www.kaggle.com/blastchar/telco-customer-churn

The dataset contains customer-level information including demographics, service subscriptions, tenure, monthly charges, total charges, and churn status.

To reproduce this analysis:

1. Download the dataset from Kaggle.
2. Place the `telco_churn.csv` file in the root directory of this project.
3. Run the notebook `TELCO_Churn_Logistic_Regression.ipynb`.

---

## Objective
- Predict which customers are likely to churn.  
- Provide **business recommendations** based on feature importance.  
- Adjust model outputs to align with **business risk tolerance** for proactive retention.

---

## Methodology

1. **Data Loading & Cleaning**
   - Removed irrelevant columns and missing values.  
   - Converted categorical variables to numerical representations where necessary.  

2. **Exploratory Data Analysis (EDA)**
   - Checked class imbalance and feature distributions.  
   - Identified key predictors correlated with churn.

3. **Baseline Model**
   - Built an initial logistic regression to confirm feature signal using ROC-AUC.  

4. **Regularized Model (F1-Optimized)**
   - Applied **Ridge (L2) regularization** to reduce overfitting and improve generalization.  
   - Performed **cross-validated hyperparameter tuning** to select the optimal regularization strength.  
   - Optimized F1 score to balance precision and recall.

5. **Threshold Optimization**
   - Used the **Precision–Recall curve** to select a probability threshold that prioritizes **recall**.  
   - This post-model adjustment aligns predictions with business risk tolerance without retraining the model.

6. **Model Interpretation**
   - Examined coefficients to identify drivers of churn:  
     - Fiber optic internet and streaming service tend to show higher churn risk  
     - Multi-service, yearly contracts, and longer tenure tend to show lower churn risk  

7. **Actionable Recommendations**
   - Audit fiber-optic service quality, internet speed, and streaming packages.  
   - Target high-risk customers with proactive retention campaigns.  
   - Optimize service bundles to reduce churn risk.

---

## Evaluation Metrics
- ROC-AUC: ~0.84 (baseline signal strong)  
- F1 score optimized for balanced precision and recall  
- Precision–Recall curve used to select threshold for recall prioritization  

---

## Technical Stack
- Python 3 (IPython kernel)  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`  
- Logistic Regression with L2 regularization  

---

## Usage
1. Clone the repository:  
```bash
git clone https://github.com/steveareiza/Telco_Churn_Logistic_Regression.git
