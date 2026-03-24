# 🏠 House Price Prediction ML Project

## 📌 Overview
This project builds a complete machine learning pipeline to predict house prices using the Ames Housing dataset.  
It demonstrates end-to-end ML workflow including preprocessing, feature engineering, model training, tuning, and evaluation.

---

## 🎯 Objective
Predict the **SalePrice** of houses based on various numerical and categorical features.

---

## 🧠 Key Concepts Applied

- Data preprocessing (missing values, encoding)
- Feature engineering (polynomial features)
- Feature scaling
- Regularization (Ridge, Lasso, ElasticNet)
- Cross-validation
- Hyperparameter tuning (GridSearchCV)
- Model comparison
- Residual analysis

---

## 📊 Dataset

- Source: Kaggle (House Prices Dataset)
- Features: 80+ variables (numerical + categorical)
- Target: `SalePrice`

---

## ⚙️ Project Structure
# 🏠 House Price Prediction ML Project

## 📌 Overview
This project builds a complete machine learning pipeline to predict house prices using the Ames Housing dataset.  
It demonstrates end-to-end ML workflow including preprocessing, feature engineering, model training, tuning, and evaluation.

---

## 🎯 Objective
Predict the **SalePrice** of houses based on various numerical and categorical features.

---

## 🧠 Key Concepts Applied

- Data preprocessing (missing values, encoding)
- Feature engineering (polynomial features)
- Feature scaling
- Regularization (Ridge, Lasso, ElasticNet)
- Cross-validation
- Hyperparameter tuning (GridSearchCV)
- Model comparison
- Residual analysis

---

## 📊 Dataset

- Source: Kaggle (House Prices Dataset)
- Features: 80+ variables (numerical + categorical)
- Target: `SalePrice`

---

## ⚙️ Project Structure
house-price-ml/
│
├── data/
│ └── raw/
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ ├── 04_model_comparison.ipynb
│ └── 05_final_evaluation.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── pipeline.py
│ ├── train.py
│ └── evaluate.py
│
├── models/
├── requirements.txt
└── README.md


---

## 🔄 Workflow

1. Data Understanding (EDA)
2. Data Preprocessing
3. Model Training + Tuning
4. Model Comparison
5. Final Evaluation

---

## 🤖 Models Used

- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest
- Support Vector Machine (SVM)

---

## 📈 Results

- Best RMSE: ~0.14
- Best Model: SVM / Ridge (depending on tuning)
- Model shows strong generalization with minimal bias

---

## 📊 Evaluation

- RMSE used as primary metric
- Residual analysis confirms:
  - Errors centered around zero
  - No major bias
  - Good predictive performance

---

## 🔍 Key Insights

- Regularization significantly improved performance
- Polynomial features were unnecessary
- Linear models performed surprisingly well
- Over-regularization (Lasso, ElasticNet) reduced performance

---

## 🚀 Future Improvements

- Try XGBoost / LightGBM
- Advanced feature engineering
- Feature selection
- Ensemble models

---

## ▶️ How to Run

```bash
pip install -r requirements.txt

## 🏆 Kaggle Submission

Final model was used to generate predictions for Kaggle competition.

Submission file: `submissions/submission.csv`