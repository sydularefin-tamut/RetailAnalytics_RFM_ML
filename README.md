# 🛍️ Retail Customer Analytics Using RFM Segmentation & Machine Learning

📊 Predicting customer lifetime value (CLTV) using RFM (Recency, Frequency, Monetary) segmentation and machine learning on a large UK retail dataset.

---

## 📚 Project Summary

This project analyzes the behavior of 495,000+ UK retail customers using:
- 📦 RFM segmentation
- 📆 Temporal spending analysis
- 🤖 Machine learning models for CLTV classification

We apply preprocessing techniques like Box-Cox transformation and use various models (RandomForest, AdaBoost, ExtraTrees, LGBM, XGBoost) to identify high-value customer segments.

---

## 🔧 Key Features

- RFM feature generation & segmentation
- Box-Cox transformation for normality
- 6-month revenue trend analysis
- Model performance evaluation
- Customer cluster prediction

---

## 📊 Dataset

- 📌 Source: UK-based eCommerce dataset
- 🧾 Records: 495,478 transactions
- 🧍 Customers: Retail clients from UK
- ⚠️ Nulls: ~26% data cleaned and preprocessed
- 👇 Data columns include:
  - InvoiceNo, StockCode, Quantity
  - InvoiceDate, UnitPrice, CustomerID, Country

---

## 🧠 Models Used

| Model         | Accuracy | Precision | Recall | F1 Score | AUC     |
|---------------|----------|-----------|--------|----------|---------|
| Logistic Reg. | 85.82%   | 80.61%    | 85.82% | 81.23%   | 84.15%  |
| Random Forest | 91.89%   | 91.38%    | 91.89% | 91.48%   | 95.22%  |
| AdaBoost      | 89.87%   | 87.81%    | 89.87% | 88.82%   | 96.16%  |
| Extra Trees   | 92.15%   | 91.49%    | 92.15% | 91.54%   | 97.39%  |
| LGBM          | 91.64%   | 91.35%    | 91.64% | 91.44%   | 97.53%  |
| XGBoost       | **92.40%** | **92.27%** | **92.40%** | **92.28%** | **97.39%** |

> ✅ XGBoost outperforms all models, closely followed by Extra Trees and LGBM.

---

## 📁 Repository Layout

- `data/`: Raw and preprocessed retail data
- `notebooks/`: Jupyter notebooks for segmentation, feature engineering, modeling, and evaluation
- `src/`: Source scripts for modular development
- `models/`: Trained and saved model files
- `reports/`: Performance metrics and visual insights

---

## 🧪 Methodology Overview

- 🧹 Data Cleaning: Removed NULLs and negative revenue
- 📊 Feature Engineering: RFM, temporal revenue, binary features
- ⏳ Temporal Analysis: Customer revenue over 6 months
- 🧪 Modeling: Supervised classification on CLTV clusters
- ✅ Evaluation: Accuracy, F1-Score, Precision, Recall, AUC

---

## 📦 Requirements

```bash
pip install -r requirements.txt
Includes:
•	pandas
•	numpy
•	scikit-learn
•	xgboost
•	lightgbm
•	seaborn
•	matplotlib
•	scipy
yaml
