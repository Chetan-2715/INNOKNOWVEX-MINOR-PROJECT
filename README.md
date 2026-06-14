# Credit Card Fraud Detection (Imbalanced Data + Anomaly Detection)

This project implements a complete machine learning pipeline to detect fraudulent credit card transactions. It systematically compares different supervised learning models under various class imbalance handling techniques, alongside an unsupervised anomaly detection approach.

---

## 📌 Dataset Overview
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
* **Total Transactions:** 284,807
* **Fraudulent Transactions:** 492 (0.1727% of the total)
* **Features:** 30 numerical features (`Time`, `Amount`, and 28 principal components `V1` to `V28` obtained via PCA).
* **Target:** `Class` (1 for fraud, 0 for normal).

---

## 🛠️ Key Techniques & Models

### Imbalance Handling Techniques
1. **Random Undersampling:** Balances the classes by randomly removing transactions from the majority class (normal).
2. **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically generates new fraud transactions based on the k-nearest neighbors of existing fraud transactions to balance the dataset.

### Models Implemented
* **Logistic Regression** (Supervised)
* **Random Forest** (Supervised)
* **XGBoost Classifier** (Supervised)
* **Isolation Forest** (Unsupervised Anomaly Detection)

---

## 📂 Project Structure

The codebase is split into 6 modular, task-specific Jupyter Notebooks. In each notebook, **each cell performs exactly one logical task** (e.g. imports, scaling, resampling, training, predicting, evaluation).

* 📄 **[01_data_analysis.ipynb](01_data_analysis.ipynb)**: Quick Exploratory Data Analysis (EDA) of class imbalance, statistical distributions, and feature correlations.
* 📄 **[02_logistic_regression.ipynb](02_logistic_regression.ipynb)**: Trains and evaluates Logistic Regression using Undersampling vs. SMOTE.
* 📄 **[03_random_forest.ipynb](03_random_forest.ipynb)**: Trains and evaluates Random Forest using Undersampling vs. SMOTE (optimized using parallel execution and tree depth limits).
* 📄 **[04_xgboost.ipynb](04_xgboost.ipynb)**: Trains and evaluates XGBoost Classifier using Undersampling vs. SMOTE.
* 📄 **[05_anomaly_detection.ipynb](05_anomaly_detection.ipynb)**: Unsupervised Anomaly Detection using Isolation Forest trained purely on normal transaction patterns.
* 📄 **[06_model_comparison.ipynb](06_model_comparison.ipynb)**: Compiles results, plots comparative ROC and Precision-Recall curves, and visualizes confusion matrices and metrics side-by-side.

---

## 📊 Summary of Results

All models were evaluated on an **80/20 train-test split (56,962 test transactions, containing 98 frauds)**.

| Model | Sampling | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC | PR-AUC | True Positives (out of 98) | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Undersampling | 0.04 | **0.92** | 0.07 | 0.9761 | 0.7298 | 90 / 98 | 2,252 |
| **Logistic Regression** | SMOTE | 0.06 | **0.92** | 0.11 | 0.9712 | 0.7689 | 90 / 98 | 1,434 |
| **Random Forest** | Undersampling | 0.04 | **0.92** | 0.08 | 0.9777 | 0.7538 | 90 / 98 | 2,037 |
| **Random Forest** | SMOTE | 0.52 | 0.86 | 0.64 | 0.9777 | 0.8245 | 84 / 98 | 79 |
| **XGBoost** | Undersampling | 0.03 | **0.92** | 0.06 | 0.9749 | 0.3686 | 90 / 98 | 2,609 |
| **XGBoost** | SMOTE | **0.72** | 0.86 | **0.78** | **0.9779** | **0.8672** | 84 / 98 | **33** |
| **Isolation Forest** | Unsupervised | 0.22 | 0.29 | 0.25 | 0.9535 | 0.1234 | 28 / 98 | 101 |

### Key Takeaways
1. **SMOTE + XGBoost** is the best model configuration, capturing **86% of frauds** with **72% precision** (only 33 false positives).
2. **Undersampling** achieves a high Recall (**92%**) but comes with a massive cost of **2,000+ false alarms** because the model fails to learn the full profile of normal transactions.
3. **Isolation Forest** has lower supervised metrics but excels at detecting **previously unseen** fraud types since it requires no training labels.

---

## 🚀 Setup & Installation

### Prerequisites
Make sure you have Python installed. Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### Running the Project
1. Download the Credit Card Fraud Detection dataset (`creditcard.csv`) from Kaggle and place it in the root folder.
2. Open your preferred IDE (e.g. VS Code, Jupyter Notebook, JupyterLab).
3. Run the notebooks sequentially starting from `01_data_analysis.ipynb`.
