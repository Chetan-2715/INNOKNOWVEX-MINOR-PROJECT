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

## 📂 Project Structure & File Descriptions

The codebase is split into 6 modular, task-specific Jupyter Notebooks and helper configuration files. In each notebook, **each cell performs exactly one logical task** (e.g., imports, scaling, resampling, training, predicting, evaluation).

### 1. 📄 [01_data_analysis.ipynb](01_data_analysis.ipynb) (Exploratory Data Analysis)
* **Goal:** Inspect dataset health, check features, and analyze the severe target class imbalance.
* **Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`.
* **Key Tasks:**
  * Displays shape, data types, and checks for **missing values** (verifying 0 null values).
  * Calculates class imbalance: **0.1727% fraud** (492 cases) vs. **99.8273% normal** (284,315 cases).
  * Visualizes the skewed distributions of transaction `Time` and `Amount` (on log scales).
  * Analyzes feature correlations with the target `Class` to identify key patterns.

### 2. 📄 [02_logistic_regression.ipynb](02_logistic_regression.ipynb) (Logistic Regression)
* **Goal:** Build a linear model baseline under Undersampling and SMOTE.
* **Key Tasks:**
  * Scales skewed numerical features `Time` and `Amount` using `RobustScaler` (minimizes outlier impact).
  * Splits features into a stratified 80/20 train-test split.
  * Implements **Random Undersampling** on the training set (~788 balanced transactions), trains Logistic Regression, and logs test metrics.
  * Implements **SMOTE** on the training set (~454k oversampled transactions), trains Logistic Regression, and logs test metrics.
  * Measures F1, Precision, Recall, ROC-AUC, and PR-AUC.

### 3. 📄 [03_random_forest.ipynb](03_random_forest.ipynb) (Random Forest Classifier)
* **Goal:** Train a non-linear ensemble tree model under both imbalance resampling methods.
* **Key Tasks:**
  * Scales and splits data in the same manner as the baseline.
  * Trains Random Forest on **Undersampled** data utilizing multi-core processing (`n_jobs=-1`).
  * Trains Random Forest on **SMOTE** data, restricting `max_depth=12` to prevent overfitting on oversampled data and optimize training speeds.
  * Evaluates and logs confusion matrices and classification reports.

### 4. 📄 [04_xgboost.ipynb](04_xgboost.ipynb) (XGBoost Classifier)
* **Goal:** Implement a high-performance Gradient Boosted Decision Tree (GBDT) model.
* **Key Tasks:**
  * Trains XGBoost under **Undersampling** (balanced set).
  * Trains XGBoost under **SMOTE** oversampling using `eval_metric='logloss'` and multi-core acceleration.
  * Compares metrics to see the impact of SMOTE on reducing false positive alarms.

### 5. 📄 [05_anomaly_detection.ipynb](05_anomaly_detection.ipynb) (Isolation Forest Anomaly Detection)
* **Goal:** Perform unsupervised anomaly detection, simulating a real-world scenario with no historical fraud labels.
* **Key Tasks:**
  * Separates normal transactions (`Class == 0`) in the training set to learn normal behavior.
  * Trains the Isolation Forest model on normal transactions (`contamination` parameter set to the actual fraud ratio).
  * Evaluates anomaly scores on the full test set (mapping outlier predictions `-1` to fraud `1` and inlier `1` to normal `0`).
  * Compares unsupervised performance metrics (e.g., Precision, Recall, F1) to the supervised models.

### 6. 📄 [06_model_comparison.ipynb](06_model_comparison.ipynb) (Overall Comparison & Visualizations)
* **Goal:** Provide a centralized comparison suite with comprehensive visualizations.
* **Key Tasks:**
  * Trains all three supervised models (Logistic Regression, Random Forest, XGBoost) under both resampling frameworks.
  * Plots **ROC Curves** side-by-side to visualize true positive vs. false positive rates.
  * Plots **Precision-Recall (PR) Curves** side-by-side (the most informative plot for highly imbalanced labels).
  * Displays a **2x3 grid of Confusion Matrices** comparing all model outputs.
  * Compiles a metrics table and plots a **triple bar chart** comparing Recall, Precision, and F1-Scores.

### 7. Helper & Git Files
* 📄 **[.gitignore](.gitignore)**: Excludes the 150MB `creditcard.csv` dataset, Jupyter Notebook checkpoints (`.ipynb_checkpoints/`), log files, and Python bytecode caches from git tracking.
* 📄 **[README.md](README.md)**: Main documentation describing setup, metrics, project layout, and results.

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
