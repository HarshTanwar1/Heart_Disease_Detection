<div align="center">

# ❤️ Heart Disease Detection

An end-to-end machine learning pipeline that predicts heart disease from clinical data — achieving ~88.7% accuracy with ensemble learning

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Compute-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>
<br>

## 🎯 Overview

This project walks through the **complete supervised-learning workflow** — from raw clinical data to a deployable, serialized model. It cleans and preprocesses patient records, trains and tunes **five** classification algorithms, combines them with **ensemble learning**, and persists the best model for instant reuse.

> The final **Stacking Classifier** achieves **~88.7% accuracy** and a **~0.89 F1 score** on the test set, outperforming every individual model

<br>

## 📊 Key Highlights

| Model                      | Accuracy  | Precision |  Recall   | F1 Score  |
| -------------------------- | :-------: | :-------: | :-------: | :-------: |
| Naïve Bayes                |   0.843   |   0.860   |   0.846   |   0.852   |
| Logistic Regression        |   0.852   |   0.850   |   0.878   |   0.864   |
| Support Vector Machine     |   0.839   |   0.864   |   0.829   |   0.846   |
| k-Nearest Neighbors        |   0.848   |   0.844   |   0.878   |   0.861   |
| Decision Tree              |   0.783   |   0.807   |   0.780   |   0.793   |
| Voting Classifier          |   0.874   |   0.873   |   0.894   |   0.884   |
| 🏆 **Stacking Classifier** | **0.887** | **0.894** | **0.894** | **0.894** |

<br>

## 🛠️ Tech Stack

| Category          | Tools                                                          |
| :---------------- | :------------------------------------------------------------- |
| **Language**      | Python 3.10                                                    |
| **ML & Modeling** | scikit-learn (classifiers, ensembles, GridSearchCV, metrics) |
| **Data**          | pandas, NumPy                                                  |
| **Visualization** | Matplotlib, Seaborn                                            |
| **Environment**   | Jupyter Notebook                                               |
| **Persistence**   | pickle                                                         |

<br>

## ✨ Features & Methodology

- 🧹 **Data Cleaning** — handles missing values, duplicate records, structural errors, and outliers.
- ⚙️ **Data Preprocessing** — Min-Max normalization of numerical features and encoding of categorical features.
- 🎛️ **Feature Selection** — separating predictors from the target variable.
- 🤖 **Five Tuned Classifiers** — Naïve Bayes, Logistic Regression, SVM, kNN, and Decision Tree, each optimized with `GridSearchCV`.
- 🧩 **Ensemble Learning** — Voting and Stacking classifiers that combine base learners to boost performance.
- 📐 **Rigorous Evaluation** — Accuracy, Precision, Recall, F1 Score, and confusion matrices.
- 💾 **Model Persistence** — the trained Stacking Classifier and Min-Max scaler are pickled for inference without retraining.

<br>

## 🗂️ Dataset

Built on the [Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (`heart.csv`) — **918 patient records**, **11 clinical features**, and a binary target.

| Feature          | Description                                            |
| ---------------- | ------------------------------------------------------ |
| `Age`            | Age of the patient (years)                             |
| `Sex`            | Sex (M / F)                                            |
| `ChestPainType`  | Chest pain type (TA, ATA, NAP, ASY)                    |
| `RestingBP`      | Resting blood pressure (mm Hg)                         |
| `Cholesterol`    | Serum cholesterol (mg/dl)                              |
| `FastingBS`      | Fasting blood sugar (1 if > 120 mg/dl, else 0)         |
| `RestingECG`     | Resting electrocardiogram results (Normal, ST, LVH)    |
| `MaxHR`          | Maximum heart rate achieved                            |
| `ExerciseAngina` | Exercise-induced angina (Y / N)                        |
| `Oldpeak`        | ST depression induced by exercise                      |
| `ST_Slope`       | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease`   | **Target** — 1 = heart disease, 0 = normal             |

<br>

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HarshTanwar1/Heart_Disease_Detection.git
cd Heart_Disease_Detection
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 4. Launch the notebook

```bash
jupyter notebook Heart_Disease_Prediction.ipynb
```


### 5. Use the pre-trained model (no retraining needed)

```python
import pickle

# Load the serialized model and scaler
model  = pickle.load(open("stacking_classifier_model.pkl", "rb"))
scaler = pickle.load(open("min_max_scaler.pkl", "rb"))
```

<br>

## 🧠 What I Learned

- The **end-to-end workflow** of a supervised ML project, from raw CSV to a deployable, serialized model.
- Practical **data cleaning & preprocessing** — missing/duplicate data, structural fixes, outliers, normalization, and categorical encoding.
- Why you must persist the **same scaler** used at training time for consistent inference.
- The strengths and trade-offs of several classifiers (Naïve Bayes, Logistic Regression, SVM, kNN, Decision Trees).
- Using **`GridSearchCV`** to systematically tune hyperparameters instead of guessing.
- How **ensemble methods** (Voting & Stacking) combine base learners to outperform any single model.
- **Model serialization with `pickle`** and evaluation beyond accuracy (precision, recall, F1).

<br>

## 🔭 Future Improvements

- 📦 **Refactor into modules** such as `preprocess.py`, `train.py`, `predict.py`.
- 🌐 **Prediction interface** — a CLI or Streamlit / Flask app that serves the pickled model.
- ⚖️ **Class balance & probability calibration** handling and reporting.
- 🚀 **More models** (Random Forest, Gradient Boosting / XGBoost) with a wider search space.
<br>

---

<div align="center">

⭐ _If you found this project helpful or interesting, consider giving it a star!_ ⭐

</div>
