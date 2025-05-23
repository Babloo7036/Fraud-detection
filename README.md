# Fraud Detection Project

## Overview

This project implements a machine learning-based fraud detection system for credit card transactions. It includes:

- A Jupyter Notebook (`Fraud_detection.ipynb`) for data exploration, preprocessing, and model training.
- A Streamlit web application (`app.py`) for real-time fraud prediction.

Using a **logistic regression model**, the system classifies transactions as **fraudulent** or **legitimate**, effectively handling a highly imbalanced dataset. The model is designed for high recall to catch as many fraud cases as possible.

---

## Key Features

### 🔍 Data Exploration and Preprocessing
- Analyzes a dataset of **6.36 million** transactions.
- No missing values in the dataset.
- Normalizes numerical features using `StandardScaler`.
- Encodes categorical feature (`type`) using `OneHotEncoder` with `drop='first'`.

### 🧠 Model Training
- Uses a `Pipeline` with `ColumnTransformer` and `LogisticRegression` (`class_weight='balanced'`).
- Achieves **94% recall** on fraud detection with **low precision (2%)**.

### 📊 Model Evaluation
- Classification report and confusion matrix included.
- Accuracy: **94.53%** (skewed due to class imbalance).
- Confusion Matrix:
      [[1802143, 104179],  **True Negatives, False Positives**
       [    156,    2308]]  **False Negatives, True Positives**



### Streamlit Web Application
- UI for entering transaction details and predicting fraud in real-time.
- Displays "Fraudulent" or "Legitimate" with visual feedback.
- Uses trained model (`fraud_detection_pipeline.pkl`).

### Scalability and Reproducibility
- Saves model with `joblib`.
- Modular code for easy updates or model swaps.

---

## Dataset

File: `Credit_card_data.csv`  
Records: 6,362,620 transactions

| Column            | Description                                        |
|-------------------|----------------------------------------------------|
| step              | Time step (int)                                    |
| type              | Transaction type (e.g., PAYMENT, TRANSFER)         |
| amount            | Transaction amount                                 |
| nameOrig          | Sender ID                                          |
| oldbalanceOrg     | Sender's balance before transaction                |
| newbalanceOrig    | Sender's balance after transaction                 |
| nameDest          | Receiver ID                                        |
| oldbalanceDest    | Receiver's balance before transaction              |
| newbalanceDest    | Receiver's balance after transaction               |
| isFraud           | Fraud label (0 = legitimate, 1 = fraudulent)       |
| isFlaggedFraud    | High-value fraud flag (0 or 1)                     |

- Fraudulent: 8,213 (0.13%)
- Flagged: 16
- No missing values

---

## Model Performance

| Metric      | Value (Fraud Class) |
|-------------|---------------------|
| Precision   | 0.02                |
| Recall      | 0.94                |
| F1-Score    | 0.04                |
| Accuracy    | 94.53%              |

High recall is prioritized to catch fraud, despite low precision.

---

## Setup Instructions

### Clone the Repository

```bash
git clone <https://github.com/Babloo7036/Fraud-detection>
cd fraud-detection-project
