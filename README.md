#Fraud Detection Project

##Overview

This project implements a machine learning-based fraud detection system for credit card transactions. It consists of a Jupyter notebook (Fraud_detection.ipynb) for data exploration, preprocessing, and model training, and a Streamlit web application (app.py) for real-time fraud prediction. The system uses a logistic regression model to classify transactions as fraudulent or legitimate, addressing the challenge of identifying rare fraudulent activities in a highly imbalanced dataset.
The project leverages a dataset of credit card transactions, preprocesses it to handle numerical and categorical features, trains a model with balanced class weights to account for imbalance, and provides an interactive interface for users to input transaction details and receive predictions. The goal is to detect fraudulent transactions with high recall while providing a user-friendly tool for practical use.
Key Features

###Data Exploration and Preprocessing:

Analyzes a large dataset of 6.36 million transactions with no missing values.
Handles numerical features (amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) with StandardScaler for normalization.
Encodes the categorical type feature (e.g., PAYMENT, TRANSFER, CASH_OUT) using OneHotEncoder with drop='first' to avoid multicollinearity.


###Model Training:

Uses a Pipeline with ColumnTransformer for streamlined preprocessing and a LogisticRegression classifier with class_weight='balanced' to handle the imbalanced dataset (0.13% fraud).
Achieves 94% recall for fraud (detects most fraudulent transactions) but low precision (2%) due to many false positives.


###Model Evaluation:

Provides detailed metrics: classification report, confusion matrix, and accuracy (94.53% overall, though dominated by the majority class).
Confusion matrix: 1,802,143 true negatives, 104,179 false positives, 156 false negatives, 2,308 true positives.


###Streamlit Web Application:

Interactive interface for inputting transaction details (type, amount, sender/receiver balances).
Displays predictions as "Fraudulent" or "Legitimate" with clear visual feedback (st.error or st.success).
Loads the trained model (fraud_detection_pipeline.pkl) for real-time predictions.


###Scalability and Reproducibility:

Saves the trained model using joblib for easy deployment.
Modular code structure allows for easy updates (e.g., swapping logistic regression for a more precise model like Random Forest).



##Dataset

The dataset (Credit_card_data.csv) contains 6,362,620 transactions with the following columns:

step: Time step (integer).
type: Transaction type (categorical: PAYMENT, TRANSFER, CASH_OUT, etc.).
amount: Transaction amount (float).
nameOrig: Sender ID (string).
oldbalanceOrg: Sender's balance before transaction (float).
newbalanceOrig: Sender's balance after transaction (float).
nameDest: Receiver ID (string).
oldbalanceDest: Receiver's balance before transaction (float).
newbalanceDest: Receiver's balance after transaction (float).
isFraud: Fraud label (0 = legitimate, 1 = fraudulent).
isFlaggedFraud: Flag for high-value suspicious transactions (0 or 1).

##Key Statistics:

Fraudulent transactions: 8,213 (0.13% of total).
Flagged fraud: 16 (extremely rare).
No missing values.

##Model Performance

The logistic regression model, trained on a train-test split, shows:

**Accuracy: 94.53% (misleading due to class imbalance).**
**Fraud Class (1):**
**Precision: 0.02 (many false positives).**
**Recall: 0.94 (catches 94% of fraud cases).**
**F1-Score: 0.04 (low due to poor precision).**


Confusion Matrix:**[[1802143, 104179]**,  True Negatives, False Positives
                 **[    156,    2308]]**  False Negatives, True Positives



The high recall is critical for fraud detection, but the low precision indicates a need for improvement to reduce false positives.
Setup Instructions

Clone the Repository:git clone <https://github.com/Babloo7036/Fraud-detection>
cd fraud-detection-project


Install Dependencies:Ensure Python 3.8+ is installed. Install required packages:pip install -r requirements.txt

Sample requirements.txt:pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib


Download the Dataset:
Place Credit_card_data.csv in the project directory.
Alternatively, use a publicly available dataset (e.g., from Kaggle: Synthetic Financial Datasets for Fraud Detection).


Run the Notebook:
Open Fraud_detection.ipynb in Jupyter Notebook or JupyterLab.
Execute cells to explore data, train the model, and save fraud_detection_pipeline.pkl.


Run the Streamlit App:streamlit run app.py


Access the app at http://localhost:8501.



Usage

Notebook (Fraud_detection.ipynb):
Explore the dataset using cells for data loading, visualization, and summary statistics.
Train the model and evaluate performance using the classification report and confusion matrix.
Save the trained model for use in the app.


Streamlit App (app.py):
Open the app in a browser.
Select a transaction type and enter numerical values for amount, sender balances, and receiver balances.
Click "Predict" to see if the transaction is fraudulent or legitimate.
Example input:
Transaction Type: TRANSFER
Amount: 10000.0
Old Balance (Sender): 10000.0
New Balance (Sender): 0.0
Old Balance (Receiver): 0.0
New Balance (Receiver): 10000.0
Result: Likely flagged as fraudulent due to zero new balance.





Potential Improvements

Model Enhancements:
Replace logistic regression with Random Forest or XGBoost for better precision.
Use SMOTE for oversampling the fraud class or adjust the decision threshold to balance precision and recall.
Incorporate additional features (e.g., transaction velocity, balance change ratios).


App Enhancements:
Add input validation to ensure logical consistency (e.g., newbalanceOrig = oldbalanceOrg - amount for PAYMENT).
Display prediction probabilities for better transparency.
Include a batch prediction feature for processing CSV files.
Visualize feature contributions to explain predictions.
