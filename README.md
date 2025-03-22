# Credit Card Fraud Detection

## Overview
This project aims to develop a classification model to detect fraudulent credit card transactions efficiently. The dataset used for this project is from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is highly imbalanced, with the majority of transactions being non-fraudulent.

The primary goal is to build a model that can accurately detect fraudulent transactions while minimizing false positives.

## Dataset
The dataset contains credit card transactions with the following key features:
- **Time**: The time of the transaction
- **Amount**: The transaction amount
- **V1 to V28**: Principal components obtained using PCA for anonymization
- **Class**: The target variable (0: Non-fraud, 1: Fraud)

## Preprocessing Steps
1. **Data Loading**:
   - Read the dataset using Pandas.
   - Check for missing values and class distribution.

2. **Handling Class Imbalance**:
   - The dataset is highly imbalanced (0: 284,315, 1: 492).
   - Applied **random undersampling** to balance the dataset (0: 984, 1: 492).

3. **Feature Engineering**:
   - Created new features:
     - **Transaction_Frequency**: Count of transactions within a given timeframe.
     - **Normalized_Amount**: Standardized transaction amount.
   - Dropped the original 'Amount' column.

4. **Train-Test Split**:
   - Split the dataset into training (80%) and testing (20%).

## Model Selection
The model used for this project is **Random Forest Classifier** due to its ability to handle imbalanced data efficiently and provide robust performance.

## Training the Model
- Trained a **RandomForestClassifier** with 50 estimators.
- Evaluated the model on the test set.

## Model Evaluation
The model was evaluated using the following metrics:
- **Precision**: 97% for non-fraud, 94% for fraud.
- **Recall**: 98% for non-fraud, 93% for fraud.
- **F1-score**: 97% for non-fraud, 94% for fraud.
- **Accuracy**: 96%
- **AUC-ROC Score**: 0.9880

## Results
```
Random Forest Results:
               precision    recall  f1-score   support

           0       0.97      0.98      0.97       205
           1       0.94      0.93      0.94        91

    accuracy                           0.96       296
   macro avg       0.96      0.95      0.96       296
weighted avg       0.96      0.96      0.96       296

RandomForest AUC: 0.9880
```

## Model Saving
- The best-performing model was saved as `fraud_detection_model.pkl` using Joblib.

## Repository Structure
```
credit-card-fraud-detection/
├── fraud_detection.py    # Main script for data preprocessing, training, and evaluation
├── fraud_detection_model.pkl   # Saved best model
├── README.md            # Documentation
├── creditcard.csv       # Dataset (not included in repo, download from Kaggle)
└── requirements.txt     # Dependencies
```

## How to Run the Code
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the fraud detection script:
   ```bash
   python fraud_detection.py
   ```

## Future Improvements
- Implement **feature selection** to reduce dimensionality.
- Use **ensemble learning** to improve fraud detection.
- Deploy the model as a **web API** for real-time fraud detection.

## Conclusion
This project successfully built a fraud detection model with high accuracy and AUC scores, efficiently handling class imbalance. The **Random Forest model** performed well in detecting fraudulent transactions with minimal false positives.

