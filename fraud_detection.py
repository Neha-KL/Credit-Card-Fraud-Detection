import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Load dataset
df = pd.read_csv("D:/creditcard.csv")

# Check class distribution
print("Original Class Distribution:\n", df['Class'].value_counts())

# Define features and target variable
X = df.drop(columns=['Class'])
y = df['Class']

# Apply Random UnderSampling to balance classes
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Check new class distribution
print("Class Distribution After Resampling:\n", y_resampled.value_counts())

# Feature Engineering
X_resampled['Transaction_Frequency'] = X_resampled.groupby('Time')['Amount'].transform('count')
X_resampled['Normalized_Amount'] = (X_resampled['Amount'] - X_resampled['Amount'].mean()) / X_resampled['Amount'].std()
X_resampled.drop(columns=['Amount'], inplace=True)  # Remove original 'Amount' column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train RandomForest Model
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# Model Predictions
rf_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# Model Evaluation
print("Random Forest Results:\n", classification_report(y_test, rf_pred))
rf_auc = roc_auc_score(y_test, rf_probs)
print(f"RandomForest AUC: {rf_auc:.4f}")

# Save the best model
joblib.dump(rf, "fraud_detection_model.pkl")
print("Best model saved as fraud_detection_model.pkl")
