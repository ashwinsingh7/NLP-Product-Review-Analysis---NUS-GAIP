import boto3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib

# AWS S3 configuration
s3_bucket_name = 'mlops-for-cv'
s3_model_path = 'models'

# Step 1: Load dataset from S3
data = pd.read_csv('s3://mlops-for-cv/trigger/card_transdata.csv')
X = data.drop('fraud', axis=1)
y = data['fraud']

# Step 2: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model performance
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = np.mean(y_pred == y_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Step 7: Save the trained model to S3
model_filename = "fraud_detection_model.pkl"
joblib.dump(model, model_filename)
s3_model_path = f"models/{model_filename}"

s3_client = boto3.client('s3')
s3_client.upload_file(model_filename, s3_bucket_name, s3_model_path)
print("Model saved to Amazon S3 successfully.")

# Step 8: Load the model from S3
s3_model_path = 'models/fraud_detection_model.pkl'
s3_obj = boto3.resource('s3').Object(s3_bucket_name, s3_model_path)
model_bytes = s3_obj.get()['Body'].read()

from io import BytesIO
model = joblib.load(BytesIO(model_bytes))

# Step 9: Make predictions on the test set using the loaded model
y_pred_test = model.predict(X_test)

# Step 10: Save predictions to CSV and upload to S3
predictions_csv_filename = 'predictions.csv'
y_pred_df = pd.DataFrame({'Predicted Labels': np.where(y_pred_test == 0, 'Not Fraud', 'Fraud'),})
y_pred_df.to_csv(predictions_csv_filename, index=False)

s3_predictions_path = f'test_output/{predictions_csv_filename}'
s3_client.upload_file(predictions_csv_filename, s3_bucket_name, s3_predictions_path)
print("Predictions saved to Amazon S3 successfully.")
