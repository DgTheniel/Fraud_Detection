import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gdown  # Required to download from Google Drive

# Google Drive link for the creditcard.csv file
file_url = "https://drive.google.com/uc?id=1gWCrzD_mjHH_KQun0QsMITqWtQYvWD8i"  # Replace with your file ID

# Download the CSV file from Google Drive
gdown.download(file_url, "creditcard.csv", quiet=False)

# Read the dataset
data = pd.read_csv("creditcard.csv")

# Check the column names to ensure 'Class' exists
st.write("Column names in the dataset:", data.columns)  # Display column names

# Strip any leading/trailing spaces in the column names
data.columns = data.columns.str.strip()

# Check the first few rows to ensure the data is loaded correctly
st.write("First few rows of the dataset:", data.head())

# Ensure the 'Class' column is present
if 'Class' in data.columns:
    X = data.drop(columns=['Class'])
    y = data['Class']
else:
    st.error("Column 'Class' not found in the dataset. Please check the dataset.")
    st.stop()  # Stop the execution if 'Class' column is missing

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train the model
try:
    model = joblib.load("fraud_detection_model.pkl")
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.warning("Model not found, training a new one...")
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, "fraud_detection_model.pkl")
    st.sidebar.success("Model trained and saved!")

# Streamlit app
st.title("Enhanced Fraud Detection App")
st.write("This app simulates real-time fraud detection and provides insights into the model's performance.")

# Sidebar slider for dynamic transaction processing
st.sidebar.subheader("Simulation Settings")
num_transactions = st.sidebar.slider("Number of transactions to process", min_value=1, max_value=len(X_test), value=10, step=1)

# Real-time transaction simulation without time.sleep for better performance
st.subheader("Transaction Simulation")
predictions = []  # to store predictions for batch display
for i in range(num_transactions):
    transaction = X_test.iloc[[i]]
    prediction = model.predict(transaction)
    label = "Fraudulent" if prediction == 1 else "Legitimate"
    predictions.append(f"Transaction {i+1}: {label}")

# Display results all at once for faster update
for prediction in predictions:
    st.write(prediction)

# Performance Metrics
st.subheader("Model Performance")
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))

# Fraud vs. Legitimate Transaction Distribution
fraud_count = sum(y_pred)
legit_count = len(y_pred) - fraud_count
st.write(f"**Legitimate Transactions**: {legit_count}")
st.write(f"**Fraudulent Transactions**: {fraud_count}")

# Visualizing Fraud vs. Legitimate Distribution
st.subheader("Fraud vs. Legitimate Transactions")
fig, ax = plt.subplots()
ax.bar(["Legitimate", "Fraudulent"], [legit_count, fraud_count], color=["green", "red"])
st.pyplot(fig)

# Feature Importance Visualization
st.subheader("Feature Importance")
importances = model.feature_importances_
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(X.columns, importances, color="blue")
ax.set_xlabel("Importance")
ax.set_title("Feature Importance in Fraud Detection")
st.pyplot(fig)

# Custom Transaction Prediction
st.subheader("Test a Custom Transaction")
custom_transaction = []
for column in X.columns:
    value = st.number_input(f"Enter value for {column}", min_value=0.0, max_value=1000000.0)
    custom_transaction.append(value)

if st.button("Predict Custom Transaction"):
    custom_prediction = model.predict([custom_transaction])
    custom_label = "Fraudulent" if custom_prediction == 1 else "Legitimate"
    st.write(f"The custom transaction is: **{custom_label}**")

# Export Predictions to CSV
if st.button("Export Predictions"):
    results = pd.DataFrame({
        'Transaction': range(1, len(y_pred)+1),
        'Prediction': ["Fraudulent" if pred == 1 else "Legitimate" for pred in y_pred]
    })
    results.to_csv("predictions.csv", index=False)
    st.success("Predictions exported to 'predictions.csv'!")
