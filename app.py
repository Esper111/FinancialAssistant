import streamlit as st
import numpy as np
import pandas as pd
import joblib

fraud_model = joblib.load('fraud_model.pkl')
transaction_model = joblib.load('arima_model.pkl')

def predict_fraud(transaction_data):
    fraud_probability = fraud_model.predict_proba([transaction_data])[0][1]
    return fraud_probability > 0.5

def predict_future_transactions(current_data, months_ahead=6):
    predicted_transactions = current_data * np.random.uniform(0.9, 1.1, months_ahead)
    return predicted_transactions

st.title("Financial Assistant App")

st.sidebar.title("Select Functionality")
option = st.sidebar.selectbox("Choose an option", ("Predict Fraud", "Predict Future Transactions"))

if option == "Predict Fraud":
    st.subheader("Fraud Detection")
    customer_id = st.text_input("Customer ID")
    tx_amount = st.number_input("Transaction Amount (TX_AMOUNT)", min_value=0.0, step=100)
    tx_time_seconds = st.number_input("Transaction Time in Seconds (TX_TIME_SECONDS)", min_value=0.0, step=0.1)
    tx_time_days = st.number_input("Transaction Time in Days (TX_TIME_DAYS)", min_value=0.0, step=0.1)
    transaction_data = [customer_id, tx_amount, tx_time_seconds, tx_time_days]
    
    if st.button("Check for Fraud"):
        result = predict_fraud(transaction_data)
        if result:
            st.warning("This transaction is potentially fraudulent.")
        else:
            st.success("This transaction appears to be safe.")

elif option == "Predict Future Transactions":
    st.subheader("Future Transaction Prediction")
    current_balance = st.number_input("Current Balance", min_value=0.0, step=0.1)
    current_spend = st.number_input("Current Monthly Spend", min_value=0.0, step=0.1)
    months_ahead = st.slider("Months Ahead to Forecast", min_value=1, max_value=12, value=6)
    
    if st.button("Predict Future Transactions"):
        future_transactions = predict_future_transactions(current_spend, months_ahead)
        st.write(f"Predicted transactions for the next {months_ahead} months:")
        for i, transaction in enumerate(future_transactions, 1):
            st.write(f"Month {i}: {transaction:.2f}")
