import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Function to load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

# Function to load the scaler
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        return pickle.load(file)

# Preprocessing function
def preprocess_input(input_data, scaler):
    feature_columns = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
    input_data = input_data[feature_columns]
    scaled_data = scaler.transform(input_data)
    return scaled_data

# Load the model and scaler
model = load_model()
scaler = load_scaler()

# Streamlit UI
st.title("Credit Card Fraud Detection")

# Input fields for user data
st.header("Enter Transaction Details")
user_input = {
    'amt': st.number_input('Transaction Amount', min_value=0.0),
    'city_pop': st.number_input('City Population', min_value=0),
    'lat': st.number_input('Cardholder Latitude'),
    'long': st.number_input('Cardholder Longitude'),
    'merch_lat': st.number_input('Merchant Latitude'),
    'merch_long': st.number_input('Merchant Longitude')
}

if st.button("Predict Fraud"):
    try:
        # Convert user input to a DataFrame
        input_df = pd.DataFrame([user_input])

        # Preprocess input data
        processed_data = preprocess_input(input_df, scaler)

        # Predict fraud
        prediction = model.predict(processed_data)[0]

        # Display the result
        if prediction == 1:
            st.error("This transaction is predicted to be FRAUDULENT!")
        else:
            st.success("This transaction is predicted to be LEGITIMATE.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
