import streamlit as st
import pandas as pd
import pickle

# Load the saved model and scaler
def load_model_and_scaler():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

# Function to preprocess user input
def preprocess_input(input_data, scaler):
    input_scaled = scaler.transform(input_data)
    return input_scaled

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Streamlit app
st.title("Customer Churn Prediction")

st.sidebar.header("Enter Customer Details")

def user_input_features():
    CreditScore = st.sidebar.number_input("Credit Score (300-850)", 300, 850, 600, step=1)
    Geography = st.sidebar.number_input("Geography (0: France, 1: Germany, 2: Spain)", 0, 2, 0, step=1)
    Gender = st.sidebar.number_input("Gender (0: Female, 1: Male)", 0, 1, 1, step=1)
    Age = st.sidebar.number_input("Age (18-100)", 18, 100, 35, step=1)
    Tenure = st.sidebar.number_input("Tenure (0-10)", 0, 10, 5, step=1)
    Balance = st.sidebar.number_input("Balance", 0.00, 250000.00, 50000.00, step=1000.00)
    NumOfProducts = st.sidebar.number_input("Number of Products (1-4)", 1, 4, 2, step=1)
    HasCrCard = st.sidebar.number_input("Has Credit Card? (0: No, 1: Yes)", 0, 1, 1, step=1)
    IsActiveMember = st.sidebar.number_input("Is Active Member? (0: No, 1: Yes)", 0, 1, 1, step=1)
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", 0.00, 200000.00, 50000.00, step=1000.00)

    data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }
    return pd.DataFrame(data, index=[0])

# Collect user input
data_input = user_input_features()

if st.button("Predict"):
    # Preprocess the input
    preprocessed_input = preprocess_input(data_input, scaler)

    # Predict using the model
    prediction = model.predict(preprocessed_input)
    probability = model.predict_proba(preprocessed_input)

    # Display results
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The customer is likely to exit.")
    else:
        st.write("The customer is likely to stay.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of exiting: {probability[0][1] * 100:.2f}%")
    st.write(f"Probability of staying: {probability[0][0] * 100:.2f}%")
