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
    # Manually encode categorical variables as one-hot
    input_data = pd.get_dummies(input_data)
    # Ensure the columns match the training data
    expected_columns = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
        "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany",
        "Geography_Spain", "Gender_Female", "Gender_Male"
    ]
    for col in expected_columns:
        if col not in input_data:
            input_data[col] = 0  # Add missing columns with default value 0
    input_data = input_data[expected_columns]  # Reorder columns to match training
    try:
        input_scaled = scaler.transform(input_data)
    except ValueError as e:
        st.error(f"Scaler input error: {e}")
        st.stop()
    return input_scaled

# Load the model and scaler
model, scaler = load_model_and_scaler()

# Streamlit app
st.title("Customer Churn Prediction")

st.sidebar.header("Enter Customer Details")

def user_input_features():
    CreditScore = st.sidebar.number_input("Credit Score (300-850)", 300, 850, 600, step=1)
    Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    Age = st.sidebar.number_input("Age (18-100)", 18, 100, 35, step=1)
    Tenure = st.sidebar.number_input("Tenure (0-10)", 0, 10, 5, step=1)
    Balance = st.sidebar.number_input("Balance", 0.00, 250000.00, 50000.00, step=1000.00)
    NumOfProducts = st.sidebar.number_input("Number of Products (1-4)", 1, 4, 2, step=1)
    HasCrCard = st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"])
    IsActiveMember = st.sidebar.selectbox("Is Active Member?", ["No", "Yes"])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", 0.00, 200000.00, 50000.00, step=1000.00)

    # Create a DataFrame for user inputs
    data = {
        "CreditScore": [CreditScore],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [1 if HasCrCard == "Yes" else 0],
        "IsActiveMember": [1 if IsActiveMember == "Yes" else 0],
        "EstimatedSalary": [EstimatedSalary],
        "Geography_France": [1 if Geography == "France" else 0],
        "Geography_Germany": [1 if Geography == "Germany" else 0],
        "Geography_Spain": [1 if Geography == "Spain" else 0],
        "Gender_Female": [1 if Gender == "Female" else 0],
        "Gender_Male": [1 if Gender == "Male" else 0],
    }
    return pd.DataFrame(data)

# Collect user input
data_input = user_input_features()

if st.button("Predict"):
    try:
        # Debug: Log input columns
        st.write("Input Data Columns:")
        st.write(data_input.columns.tolist())

        # Preprocess the input
        preprocessed_input = preprocess_input(data_input, scaler)

        # Predict using the model
        prediction = model.predict(preprocessed_input)
        probability = model.predict_proba(preprocessed_input)

        # Display results
        st.subheader("Prediction")
        if prediction[0] == 1:
            st.error("The customer is likely to exit.")
        else:
            st.success("The customer is likely to stay.")

        st.subheader("Prediction Probability")
        st.write(f"Probability of exiting: {probability[0][1] * 100:.2f}%")
        st.write(f"Probability of staying: {probability[0][0] * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
