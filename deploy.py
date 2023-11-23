# Import necessary libraries
import streamlit as st
import pandas as pd
from keras.models import load_model
import pickle

# function to convert contract from categorical to numeric

def preprocess_contract(data):
    
    data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# function to convert payment options from categorical to numeric

def preprocess_payment(data):

    data['PaymentMethod'] = data['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})

# function to scale the numerical features

def scale_numeric_features(data, numerical_features, path):

    scaler = load_scaler(path)
    data[numerical_features] = scaler.transform(data[numerical_features])
    return data

def load_scaler(path):

    scaler = pickle.load(open(path, 'rb'))

    return scaler

# Function to preprocess input data
def preprocess_input(data):

    preprocess_contract(data)
    preprocess_payment(data)
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    path = 'scaler.pkl'
    data = scale_numeric_features(data, numerical_features, path)
    features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'PaymentMethod']

    return data[features]


# Function to load the trained model

def load():
    
    model = load_model('Delphine_Churning_Customers_Model.h5')
    return model


# Function to make predictions
def predict_churn(model, input_data):
    
    prediction = model.predict(input_data)
    if prediction[0][0] >= 0.5:
        result = f"The customer is predicted to churn. Confidence: {round(prediction[0][0] * 100, 2)} %."
    else:
        result = f"The customer is predicted not to churn. Confidence: {round(prediction[0][0] * 100 * 2, 2)} %."

    return result

# Streamlit app

def main():
    
    st.title("Delphine Churn Prediction App")
    st.header("User Input Features")

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=0.0)
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    predict = st.button("Predict")

    # DataFrame with the input features

    input_data = pd.DataFrame({
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'TotalCharges': [total_charges],
        'tenure': [tenure]
    })

    if predict:

        model = load()
        input_data = preprocess_input(input_data)
        result = predict_churn(model, input_data)
        st.write(result)

if __name__ == "__main__":
    main()
