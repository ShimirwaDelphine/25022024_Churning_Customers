import streamlit as st
import pandas as pd
from keras.model import load_model
import pickle

def load_the_model():

    classification_model = load_model('Delphine_Churning_Customers_Model.h5')
    
    return classification_model

def prediction_on_churning(classification_model, input_data):

    prediction = classification_model.predict(input_data)

    if prediction[0][0] >= 0.5:

        result = "will churn."
    
    else:

        result = "will not churn."

    return result

def preprocess_contract(data):

    if data['Contract'] == 'Month-to-month':
        data['Contract'] = 0
    
    elif data['Contract'] == 'One year':
        data['Contract'] = 1
    
    else :
        data['Contract'] = 2

def preprocess_payment_method(data):

    if data['PaymentMethod'] == 'Electronic check' :
        data['PaymentMethod'] = 0
    
    elif data['PaymentMethod'] == 'Mailed check':
        data['PaymentMethod'] = 1
    
    elif data['PaymentMethod'] == 'Bank transfer (automatic)':
        data['PaymentMethod'] = 2
    
    else:
        data['PaymentMethod'] = 3

def preprocess_internet_service(data):

    if data['InternetService'] == 'DSL' :
        data['InternetService'] = 0
    
    elif data['InternetService'] == 'Fiber Optic':
        data['InternetService'] = 1
    
    else:
        data['InternetService'] = 2

def preprocess_gender(data):

    if data['gender'] == 'Female':
        data['gender'] = 0
    else:
        data['gender'] = 1

def preprocess_tech_support(data):

    if data['TechSupport'] == 'No':
        data['TechSupport'] = 0
    else:
        data['TechSupport'] = 1

def preprocess_partner(data):

    if data['Partner'] == 'No':
        data['Partner'] = 0
    else:
        data['Partner'] = 1

def preprocess_paperless_billing(data):

    if data['PaperlessBilling'] == 'False':
        data['PaperlessBilling'] = 0
    else:
        data['PaperlessBilling'] = 1

def preprocess_input_data(data):
    
    preprocess_contract(data)
    preprocess_payment_method(data)
    preprocess_internet_service(data)
    preprocess_gender(data)
    preprocess_tech_support(data)
    preprocess_paperless_billing(data)

    scale_data(data)

    features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'PaymentMethod', 'InternetService', 'PaperlessBilling', 'gender', 'TechSupport', 'Partner']
    data_input = data[features]

    return data_input

def scale_data(data):

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_features] = scaler.transform(data['numerical_features'])

    return data

def main():

    # Set the title of the app
    st.title("Customer Churn Prediction App")

    # Create input form for user to enter feature values
    st.header("User Input Features")

    # Create input fields for each feature
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=0.0)
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.checkbox("Paperless Billing")
    gender = st.selectbox("Gender", ["Male", "Female"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    predict = st.button("Predict")

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'InternetService': [internet_service],
        'PaperlessBilling': [paperless_billing],
        'gender': [gender],
        'TechSupport': [tech_support],
        'Partner': [partner],
        'TotalCharges': [total_charges],
        'tenure': [tenure]
    })

    if predict:

        input_data = preprocess_input_data()
        model = load_the_model()
        prediction = prediction_on_churning(model, input_data)

        st.subheader("Prediction")
        st.write(f"The custormer {prediction}")
