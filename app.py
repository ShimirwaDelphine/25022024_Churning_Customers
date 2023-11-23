from flask import Flask, render_template, request
import pandas as pd
from keras.models import load_model
import pickle

app = Flask(__name__)

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

    data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

def preprocess_payment_method(data):

    data['PaymentMethod'] = data['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})

def preprocess_internet_service(data):

    data['InternetService'] = data['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})

def preprocess_gender(data):

    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

def preprocess_tech_support(data):

   data['TechSupport'] = data['TechSupport'].map({'No': 0, 'Yes': 1, 'No internet service': 2})

def preprocess_partner(data):

    data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})

def preprocess_paperless_billing(data):

    data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})

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
    data[numerical_features] = scaler.transform(data[numerical_features])

    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        tenure = int(request.form['tenure'])
        contract = request.form['contract']
        payment_method = request.form['payment_method']
        internet_service = request.form['internet_service']
        paperless_billing = request.form['paperless_billing']
        gender = request.form['gender']
        tech_support = request.form['TechSupport']
        partner = request.form['partner']


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

        # Perform preprocessing
        input_data = preprocess_input_data(input_data)

        # Load the model
        model = load_the_model()

        # Make prediction
        prediction = prediction_on_churning(model, input_data)

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=3000)
