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

    data['PaperlessBilling'] = int(data['PaperlessBilling'])

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
        tech_support = request.form['tech_support']
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
    app.run(debug=True)
