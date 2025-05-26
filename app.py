from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
def get_details(disease):
    try:
        desc = " ".join(description_df[description_df["Disease"] == disease]["Description"].values)
        precaution = precautions_df[precautions_df["Disease"] == disease].iloc[0, 1:].tolist()
        symptoms = symptoms_df[symptoms_df["Disease"] == disease].iloc[0, 1:].tolist()
        medicine = medications_df[medications_df["Disease"] == disease]["Medication"].tolist()
        diet = diets_df[diets_df["Disease"] == disease]["Diet"].tolist()
        workout = workout_df[workout_df["disease"] == disease]["workout"].tolist()
        doctor = doctor_df[doctor_df["Disease"] == disease]
        return desc, precaution, symptoms, medicine, diet, workout, doctor
    except Exception as e:
        return "No data found", [], [], [], [], [], []
# Load models
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open("model2.pkl", "rb"))

# Load encoders and labels
label_encoder = pickle.load(open('disease_label_encoder.pkl', 'rb'))
disease_labels = {i: label for i, label in enumerate(label_encoder.classes_)}

# Load additional data for symptom-based prediction
df = pd.read_csv("helper_data/Training.csv")
symptoms_dict = {col: i for i, col in enumerate(df.columns) if col != "prognosis"}
disease_dict = {i: val for i, val in enumerate(df["prognosis"].unique())}

symptoms_df = pd.read_csv("helper_data/symtoms_df.csv")
precautions_df = pd.read_csv("helper_data/precautions_df.csv")
workout_df = pd.read_csv("helper_data/workout_df.csv")
description_df = pd.read_csv("helper_data/description.csv")
medications_df = pd.read_csv("helper_data/medications.csv")
diets_df = pd.read_csv("helper_data/diets.csv")
doctor_df = pd.read_csv("helper_data/Doctor.csv")

# Dummy user for login
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# Helper functions for symptom-based prediction
def get_prediction(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1
    return disease_dict[model2.predict([input_vector])[0]]

# Routes
@app.route('/')
def home():
    return render_template('index1.html', symptoms=symptoms_dict.keys())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict1.html')

@app.route('/predict', methods=['GET','POST'])
def predict_disease():
    try:
        # Model 1-based prediction
        fever = 1 if request.form.get('fever') == 'Yes' else 0
        cough = 1 if request.form.get('cough') == 'Yes' else 0
        breathing = 1 if request.form.get('breathing') == 'Yes' else 0
        gender = 1 if request.form.get('gender') == 'Female' else 0
        age = float(request.form.get('age', 0))
        bp = float(request.form.get('blood_pressure', 0))
        cholesterol = float(request.form.get('cholesterol', 0))

        features = np.array([[fever, cough, breathing, age, gender, bp, cholesterol]])
        prediction = model1.predict(features)
        predicted_class = int(prediction[0])
        predicted_disease = disease_labels.get(predicted_class, "Unknown Disease")

        return render_template('predict1.html', prediction_text=f"Predicted Disease: {predicted_disease}")
    except Exception as e:
        return render_template('predict1.html', prediction_text=f"Error: {str(e)}")

@app.route('/symptom_predict', methods=['GET', 'POST'])
def symptom_predict():
    if request.method == 'POST':
        # Model 2-based symptom prediction
        symptoms_input = request.form.getlist("symptoms")
        predicted_disease = get_prediction(symptoms_input)
        desc, precaution, symptoms, medicine, diet, workout, doctor = get_details(predicted_disease)

        return render_template("result.html",
                               disease=predicted_disease,
                               description=desc,
                               precautions=precaution,
                               symptoms=symptoms,
                               medications=medicine,
                               diets=diet,
                               workouts=workout,
                               doctors=doctor.to_dict(orient='records'))
    else:
        # Handle GET request by rendering the symptom selection page
        return render_template('jio.html', symptoms=symptoms_dict.keys())

if __name__ == '__main__':
    app.run(debug=True)
