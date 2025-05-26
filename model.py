import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import catboost as cb

# Load dataset
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Drop unwanted columns
df.drop(columns=['Fatigue', 'Outcome Variable'], inplace=True)

# Encode string columns
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])  # <-- THIS gets saved

for column in df.select_dtypes(include=['object']).columns:
    if column != 'Disease':
        df[column] = LabelEncoder().fit_transform(df[column])

# Features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
model = cb.CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model1.pkl', 'wb'))

# ✅ Save the label encoder too
pickle.dump(label_encoder, open('disease_label_encoder.pkl', 'wb'))

print("Model and label encoder saved successfully.")
