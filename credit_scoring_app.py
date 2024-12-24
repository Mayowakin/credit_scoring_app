import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title and description
st.title("Bank Credit Scoring - Machine Learning App")
st.write("This app predicts whether a client will subscribe to a term deposit using machine learning.")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("bank.csv", sep=";")
    return data

data = load_data()

# Data Cleaning and Encoding
label_encoders = {}
categorical_mappings = {}

# Apply Label Encoding and Save Mappings
for col in data.columns:
    if data[col].dtype == object:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        categorical_mappings[col] = dict(zip(le.transform(le.classes_), le.classes_))

# Feature and target split
x = data.drop("y", axis=1)
y = data["y"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Scaling the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model Training (Random Forest Classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# User Input Section
st.sidebar.header("Input Parameters")

def user_input_features():
    inputs = {}
    for col in x.columns:
        if col in categorical_mappings:  # If the column is categorical
            # Use descriptive labels for dropdown options
            options = list(categorical_mappings[col].values())
            selected = st.sidebar.selectbox(f"{col}", options)
            # Map back to encoded numerical value
            inputs[col] = list(categorical_mappings[col].keys())[list(categorical_mappings[col].values()).index(selected)]
        else:  # Continuous variables
            inputs[col] = st.sidebar.slider(f"{col}", int(data[col].min()), int(data[col].max()))
    return pd.DataFrame([inputs])

input_data = user_input_features()

# Predicting for User Input
if st.sidebar.button("Predict"):
    input_data_scaled = sc.transform(input_data)
    prediction = clf.predict(input_data_scaled)
    result = "Subscribed" if prediction[0] == 1 else "Not Subscribed"
    st.write(f"## Prediction: {result}")
