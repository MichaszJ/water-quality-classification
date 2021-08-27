import streamlit as st

import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# header
st.title('Water Potability Classifier')
st.write('Created by Michal Jagodzinski')
st.write('WARNING! This is not to be used for real-world classification, it is not guaranteed to give accurate results')

# importing model and scaler
svc_model = pickle.load(open('svc-model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# getting user input
st.markdown('## Enter Water Parameters')

features = {
    'pH Value': 0, 
    'Hardness [mg/L]': 0, 
    'Solids [ppm]': 0, 
    'Chloramines [ppm]': 0, 
    'Sulfates [mg/L]': 0, 
    'Conductivity [μS/cm]': 0, 
    'Organic Carbon [ppm]': 0, 
    'Trihalomethanes [μg/L]': 0,
    'Turbidity [NTU]': 0
}

for feature in features:
    features[feature] = st.text_input(f'{feature}: ', 0)

if all((features[feature] != 0 and features[feature] != "") for feature in features):
    st.write('Input Data:')
    st.write(features)

    data = np.array([float(features[feature]) for feature in features])
    scaled_data = scaler.transform(data.reshape(1, -1))

    st.write('Scaled Data:')
    st.write(scaled_data)

    prediction = svc_model.predict(scaled_data)

    st.write(f'Is the water safe to drink? {prediction[0] == 1}')