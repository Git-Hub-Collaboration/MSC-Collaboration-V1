import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the cleaned data
data_path = 'training_data.csv'
data = pd.read_csv(data_path)

model_path = 'nn_model.h5'
model = load_model(model_path)
# Data Preprocessing Function
def preprocess_data(data, label_encoder):
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    
    return data

# Convert data to the appropriate type for TensorFlow
def convert_to_tensor_input(data):
    return np.asarray(data).astype(np.float32)

# Data visualization
st.title("Loan Prediction  Analysis")
st.write("Visualizing the cleaned data used to train the model")

st.dataframe(data.head())
