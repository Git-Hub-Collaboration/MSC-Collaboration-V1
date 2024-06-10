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

# Feature Correlation Visualization
st.subheader("Feature Correlation")
visualize_required = 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area'
selected_features = st.multiselect("Select features to visualize correlation", data.columns, default=visualize_required)
if st.button("Predict"):
    user_input_tensor = convert_to_tensor_input(user_input_df)
    prediction = model.predict(user_input_tensor)
    st.write(f"Prediction: {prediction[0]}")

# Visualize Predictions
st.title("Prediction Visualization")
st.write("Visualizing the predictions made by the model")

# Ensure prediction is only accessed if it is set
if prediction is not None:
    fig, ax = plt.subplots()
    ax.bar(['Prediction'], [prediction[0]])
    st.pyplot(fig)
