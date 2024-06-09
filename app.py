import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the cleaned data
data_path = 'transformed.csv'
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
selected_features = st.multiselect("Select features to visualize correlation", data.columns, default=data.columns.tolist())
if selected_features:
    fig, ax = plt.subplots()
    sns.heatmap(data[selected_features].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.subheader("Feature Distribution")
selected_feature = st.selectbox("Select feature to visualize", data.columns)
fig, ax = plt.subplots()
ax.hist(data[selected_feature].dropna(), bins=30)
st.pyplot(fig)

# Upload new cleaned data for prediction
st.title("Upload Cleaned Data for Prediction")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write(new_data.head())
    # Preprocess the new data
    new_data = preprocess_data(new_data, LabelEncoder())
    new_data_tensor = convert_to_tensor_input(new_data)
    
    if st.button("Predict New Data"):
        new_predictions = model.predict(new_data_tensor)
        st.write("Predictions for the uploaded data:", new_predictions)

# Prediction
st.title("Make Predictions")
st.write("Input data for prediction")

# Collect user input
user_input = {}
for feature in data.columns:
    if feature in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']:
        user_input[feature] = st.selectbox(f"Select {feature}", data[feature].unique())
    else:
        user_input[feature] = st.text_input(f"Input {feature}")

user_input_df = pd.DataFrame(user_input, index=[0])

# Preprocess user input
label_encoder = LabelEncoder()
user_input_df = preprocess_data(user_input_df, label_encoder)

prediction = None

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
