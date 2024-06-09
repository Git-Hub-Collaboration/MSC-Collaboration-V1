import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the cleaned data
data_path = 'df1_loan.csv'
data = pd.read_csv(data_path)

model_path = 'nn_model.h5'
model = load_model(model_path)

# Data Preprocessing Function
def preprocess_data(data, label_encoder):
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    
    return data

# Data visualization
st.title("Data Visualization")
st.write("Visualizing the cleaned data used to train the model")

st.dataframe(data.head())

# Example plot
st.subheader("Feature Distribution")
selected_feature = st.selectbox("Select feature to visualize", data.columns)
fig, ax = plt.subplots()
ax.hist(data[selected_feature].dropna(), bins=30)
st.pyplot(fig)

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

# Initialize prediction variable
prediction = None

if st.button("Predict"):
    prediction = model.predict(user_input_df)
    st.write(f"Prediction: {prediction[0]}")

# Visualize Predictions
st.title("Prediction Visualization")
st.write("Visualizing the predictions made by the model")

# Ensure prediction is only accessed if it is set
if prediction is not None:
    # Example prediction visualization
    fig, ax = plt.subplots()
    ax.bar(['Prediction'], [prediction[0]])
    st.pyplot(fig)