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
