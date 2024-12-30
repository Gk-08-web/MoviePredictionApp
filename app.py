# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "C:/Users/gargi/MoviePrediction/movie_metadata_renamed.csv"  # Ensure this path is correct
data = pd.read_csv(file_path)

# Handle missing values
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Encode target variable
label_encoder = LabelEncoder()
data['Category'] = data['imdb_score'].apply(lambda score: 'Hit' if score >= 6 else ('Average' if score >= 3 else 'Flop'))
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])

# Define features and target variable
X = data[numeric_cols]
y = data['Category_encoded']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Streamlit UI elements
st.title("Movie Rating Prediction")
num_critic_for_reviews = st.number_input("Number of Critics for Reviews", min_value=0)
duration = st.number_input("Duration (minutes)", min_value=0)
budget = st.number_input("Budget", min_value=0)
imdb_score = st.number_input("IMDB Score", min_value=0.0, max_value=10.0)

if st.button("Predict"):
    input_data = np.array([[num_critic_for_reviews, duration, budget, imdb_score]])
    prediction = rf.predict(input_data)
    category = label_encoder.inverse_transform(prediction)[0]
    st.write(f"Predicted Category: {category}")
