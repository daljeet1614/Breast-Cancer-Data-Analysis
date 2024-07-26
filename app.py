import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target

# Map target values to 'M' and 'B'
df['diagnosis'] = df['diagnosis'].map({0: 'B', 1: 'M'})

# Split data
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]

# Train model
mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001, learning_rate='adaptive', max_iter=1000, random_state=42)
mlp.fit(X_train_selected, y_train)

# Streamlit app
st.title("Breast Cancer Diagnosis Prediction")

st.write("## Dataset Overview")
st.write(df.head())

st.write("## Selected Features")
st.write(selected_features)

st.write("## Model Evaluation")
y_pred = mlp.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("## Make a Prediction")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"Enter value for {feature}", float(X[feature].min()), float(X[feature].max()))

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_selected = selector.transform(input_df)
    prediction = mlp.predict(input_selected)
    st.write("Prediction (M=1, B=0):", prediction[0])
