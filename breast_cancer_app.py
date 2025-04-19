import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer

# Set page config
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# Title and description
st.title("Breast Cancer Detection System")
st.write("This application predicts whether a breast mass is benign or malignant based on measurements.")

# Add definitions in a highlighted box
st.info("""
**Important Terms:**
- **Benign**: Noncancerous tumors
- **Malignant**: Cancerous tumors

[Learn more about benign and malignant tumors](https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240)
""")

# Load and prepare data
@st.cache_data
def load_data():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model, imputer

# Load data and train model
X, y = load_data()
model, imputer = train_model(X, y)

# Create input form
st.header("Patient Measurements")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mean Values")
    radius_mean = st.number_input("Mean Radius", value=14.0)
    texture_mean = st.number_input("Mean Texture", value=14.0)
    perimeter_mean = st.number_input("Mean Perimeter", value=87.0)
    area_mean = st.number_input("Mean Area", value=566.0)
    smoothness_mean = st.number_input("Mean Smoothness", value=0.098)
    compactness_mean = st.number_input("Mean Compactness", value=0.08)
    concavity_mean = st.number_input("Mean Concavity", value=0.07)
    concave_points_mean = st.number_input("Mean Concave Points", value=0.048)
    symmetry_mean = st.number_input("Mean Symmetry", value=0.19)
    fractal_dimension_mean = st.number_input("Mean Fractal Dimension", value=0.058)

with col2:
    st.subheader("Standard Error Values")
    radius_se = st.number_input("Radius SE", value=0.4)
    texture_se = st.number_input("Texture SE", value=1.2)
    perimeter_se = st.number_input("Perimeter SE", value=2.8)
    area_se = st.number_input("Area SE", value=40.0)
    smoothness_se = st.number_input("Smoothness SE", value=0.007)
    compactness_se = st.number_input("Compactness SE", value=0.02)
    concavity_se = st.number_input("Concavity SE", value=0.03)
    concave_points_se = st.number_input("Concave Points SE", value=0.01)
    symmetry_se = st.number_input("Symmetry SE", value=0.02)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)

with col3:
    st.subheader("Worst Values")
    radius_worst = st.number_input("Worst Radius", value=16.0)
    texture_worst = st.number_input("Worst Texture", value=20.0)
    perimeter_worst = st.number_input("Worst Perimeter", value=100.0)
    area_worst = st.number_input("Worst Area", value=700.0)
    smoothness_worst = st.number_input("Worst Smoothness", value=0.12)
    compactness_worst = st.number_input("Worst Compactness", value=0.15)
    concavity_worst = st.number_input("Worst Concavity", value=0.2)
    concave_points_worst = st.number_input("Worst Concave Points", value=0.1)
    symmetry_worst = st.number_input("Worst Symmetry", value=0.3)
    fractal_dimension_worst = st.number_input("Worst Fractal Dimension", value=0.08)

if st.button("Predict"):
    try:
        # Create input array with all 30 features
        input_data = np.array([
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
            fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
            smoothness_se, compactness_se, concavity_se, concave_points_se,
            symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
            perimeter_worst, area_worst, smoothness_worst, compactness_worst,
            concavity_worst, concave_points_worst, symmetry_worst,
            fractal_dimension_worst
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Show result
        st.header("Prediction Result")
        if prediction[0] == 0:  # In sklearn dataset, 0 = malignant, 1 = benign
            st.error("Result: Malignant (Cancerous)")
        else:
            st.success("Result: Benign (Non-cancerous)")
        
        # Add confidence note
        st.info("Note: This is a preliminary screening tool. Please consult with healthcare professionals for proper diagnosis.")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("### About")
st.write("""
This breast cancer detection system uses machine learning to predict whether a breast mass is **benign** (non-cancerous) or **malignant** (cancerous).
The model is trained on the Wisconsin Breast Cancer dataset from scikit-learn and uses various measurements of the breast mass to make predictions.
""")



