# Breast Cancer Detection System

A machine learning-based web application that predicts whether a breast mass is benign or malignant using various measurements. This tool is designed to assist in preliminary breast cancer screening.

> **Important Terms:**
> - **Benign** tumors are noncancerous
> - **Malignant** tumors are cancerous
>
> [Learn more about benign and malignant tumors](https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240)

## Overview

This application uses a Support Vector Machine (SVM) classifier trained on the Wisconsin Breast Cancer dataset to predict breast cancer diagnosis. It provides a user-friendly interface where medical professionals can input various measurements of breast mass and receive an immediate prediction.

## Features

- Real-time prediction of breast cancer diagnosis (Benign/Malignant)
- Interactive web interface built with Streamlit
- Support Vector Machine classifier with 95%+ accuracy
- Input validation and error handling
- Comprehensive measurement input system including:
  - Mean values of cell nuclei
  - Standard error measurements
  - Worst (largest) values of features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run breast_cancer_app.py
```

## Required Dependencies

- Python 3.7+
- streamlit
- pandas
- numpy
- scikit-learn

## Usage

1. Launch the application using the command above
2. Enter the required measurements in the input fields:
   - Mean Values (radius, texture, perimeter, etc.)
   - Standard Error Values
   - Worst Values
3. Click the "Predict" button
4. View the prediction result (Benign or Malignant)

## Dataset

The system uses the Wisconsin Breast Cancer dataset, which includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

Features include:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

## Model Performance

- Algorithm: Support Vector Machine (SVM) with linear kernel
- Accuracy: ~95.6%
- Training/Test Split: 80/20

## Important Note

This tool is designed for preliminary screening only and should not be used as the sole basis for diagnosis. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment.

## Acknowledgments

- Wisconsin Breast Cancer Dataset
- Scikit-learn team
- Streamlit framework developers
