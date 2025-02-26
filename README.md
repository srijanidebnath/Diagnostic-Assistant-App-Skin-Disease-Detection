# Diagnostic-Assistant-App-Skin-Disease-Detection

# Overview

This project is a multi-model diagnostic assistant that leverages AI to analyze medical images and patient data for accurate skin disease diagnosis. It aids healthcare 

professionals and the general public in the early detection of Mpox (formerly known as monkeypox) and other skin conditions, facilitating prompt medical intervention.

Features:

Multi-Model Approach:

Image Classifier: Uses a fine-tuned Vision Transformer (ViT_base_patch16_224) to classify skin lesions into Chickenpox, Cowpox, HFMD, Measles, Monkeypox, and Other with 88.89% accuracy.
Patient Data Classifier: A fine-tuned distilroberta-base binary classifier that evaluates patient symptoms and medical history to determine the likelihood of Monkeypox (Mpox) having an accuracy of 74.68%.

Web Application:


1. Built using Streamlit.
   
2. Image-Based Skin Lesion Detection: Upload an image to get diagnostic predictions.
   
3. Patient Data Classification: Input patient symptoms to assess Mpox likelihood.
   
4. Multilingual Support: Available in English (default) and Spanish.
   
5. Downloadable Reports: Generate and download PDF reports for both diagnostic analyses.

6. Doctor Consultation Links: Direct links to book consultations with a doctor.

   
# Project Background

# Mpox Resurgence:

Mpox, formerly known as monkeypox, has recently re-emerged with over 15,000 cases and 461 deaths reported globally, especially affecting sub-Saharan Africa.

The Clade 1b variant of Mpox, with a mortality rate of approximately 3%, has raised global health concerns.

As of September 2024, India has reported 32 cases since the World Health Organization (WHO) declared Mpox a public health emergency in 2022.

Need for Early Detection:

With India's dense population, early detection and prevention are critical to prevent potential outbreaks.

Misdiagnosis due to overlapping symptoms with other skin conditions can delay treatment and increase disease transmission.

Model Details:

1. Medical Image Multiclass Classifier:

Model: Fine-tuned Vision Transformer (ViT_base_patch16_224)

Dataset: Mpox Skin Lesion Dataset v2.0

Accuracy: 88.89%

Classes: Chickenpox, Cowpox, HFMD, Measles, Monkeypox, Other

2. Tabular Mpox Patient Data Binary Classifier:

Model: Fine-tuned distilroberta-base

Dataset: Mpox Patient Dataset

Accuracy: 74.68%

Purpose: Evaluates patient symptoms and medical history to detect the likelihood of Mpox

Outcome: Binary prediction (Mpox Positive or Negative)


# Installation & Setup

1. Download the Project:

Download the folder as a zip file from this link: https://drive.google.com/drive/folders/1a9xyKWCktcuUjtwGahsacjPBl14EIu40?usp=sharing

![Screenshot (13976)](https://github.com/user-attachments/assets/a586620d-69de-40e8-9a29-c4f4f3a1c514)


2. Extract Files:

Extract the downloaded zip file to a local directory.

Ensure all files are in the same directory.

Note: If the files are not in the same directory, update the model paths in the code where the models are loaded.

3. Open in Code Editor:

Open the extracted folder in your favorite code editor or IDE.

4. Run the Web App:

Open a terminal in the project folder and run: streamlit run app_main.py

5. The Streamlit web app will open in your local browser. (Note: This may take a few minutes as the transformers library downloads necessary resources.)

# Usage Instructions

Once the web app opens in your browser, you can:

# Image-Based Diagnosis:

Upload an image of a skin lesion to the Image Classifier for analysis and diagnosis.

# Patient Data Classification:

Fill in the patient symptoms section to determine whether you are Mpox positive or negative via the binary classifier.

# Multilingual Support:

Change the language to Spanish from the sidebar (default language is English).

# Downloadable Reports:

Download diagnostic reports in PDF format for both the image analysis and patient data classification.

# Doctor Consultation:

Click on the provided doctor consultation link to book a consultation.
