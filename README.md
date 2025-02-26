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

**1. Medical Image Multiclass Classifier:**

Model: Fine-tuned Vision Transformer (ViT_base_patch16_224)

**Dataset: Mpox Skin Lesion Dataset v2.0**

**Accuracy: 88.89%**

Classes: Chickenpox, Cowpox, HFMD, Measles, Monkeypox, Other

**2. Tabular Mpox Patient Data Binary Classifier:**

Model: Fine-tuned distilroberta-base

**Dataset: Mpox Patient Dataset**

**Accuracy: 74.68%**

Purpose: Evaluates patient symptoms and medical history to detect the likelihood of Mpox

Outcome: Binary prediction (Mpox Positive or Negative)


# Installation & Setup

**1. Download the Project:**

**Download the folder as a zip file from this link:** https://drive.google.com/drive/folders/1a9xyKWCktcuUjtwGahsacjPBl14EIu40?usp=sharing



![Screenshot (13976)](https://github.com/user-attachments/assets/a586620d-69de-40e8-9a29-c4f4f3a1c514)



**2. Extract Files:**

Extract the downloaded zip file to a local directory.



![Screenshot (13977)](https://github.com/user-attachments/assets/a9760cf9-f91c-4b4e-aead-972834a68732)



**Ensure all files are in the same directory.**

**Note: If the files are not in the same directory, update the model paths in the code where the models are loaded.**



![Screenshot (13978)](https://github.com/user-attachments/assets/33438850-9760-4c4e-938c-7f7610b36cca)



**3. Open in Code Editor:**

**Open the extracted folder in your favorite code editor or IDE.**


**4. Run the Web App:**

**Open a terminal in the project folder and run:** streamlit run app_main.py



![Screenshot (13980)](https://github.com/user-attachments/assets/87ad84a5-44dc-45c1-ac23-1d69ac0fbec8)



**5. The Streamlit web app will open in your local browser. (Note: This may take a few minutes as the transformers library downloads necessary resources.)**



![Screenshot (13981)](https://github.com/user-attachments/assets/cce6bb7a-0e65-4a67-9120-88c92907a4e7)



# Usage Instructions

**Once the web app opens in your browser, you can:**


# Image-Based Diagnosis:

**Upload an image of a skin lesion to the Image Classifier for analysis and diagnosis.**



![Screenshot (13982)](https://github.com/user-attachments/assets/dc3dde8a-8155-4989-8edf-c7c871aa0ae7)



![Screenshot (13983)](https://github.com/user-attachments/assets/0c9ccf1d-a9c6-4278-99ab-226dd830f64d)




# Patient Data Classification:

**Fill in the patient symptoms section to determine whether you are Mpox positive or negative via the binary classifier.**



![Screenshot (13984)](https://github.com/user-attachments/assets/86141fea-2f4b-4fc6-8902-751e7c276119)



![Screenshot (13985)](https://github.com/user-attachments/assets/c267bf7d-551c-4453-a2ee-71a5998ec7bd)



# Multilingual Support:

**Change the language to Spanish from the sidebar (default language is English).**



![Screenshot (13987)](https://github.com/user-attachments/assets/81cf0570-c258-4986-9669-c59cb0478a4b)


# Downloadable Reports:

**Download diagnostic reports in PDF format for both the image analysis and patient data classification.**



![Screenshot (13986)](https://github.com/user-attachments/assets/49080ca7-ec39-4f01-ae74-fa1be7d1e922)


# Doctor Consultation:

**Click on the provided doctor consultation link to book a consultation.**



![Screenshot (13999)](https://github.com/user-attachments/assets/bc18c1df-0da7-4c88-a57e-459faf899e7c)

