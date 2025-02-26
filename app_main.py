#Code for the integration of our dual-model framework with streamlit web app functionalities. 
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import timm
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import base64
import io
from fpdf import FPDF

# -----------------------------
# 1. LANGUAGE & UI STRINGS
# -----------------------------
language_options = {
    "English": {
        "title_image": "🔎 AI-Powered Medical Image Classifier",
        "upload_image": "Upload an Image",
        "prediction": "Prediction",
        "healthy_message": "No skin disease has been detected by our model. However, you may have other medical conditions. Please consult a doctor accordingly.",
        "infected_message": "⚠️ You may have **{}**. Please consult a doctor immidiately.",
        "confidence": "Confidence: {:.2f}%",
        "save_report": "Download Report",
        "doctor_link": "📞 Book a Consultation with a Doctor",
        "sidebar_title": "Medical Image Classifier",
        "sidebar_model": "Model: Vision Transformer (ViT)",
        "sidebar_trained": "Trained on: Medical Dataset",
        "sidebar_usage": "Usage: Identifying skin lesions",
        "sidebar_accuracy": "Accuracy: 88.89%",

        "sidebar_title_2": "Mpox Patient Symptoms Classifier",
        "sidebar_model2": "Model: DistilRoBERTa",
        "sidebar_trained2": "Trained on: Mpox Patient Symptom Data",
        "sidebar_usage2": "Usage: Classifying patient data for Mpox",
        "sidebar_accuracy2": "Accuracy: 74.68%",

        "title_tabular": "🦠 Mpox Patient Symptoms Classifier",
        "tabular_intro": "Enter Patient Details",
        "systemic_illness": "Systemic Illness",
        "rectal_pain": "Rectal Pain",
        "sore_throat": "Sore Throat",
        "penile_oedema": "Penile Oedema",
        "oral_lesions": "Oral Lesions",
        "solitary_lesion": "Solitary Lesion",
        "swollen_tonsils": "Swollen Tonsils",
        "hiv_infection": "HIV Infection",
        "sti_infection": "Sexually Transmitted Infection",
        "predict_button": "Predict Mpox Status",
        "tabular_prediction": "Mpox Prediction",
        "mpox_positive": "Patient is likely Mpox Positive. Consult a doctor immediately.",
        "mpox_negative": "Patient is likely Mpox Negative."
    },
    "Spanish": {
        "title_image": "🔎 Clasificador de Imágenes Médicas con IA",
        "upload_image": "Sube una imagen",
        "prediction": "Predicción",
        "healthy_message": "✅ ¡Estás **sano**! No hay signos de infección.",
        "infected_message": "⚠️ Puede que tengas **{}**. Consulta a un médico.",
        "confidence": "Confianza: {:.2f}%",
        "save_report": "Descargar Informe",
        "doctor_link": "📞 Reserva una consulta con un médico",
        "sidebar_title": "Clasificador de Imágenes Médicas",
        "sidebar_model": "Modelo: Vision Transformer (ViT)",
        "sidebar_trained": "Entrenado en: Conjunto de datos médicos",
        "sidebar_usage": "Uso: Identificación de lesiones cutáneas",
        "sidebar_accuracy": "Precisión: 88.89%",

        "sidebar_title_2": "Clasificador de Síntomas de Pacientes con Mpox",
        "sidebar_model2": "Modelo: DistilRoBERTa",
        "sidebar_trained2": "Entrenado en: Datos de Síntomas de Mpox",
        "sidebar_usage2": "Uso: Clasificar datos de pacientes con Mpox",
        "sidebar_accuracy2": "Precisión: 74.68%",

        "title_tabular": "🦠 Clasificador de Síntomas de Pacientes con Mpox",
        "tabular_intro": "Ingrese datos del paciente",
        "systemic_illness": "Enfermedad Sistémica",
        "rectal_pain": "Dolor Rectal",
        "sore_throat": "Dolor de Garganta",
        "penile_oedema": "Edema Peniano",
        "oral_lesions": "Lesiones Orales",
        "solitary_lesion": "Lesión Solitaria",
        "swollen_tonsils": "Amígdalas Hinchadas",
        "hiv_infection": "Infección por VIH",
        "sti_infection": "Infección de Transmisión Sexual",
        "predict_button": "Predecir Estado de Mpox",
        "tabular_prediction": "Predicción de Mpox",
        "mpox_positive": "El paciente probablemente sea positivo para Mpox. Consulte a un médico.",
        "mpox_negative": "El paciente probablemente sea negativo para Mpox."
    }
}

# -----------------------------
# 2. STREAMLIT SETUP
# -----------------------------
st.set_page_config(page_title="Diagnostic Assistant", page_icon="🩺", layout="centered")

# Language selection
language = st.sidebar.selectbox("🌍 Select Language / Seleccionar idioma", ["English", "Spanish"])
text = language_options[language]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 3. IMAGE CLASSIFIER (ViT)
# -----------------------------
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ViTClassifier, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

vit_model = ViTClassifier(num_classes=6).to(device)
vit_model.load_state_dict(torch.load("best_vit_model_epoch10_ft10.pth", map_location=device))
vit_model.eval()

class_labels = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# SIDEBAR INFO FOR IMAGE CLASSIFIER
# -----------------------------
st.sidebar.header(text["sidebar_title"])
st.sidebar.write(f"- **{text['sidebar_model']}**")
st.sidebar.write(f"- **{text['sidebar_trained']}**")
st.sidebar.write(f"- **{text['sidebar_usage']}**")
st.sidebar.write(f"- **{text['sidebar_accuracy']}**")

# -----------------------------
# IMAGE CLASSIFIER UI
# -----------------------------
st.subheader(text["title_image"])
st.write(text["upload_image"])

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if "previous_image_results" not in st.session_state:
    st.session_state.previous_image_results = []

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    # Prediction
    img_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = vit_model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_label = class_labels[predicted_class]
        confidence = probabilities[0][predicted_class].item() * 100

    # Build result text
    if predicted_label == "Healthy":
        result_text = f"<p style='color:green;'>{text['healthy_message']}</p>"
    else:
        result_text = f"<p style='color:red;'>{text['infected_message'].format(predicted_label)}</p>"
    confidence_msg = text["confidence"].format(confidence)

    # Bordered container
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; margin-top: 10px;">
            <h4>{text["prediction"]}</h4>
            {result_text}
            <p>{confidence_msg}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # PDF for Image Classifier
    def generate_pdf_image(image, prediction, confidence_val):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, text["title_image"].encode('latin-1', 'ignore').decode('latin-1'), ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Prediction: {prediction}", ln=True, align="L")
        pdf.cell(200, 10, f"Confidence: {confidence_val:.2f}%", ln=True, align="L")

        img_byte_arr = io.BytesIO()
        rgb_image = image.convert("RGB")
        rgb_image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(img_byte_arr.getvalue())

        pdf.image(img_path, x=50, y=None, w=80)

        pdf_output = io.BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)

        return pdf_output

    pdf_bytes = generate_pdf_image(image, predicted_label, confidence)
    b64_pdf = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="medical_image_report.pdf">{text["save_report"]}</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.markdown(f"[{text['doctor_link']}]({'https://www.telemedicine.com'})")

# -----------------------------
# 4. TABULAR CLASSIFIER (DistilRoBERTa)
# -----------------------------
tokenizer2 = AutoTokenizer.from_pretrained("distilroberta-base")
model2 = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
model2.load_state_dict(torch.load("best_distilbert_model.pth", map_location=device))
model2.to(device)
model2.eval()

# -----------------------------
# SIDEBAR INFO FOR MPox SYMPTOMS CLASSIFIER
# -----------------------------
st.sidebar.header(text["sidebar_title_2"])
st.sidebar.write(f"- **{text['sidebar_model2']}**")
st.sidebar.write(f"- **{text['sidebar_trained2']}**")
st.sidebar.write(f"- **{text['sidebar_usage2']}**")
st.sidebar.write(f"- **{text['sidebar_accuracy2']}**")

# -----------------------------
# TABULAR CLASSIFIER UI
# -----------------------------
st.subheader(text["title_tabular"])
st.write(text["tabular_intro"])

systemic_illness_options = ["None", "Fever", "Swollen Lymph Nodes", "Muscle Aches and Pain"]
systemic_illness = st.selectbox(text["systemic_illness"], systemic_illness_options)

def yes_no_dropdown(label):
    return st.selectbox(label, ["No", "Yes"])

rectal_pain_opt = yes_no_dropdown(text["rectal_pain"])
sore_throat_opt = yes_no_dropdown(text["sore_throat"])
penile_oedema_opt = yes_no_dropdown(text["penile_oedema"])
oral_lesions_opt = yes_no_dropdown(text["oral_lesions"])
solitary_lesion_opt = yes_no_dropdown(text["solitary_lesion"])
swollen_tonsils_opt = yes_no_dropdown(text["swollen_tonsils"])
hiv_infection_opt = yes_no_dropdown(text["hiv_infection"])
sti_infection_opt = yes_no_dropdown(text["sti_infection"])

if st.button(text["predict_button"]):
    def to_int(opt):
        return 1 if opt == "Yes" else 0

    bool_dict = {
        "RectalPain": to_int(rectal_pain_opt),
        "SoreThroat": to_int(sore_throat_opt),
        "PenileOedema": to_int(penile_oedema_opt),
        "OralLesions": to_int(oral_lesions_opt),
        "SolitaryLesion": to_int(solitary_lesion_opt),
        "SwollenTonsils": to_int(swollen_tonsils_opt),
        "HIVInfection": to_int(hiv_infection_opt),
        "STIInfection": to_int(sti_infection_opt),
    }

    text_features = str(systemic_illness) + " " + " ".join(str(v) for v in bool_dict.values())

    encoding = tokenizer2(
        text_features,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model2(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence_tab = probs[0, predicted_class].item() * 100

    label_map = {0: text["mpox_negative"], 1: text["mpox_positive"]}
    result_label = label_map[predicted_class]
    
    # For the PDF, show only "Positive"/"Negative" (or "Positivo"/"Negativo" in Spanish)
    if language == "English":
        pdf_prediction = "Positive" if predicted_class == 1 else "Negative"
    else:
        pdf_prediction = "Positivo" if predicted_class == 1 else "Negativo"

    # Bordered container
    st.markdown(
        f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; margin-top: 10px;">
            <h4>{text["tabular_prediction"]}</h4>
            <p><strong>{result_label}</strong></p>
            <p><strong>{text['confidence'].split(':')[0]}:</strong> {confidence_tab:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # PDF with tabular data
    def generate_pdf_tabular(systemic_val, bool_vals, prediction, confidence_val):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, text["title_tabular"].encode('latin-1', 'ignore').decode('latin-1'), ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.ln(5)

        # Table header
        pdf.cell(60, 10, "Feature", 1, 0, "C")
        pdf.cell(60, 10, "Value", 1, 1, "C")

        # Systemic Illness row
        pdf.cell(60, 10, text["systemic_illness"], 1, 0, "C")
        pdf.cell(60, 10, systemic_val, 1, 1, "C")

        # Boolean features in a table
        label_map_bool = {
            "RectalPain": text["rectal_pain"],
            "SoreThroat": text["sore_throat"],
            "PenileOedema": text["penile_oedema"],
            "OralLesions": text["oral_lesions"],
            "SolitaryLesion": text["solitary_lesion"],
            "SwollenTonsils": text["swollen_tonsils"],
            "HIVInfection": text["hiv_infection"],
            "STIInfection": text["sti_infection"],
        }

        for key, label_ in label_map_bool.items():
            val_str = "Yes" if bool_vals[key] == 1 else "No"
            pdf.cell(60, 10, label_, 1, 0, "C")
            pdf.cell(60, 10, val_str, 1, 1, "C")

        # Prediction row (only Positive/Negative)
        pdf.cell(60, 10, "Prediction", 1, 0, "C")
        pdf.cell(60, 10, prediction, 1, 1, "C")

        # Confidence row
        pdf.cell(60, 10, "Confidence", 1, 0, "C")
        pdf.cell(60, 10, f"{confidence_val:.2f}%", 1, 1, "C")

        pdf_output = io.BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        return pdf_output

    pdf_tab = generate_pdf_tabular(systemic_illness, bool_dict, pdf_prediction, confidence_tab)
    b64_pdf_tab = base64.b64encode(pdf_tab.read()).decode()
    href_tab = f'<a href="data:application/octet-stream;base64,{b64_pdf_tab}" download="mpox_tabular_report.pdf">{text["save_report"]}</a>'
    st.markdown(href_tab, unsafe_allow_html=True)

    st.markdown(f"[{text['doctor_link']}]({'https://www.telemedicine.com'})")
