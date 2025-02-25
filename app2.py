import streamlit as st
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import base64
import io
from fpdf import FPDF

# Load the trained ViT model
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ViTClassifier, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ViTClassifier(num_classes=6).to(device)
model.load_state_dict(torch.load(r"C:\Users\srija\Downloads\best_vit_model_epoch10_ft10.pth", map_location=device))
model.eval()

# Class labels
class_labels = ["Chickenpox", 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']  

# Language options
language_options = {
    "English": {
        "title": "🔎 AI-Powered Medical Image Classifier",
        "upload": "Upload an Image",
        "prediction": "Prediction",
        "healthy_message": "✅ You are **healthy**! No signs of infection.",
        "infected_message": "⚠️ You may have **{}**. Please consult a doctor.",
        "confidence": "Confidence: {:.2f}%",
        "save_report": "Download Report",
        "doctor_link": "📞 Book a Consultation with a Doctor",
        "sidebar_title": "About This Model",
        "sidebar_model": "Model: Vision Transformer (ViT)",
        "sidebar_trained": "Trained on: Medical Dataset",
        "sidebar_usage": "Usage: Identifying skin lesions",
        "sidebar_accuracy": "Accuracy: 88.89%"
    },
    "Spanish": {
        "title": "🔎 Clasificador de Imágenes Médicas con IA",
        "upload": "Sube una imagen",
        "prediction": "Predicción",
        "healthy_message": "✅ ¡Estás **sano**! No hay signos de infección.",
        "infected_message": "⚠️ Puede que tengas **{}**. Consulta a un médico.",
        "confidence": "Confianza: {:.2f}%",
        "save_report": "Descargar Informe",
        "doctor_link": "📞 Reserva una consulta con un médico",
        "sidebar_title": "Sobre este modelo",
        "sidebar_model": "Modelo: Vision Transformer (ViT)",
        "sidebar_trained": "Entrenado en: Conjunto de datos médicos",
        "sidebar_usage": "Uso: Identificación de lesiones cutáneas",
        "sidebar_accuracy": "Precisión: 88.89%"
    }
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="Medical Image Classifier", page_icon="🩺", layout="centered")

# Language selection
language = st.sidebar.selectbox("🌍 Select Language / Seleccionar idioma", ["English", "Spanish"])
text = language_options[language]

st.title(text["title"])
st.write(text["upload"])

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Store previous uploads
if "previous_results" not in st.session_state:
    st.session_state.previous_results = []

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)  # Smaller image size

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = class_labels[predicted_class]
        confidence = probabilities[0][predicted_class].item() * 100

    # Display prediction message
    st.subheader(text["prediction"])
    if predicted_label == "Healthy":
        st.success(text["healthy_message"])
    else:
        st.error(text["infected_message"].format(predicted_label))

    st.write(text["confidence"].format(confidence))

    # Save result in session state
    st.session_state.previous_results.append((image, predicted_label, confidence))

    # Generate PDF Report
    def generate_pdf(image, prediction, confidence):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, text["title"], ln=True, align="C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, f"Prediction: {prediction}", ln=True, align="L")
        pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True, align="L")

        # Convert image to JPEG and add to PDF
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(img_byte_arr.getvalue())

        pdf.image(img_path, x=50, y=None, w=100)

        # Save to bytes
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        return pdf_output

    # Download report button
    pdf_bytes = generate_pdf(image, predicted_label, confidence)
    b64_pdf = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="medical_report.pdf">{text["save_report"]}</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Doctor consultation link
    st.markdown(f"[{text['doctor_link']}]({'https://www.telemedicine.com'})")

# Show previous results if available
if len(st.session_state.previous_results) > 1:
    st.subheader("📊 Compare Previous Uploads")
    df = pd.DataFrame(st.session_state.previous_results, columns=["Image", "Prediction", "Confidence"])
    df["Confidence"] = df["Confidence"].astype(str) + "%"
    st.table(df[["Prediction", "Confidence"]])

# Sidebar Info
st.sidebar.header(text["sidebar_title"])
st.sidebar.write(f"- **{text['sidebar_model']}**")
st.sidebar.write(f"- **{text['sidebar_trained']}**")
st.sidebar.write(f"- **{text['sidebar_usage']}**")
st.sidebar.write(f"- **{text['sidebar_accuracy']}**")
