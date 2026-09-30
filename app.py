"""
🌿 Kilimo Smart — AI Agronomist Platform (PyTorch / ResNet-18)
==============================================================
Plant Disease Detection + Market Intelligence + Weather
Author  : Paul N. Magima
Contact : emryspaul7@gmail.com | +254 759 670 456
"""

import io
import os
import logging

from PIL import UnidentifiedImageError

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

# ==================================================================
# 1️⃣  MODEL ARCHITECTURE — ResNet-18 (DO NOT CHANGE)
# ==================================================================

def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==================================================================
# 2️⃣  CLASS NAMES — 15 PlantVillage classes (DO NOT CHANGE ORDER)
# ==================================================================
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ==================================================================
# 3️⃣  MODEL FILE PATH (DO NOT CHANGE)
# ==================================================================
MODEL_PATH = "plant_disease_model.pth"

# ==================================================================
# 4️⃣  PREPROCESSING (DO NOT CHANGE — must match training pipeline)
# ==================================================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ==================================================================
# 5️⃣  DEVICE
# ==================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Running on: {DEVICE}")

# ==================================================================
# 6️⃣  LOAD MODEL (DO NOT CHANGE)
# ==================================================================
model = build_model(num_classes=len(CLASS_NAMES))

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as checkpoint:
        is_lfs_pointer = checkpoint.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
    if is_lfs_pointer:
        logger.warning("Model file is a Git LFS pointer; download the actual checkpoint before serving predictions")
        model = None
    else:
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            model = model.to(DEVICE)
            model.eval()
            logger.info("Model loaded from %s", MODEL_PATH)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Model unavailable: %s", exc)
            model = None
else:
    logger.warning("Model checkpoint not found: %s", MODEL_PATH)
    model = None

# ==================================================================
# 7️⃣  DISEASE INFO DATABASE
# ==================================================================
DISEASE_INFO = {

    "Pepper__bell___Bacterial_spot": {
        "label": "Pepper — Bacterial Spot",
        "severity": "moderate",
        "description": "Bacterial infection (Xanthomonas) causing small, water-soaked spots on leaves and fruits that turn brown with yellow halos. Spreads rapidly in warm, wet conditions.",
        "remedy": "Use disease-free planting material. Remove and destroy all crop debris after harvest or plow deeply into soil. Avoid overhead irrigation. Rotate crops with non-solanaceous plants.",
        "prevention": "Use certified disease-free seeds. Space plants for good airflow. Apply copper-based bactericide preventively during wet seasons."
    },

    "Pepper__bell___healthy": {
        "label": "Pepper — Healthy 🌱",
        "severity": "none",
        "description": "Your pepper plant appears to be completely healthy. No signs of disease or infection were detected.",
        "remedy": "No treatment needed. Continue regular care.",
        "prevention": "Monitor regularly for pests. Maintain even watering and balanced fertilization."
    },

    "Potato___Early_blight": {
        "label": "Potato — Early Blight",
        "severity": "moderate",
        "description": "Caused by Alternaria solani fungus. Shows as dark brown, circular spots with concentric rings on older lower leaves first.",
        "remedy": "Apply mancozeb or chlorothalonil fungicide at first symptoms. Remove heavily infected leaves. Ensure proper plant spacing.",
        "prevention": "Use certified seed. Rotate crops every 2–3 years. Apply preventive fungicide during wet weather."
    },

    "Potato___Late_blight": {
        "label": "Potato — Late Blight",
        "severity": "severe",
        "description": "Caused by Phytophthora infestans — the same pathogen that caused the Irish Famine. Fast-spreading, can destroy an entire field in days.",
        "remedy": "Apply systemic fungicides (metalaxyl + mancozeb) immediately. Remove and burn infected plant material. Avoid overhead irrigation.",
        "prevention": "Plant resistant varieties. Apply preventive fungicide before and during rainy season. Destroy volunteer potato plants."
    },

    "Potato___healthy": {
        "label": "Potato — Healthy 🌱",
        "severity": "none",
        "description": "Your potato plant appears completely healthy. No signs of disease detected.",
        "remedy": "No treatment needed.",
        "prevention": "Monitor regularly. Maintain good drainage and crop rotation."
    },

    "Tomato___Bacterial_spot": {
        "label": "Tomato — Bacterial Spot",
        "severity": "moderate",
        "description": "Caused by Xanthomonas bacteria. Small, water-soaked spots on leaves, stems and fruit that turn dark with yellow halos.",
        "remedy": "Remove infected plant material. Apply copper-based bactericide. Avoid overhead irrigation.",
        "prevention": "Use certified seeds. Maintain plant spacing. Apply copper spray preventively in wet seasons."
    },

    "Tomato___Early_blight": {
        "label": "Tomato — Early Blight",
        "severity": "moderate",
        "description": "Caused by Alternaria solani. Dark brown spots with concentric rings (target pattern) on older leaves, yellowing and defoliation.",
        "remedy": "Apply mancozeb or chlorothalonil fungicide. Remove and burn infected leaves. Stake plants for airflow.",
        "prevention": "Rotate crops every 2 years. Mulch soil to prevent splash. Remove infected material promptly."
    },

    "Tomato___Late_blight": {
        "label": "Tomato — Late Blight",
        "severity": "severe",
        "description": "Caused by Phytophthora infestans. Water-soaked lesions on leaves that turn brown-black. White mold appears on undersides in humid conditions.",
        "remedy": "Apply metalaxyl-based fungicide immediately. Remove all infected material and burn. Avoid wet foliage.",
        "prevention": "Plant resistant varieties. Never leave infected material in the field. Monitor weather — apply protective fungicide before wet periods."
    },

    "Tomato___Leaf_Mold": {
        "label": "Tomato — Leaf Mold",
        "severity": "moderate",
        "description": "Caused by Passalora fulva fungus. Pale greenish-yellow spots on upper leaf surface with olive-green velvety mold on the underside.",
        "remedy": "Grow resistant varieties. Avoid overhead watering. Apply fungicide (mancozeb or copper) if severe.",
        "prevention": "Maintain humidity below 85%. Increase plant spacing. Remove lower infected leaves promptly."
    },

    "Tomato___Septoria_leaf_spot": {
        "label": "Tomato — Septoria Leaf Spot",
        "severity": "moderate",
        "description": "Caused by Septoria lycopersici. Small circular spots with dark margins and grey centres with tiny black dots inside.",
        "remedy": "Remove and destroy infected debris. Apply chlorothalonil or copper fungicide if needed.",
        "prevention": "Rotate crops — do not plant tomatoes in same spot for 2+ years. Mulch to prevent soil splash."
    },

    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "label": "Tomato — Spider Mites",
        "severity": "moderate",
        "description": "Infestation by Tetranychus urticae. Tiny mites cause stippling on leaves, fine webbing on undersides, and eventual leaf bronzing.",
        "remedy": "Use miticides or insecticidal soaps. Introduce predatory mites for biological control.",
        "prevention": "Monitor regularly. Avoid water stress. Use reflective mulch to disorient mites."
    },

    "Tomato___Target_Spot": {
        "label": "Tomato — Target Spot",
        "severity": "moderate",
        "description": "Caused by Corynespora cassiicola. Brown lesions with concentric rings on leaves, stems and fruit.",
        "remedy": "Remove plant debris. Apply azoxystrobin or chlorothalonil if severe.",
        "prevention": "Improve air circulation. Avoid overhead irrigation. Rotate crops."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "label": "Tomato — Yellow Leaf Curl Virus",
        "severity": "severe",
        "description": "Viral disease transmitted by silverleaf whiteflies. Causes upward curling and yellowing of leaves, stunted growth, severe yield loss.",
        "remedy": "Control whiteflies with insecticides or yellow sticky traps. Remove and destroy infected plants immediately.",
        "prevention": "Use certified virus-free transplants. Install insect-proof netting. Plant resistant varieties in high-risk areas."
    },

    "Tomato___Tomato_mosaic_virus": {
        "label": "Tomato — Mosaic Virus",
        "severity": "severe",
        "description": "Caused by Tomato mosaic virus. Mottled light and dark green mosaic pattern on leaves, leaf distortion, stunting.",
        "remedy": "Use virus-free seeds. Sanitize tools with 10% bleach. Remove and destroy infected plants.",
        "prevention": "Use resistant varieties. Wash hands before handling plants. Disinfect tools between plants."
    },

    "Tomato___healthy": {
        "label": "Tomato — Healthy 🌱",
        "severity": "none",
        "description": "Your tomato plant appears completely healthy. No signs of disease, virus or pest damage detected.",
        "remedy": "No treatment needed. Your plant is doing great!",
        "prevention": "Monitor regularly. Maintain even watering and balanced NPK fertilization. Stake plants."
    },
}


# ==================================================================
# 8️⃣  PREDICTION FUNCTION (DO NOT CHANGE)
# ==================================================================

def predict_disease(image_bytes: bytes) -> dict:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(pil_img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    if model is not None:
        with torch.no_grad():
            logits = model(tensor)
        probs      = torch.softmax(logits, dim=1)[0]
        class_idx  = probs.argmax().item()
        confidence = float(probs[class_idx])
    else:
        raise RuntimeError("Disease model is unavailable")

    disease_key  = CLASS_NAMES[class_idx]
    disease_data = DISEASE_INFO.get(disease_key, {
        "label":       disease_key,
        "severity":    "unknown",
        "description": "Disease detected but no information available.",
        "remedy":      "Please consult a local agronomist.",
        "prevention":  "N/A",
    })

    return {
        "disease_key": disease_key,
        "confidence":  round(confidence, 4),
        **disease_data,
    }


# ==================================================================
# 9️⃣  FLASK APP & ROUTES
# ==================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/health")
def health():
    return jsonify({"status": "ok" if model is not None else "unavailable", "model_loaded": model is not None}), (200 if model is not None else 503)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image exceeds 10 MB upload limit"}), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Plant disease detection endpoint — called by JS fetch in the scanner tab."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image field found in the request"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file was selected"}), 400
        if model is None:
            return jsonify({"error": "Disease model is unavailable"}), 503
        if not file.mimetype.startswith("image/"):
            return jsonify({"error": "Upload an image file"}), 400
        image_bytes = file.read()
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({"error": "Invalid or corrupted image"}), 400
        result = predict_disease(image_bytes)
        return jsonify(result)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500


# ==================================================================
# 🚀  START SERVER
# ==================================================================

if __name__ == "__main__":
    print("🌿 Kilimo Smart server starting...")
    print("👤 Author: Paul N. Magima | emryspaul7@gmail.com")
    print("📡 Open browser at: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
