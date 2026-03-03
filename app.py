"""
🌿 PlantMD — Plant Disease Detection App (PyTorch / ResNet-18)
==============================================================
Dataset : PlantVillage (15 classes)
Model   : ResNet-18 fine-tuned, weights saved with torch.save(model.state_dict())
Author  : Your name here
"""

import io
import os

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

# ==================================================================
# 1️⃣  MODEL ARCHITECTURE  — ResNet-18  (confirmed from training code)
#
#     WHY ResNet-18?
#     Your training code did:
#         model = models.resnet18(pretrained=True)
#         model.fc = nn.Linear(num_ftrs, num_classes)   # num_classes = 15
#     So we recreate the exact same skeleton here, then load your weights.
# ==================================================================

def build_model(num_classes: int):
    model = models.resnet18(weights=None)          # skeleton only, no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # 512 → 15
    return model


# ==================================================================
# 2️⃣  CLASS NAMES  — 15 PlantVillage classes in alphabetical order
#
#     WHY alphabetical?
#     torchvision.datasets.ImageFolder sorts folder names alphabetically
#     and assigns index 0, 1, 2 ... in that order.
#     Your model's output neuron 0 = index 0 = first name below.
#     Getting this order wrong means the app shows the WRONG disease!
# ==================================================================
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",          # index 0
    "Pepper__bell___healthy",                 # index 1
    "Potato___Early_blight",                  # index 2
    "Potato___Late_blight",                   # index 3
    "Potato___healthy",                       # index 4
    "Tomato___Bacterial_spot",                # index 5
    "Tomato___Early_blight",                  # index 6
    "Tomato___Late_blight",                   # index 7
    "Tomato___Leaf_Mold",                     # index 8
    "Tomato___Septoria_leaf_spot",            # index 9
    "Tomato___Spider_mites_Two_spotted_spider_mite",  # index 10
    "Tomato___Target_Spot",                   # index 11
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", # index 12
    "Tomato___Tomato_mosaic_virus",           # index 13
    "Tomato___healthy",                       # index 14
]

# ==================================================================
# 3️⃣  MODEL FILE PATH
#
#     Your training code saved the model with:
#         torch.save(model.state_dict(), '/kaggle/working/plant_disease_model.pth')
#     Download that file from Kaggle and place it next to app.py.
#     Then update the filename below to match exactly.
# ==================================================================
MODEL_PATH = "plant_disease_model.pth"

# ==================================================================
# 4️⃣  IMAGE SIZE & PREPROCESSING
#
#     Your training code used:
#         transforms.Resize(256)
#         transforms.CenterCrop(224)
#     We use the SAME pipeline here so the model sees images
#     in the exact same format it was trained on.
#     Using different preprocessing = wrong/garbage predictions!
# ==================================================================
preprocess = transforms.Compose([
    transforms.Resize(256),                          # resize shorter side to 256
    transforms.CenterCrop(224),                      # crop center 224×224
    transforms.ToTensor(),                           # convert to [0,1] float tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                  # ImageNet mean (same as training)
        std=[0.229, 0.224, 0.225]                    # ImageNet std  (same as training)
    ),
])

# ==================================================================
# 5️⃣  DEVICE — uses GPU automatically if available, else CPU
#     On free hosting (Render.com) there is no GPU so it uses CPU.
#     That's fine — ResNet-18 is fast enough on CPU for inference.
# ==================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Running on: {DEVICE}")

# ==================================================================
# 6️⃣  LOAD THE MODEL
#
#     We build the empty skeleton first, then pour your saved
#     weights into it with load_state_dict().
#     map_location=DEVICE ensures it loads on CPU even if you
#     saved it on a Kaggle GPU.
# ==================================================================
model = build_model(num_classes=len(CLASS_NAMES))

if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict):
        model.load_state_dict(state)   # normal case: state_dict saved
    else:
        model = state                  # full model object saved
    model = model.to(DEVICE)
    model.eval()   # ← CRITICAL: switches off dropout & batchnorm training behaviour
    print(f"✅  Model loaded from '{MODEL_PATH}'")
else:
    print(f"⚠️  '{MODEL_PATH}' not found — running in MOCK mode")
    print("    Download plant_disease_model.pth from Kaggle → place next to app.py → restart")
    model = None


# ==================================================================
# 7️⃣  DISEASE INFO DATABASE
#
#     Taken directly from your training notebook's remedies dict.
#     Added: label (human-friendly name), severity, description,
#     and prevention tip for each class.
#     Keys MUST match CLASS_NAMES exactly — one entry per class.
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
        "prevention": "Monitor regularly for pests. Maintain even watering and balanced fertilization to prevent stress-induced susceptibility."
    },

    "Potato___Early_blight": {
        "label": "Potato — Early Blight",
        "severity": "moderate",
        "description": "Caused by Alternaria solani fungus. Shows as dark brown, circular spots with concentric rings (like a target) on older lower leaves first. Leaves may yellow and drop.",
        "remedy": "Apply appropriate fungicide (chlorothalonil or mancozeb) at first sign of disease. Destroy volunteer solanaceous plants. Practice crop rotation. Reduce plant stress by fertilizing and watering adequately.",
        "prevention": "Rotate crops with non-host plants. Use certified healthy seed tubers. Avoid overhead irrigation. Remove and destroy infected crop debris."
    },

    "Potato___Late_blight": {
        "label": "Potato — Late Blight",
        "severity": "severe",
        "description": "Caused by Phytophthora infestans — the same pathogen that caused the Irish Famine. Dark, water-soaked lesions with white mold on leaf undersides. Spreads explosively in cool, wet weather.",
        "remedy": "Destroy infected tubers and volunteer plants immediately. Apply fungicide to hills at emergence. Water early in the day to reduce overnight leaf wetness. Plant resistant varieties. Apply protective fungicide when conditions favor disease.",
        "prevention": "Plant certified disease-free seed tubers. Use resistant varieties. Hill soil around plants to protect tubers. Monitor weather forecasts — apply fungicide before expected rain in high-risk periods."
    },

    "Potato___healthy": {
        "label": "Potato — Healthy 🌱",
        "severity": "none",
        "description": "Your potato plant appears completely healthy. No signs of blight or other diseases detected.",
        "remedy": "No treatment needed.",
        "prevention": "Monitor for pests and maintain even watering and fertilization to prevent stress."
    },

    "Tomato___Bacterial_spot": {
        "label": "Tomato — Bacterial Spot",
        "severity": "moderate",
        "description": "Caused by Xanthomonas bacteria. Small, dark, water-soaked spots on leaves, stems and fruit. Spots may have yellow halos. Severe infections cause leaf drop and unmarketable fruit.",
        "remedy": "Use certified seed and healthy transplants. Remove crop debris. Avoid sprinkler irrigation — water at base only. Rotate crops. Apply copper-based bactericide.",
        "prevention": "Use resistant varieties. Avoid working with plants when wet. Sanitize tools with 10% bleach solution. Maintain proper plant spacing."
    },

    "Tomato___Early_blight": {
        "label": "Tomato — Early Blight",
        "severity": "moderate",
        "description": "Caused by Alternaria solani. Dark brown spots with concentric rings appear on older lower leaves first, then spread upward. Causes significant defoliation and yield loss.",
        "remedy": "Apply fungicide (chlorothalonil or copper-based) at first sign. Destroy volunteer solanaceous plants. Practice crop rotation. Stake plants to improve air circulation.",
        "prevention": "Rotate crops. Mulch soil surface to prevent fungal spore splash. Remove and destroy infected lower leaves early. Avoid overhead watering."
    },

    "Tomato___Late_blight": {
        "label": "Tomato — Late Blight",
        "severity": "severe",
        "description": "Caused by Phytophthora infestans. Large, dark, greasy-looking lesions on leaves and stems with white sporulation on undersides. Fruits develop dark, firm rot. Can destroy entire crop within days.",
        "remedy": "Destroy infected plants immediately — do NOT compost. Apply mancozeb or chlorothalonil fungicide preventively. Water early to dry leaves before nightfall. Plant resistant varieties. Remove debris and rotate crops.",
        "prevention": "Plant resistant varieties. Ensure proper plant spacing for airflow. Monitor weather — apply protective fungicide before wet periods. Never leave infected material in the field."
    },

    "Tomato___Leaf_Mold": {
        "label": "Tomato — Leaf Mold",
        "severity": "moderate",
        "description": "Caused by Passalora fulva fungus. Pale greenish-yellow spots on upper leaf surface with olive-green to grayish-purple velvety mold on the underside. Common in humid greenhouse conditions.",
        "remedy": "Grow resistant varieties. Avoid leaf wetting and overhead watering. Ensure good air circulation with proper plant spacing. Remove and burn infected debris. Apply fungicide (mancozeb or copper) if severe.",
        "prevention": "Maintain humidity below 85% in greenhouses. Increase plant spacing. Remove lower infected leaves promptly. Avoid working with plants when wet."
    },

    "Tomato___Septoria_leaf_spot": {
        "label": "Tomato — Septoria Leaf Spot",
        "severity": "moderate",
        "description": "Caused by Septoria lycopersici fungus. Small, circular spots with dark brown margins and lighter grey centers, often with tiny black dots (pycnidia) inside. Starts on lower leaves and moves upward.",
        "remedy": "Plant disease-free material. Remove and destroy infected debris or plow deep. Avoid overhead irrigation. Stake plants for better airflow. Apply fungicide (chlorothalonil or copper) if needed.",
        "prevention": "Rotate crops — do not plant tomatoes in same spot for 2+ years. Mulch to prevent soil splash. Remove infected leaves immediately when spotted."
    },

    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "label": "Tomato — Spider Mites",
        "severity": "moderate",
        "description": "Infestation by Tetranychus urticae (two-spotted spider mite). Tiny mites cause stippling (small yellow dots) on leaves, fine webbing on undersides, and eventual leaf bronzing and drop. Thrives in hot, dry conditions.",
        "remedy": "Use miticides or insecticidal soaps. Introduce predatory mites (Phytoseiulus persimilis) for biological control. Maintain humidity and avoid dusty conditions. Remove and destroy heavily infested leaves.",
        "prevention": "Monitor regularly — mites multiply fast. Avoid water stress which weakens plants. Use reflective mulch to disorient mites. Avoid broad-spectrum insecticides that kill natural predators."
    },

    "Tomato___Target_Spot": {
        "label": "Tomato — Target Spot",
        "severity": "moderate",
        "description": "Caused by Corynespora cassiicola fungus. Brown lesions with concentric rings (target-like pattern) on leaves, stems and fruit. Causes defoliation and direct fruit damage.",
        "remedy": "Remove plant debris and burn. Avoid excess nitrogen fertilization. Apply suitable fungicides (azoxystrobin or chlorothalonil) if severe.",
        "prevention": "Improve air circulation with proper spacing and staking. Avoid overhead irrigation. Rotate crops. Remove infected leaves promptly."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "label": "Tomato — Yellow Leaf Curl Virus",
        "severity": "severe",
        "description": "Viral disease transmitted by silverleaf whiteflies (Bemisia tabaci). Causes upward curling and yellowing of leaves, stunted growth, and severe yield loss. Plants infected early may produce no fruit at all.",
        "remedy": "Control whiteflies (the virus vectors) with insecticides or yellow sticky traps. Use resistant tomato varieties. Remove and destroy infected plants immediately. Use reflective silver mulches to deter whiteflies.",
        "prevention": "Use certified virus-free transplants. Install insect-proof netting in nurseries. Monitor and control whitefly populations early. Plant resistant varieties in high-risk areas."
    },

    "Tomato___Tomato_mosaic_virus": {
        "label": "Tomato — Mosaic Virus",
        "severity": "severe",
        "description": "Caused by Tomato mosaic virus (ToMV). Symptoms include mottled light and dark green mosaic pattern on leaves, leaf distortion, stunting and reduced fruit set. Spreads through contact, tools and infected seed.",
        "remedy": "Use virus-free seeds and transplants. Sanitize all tools with 10% bleach or trisodium phosphate solution. Remove and destroy infected plants. Control aphids and weeds that harbour the virus. Rotate crops.",
        "prevention": "Use resistant varieties. Wash hands thoroughly before handling plants. Never use tobacco products near tomatoes (tobacco mosaic virus can infect tomatoes). Disinfect tools between plants."
    },

    "Tomato___healthy": {
        "label": "Tomato — Healthy 🌱",
        "severity": "none",
        "description": "Your tomato plant appears completely healthy. No signs of disease, virus or pest damage detected.",
        "remedy": "No treatment needed. Your plant is doing great!",
        "prevention": "Monitor regularly for pests. Maintain even watering and balanced NPK fertilization. Stake plants for good airflow."
    },
}


# ==================================================================
#  PREDICTION FUNCTION
#
#  This is the core logic that runs every time a user uploads a photo.
#  Step by step:
#    A) Open the uploaded image bytes with PIL
#    B) Apply the same preprocessing used during training
#    C) Run the model — get raw scores (logits) for all 15 classes
#    D) Convert scores to probabilities with softmax
#    E) Pick the class with the highest probability
#    F) Look up that class in DISEASE_INFO and return everything
# ==================================================================

def predict_disease(image_bytes: bytes) -> dict:

    # A: Open image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # B: Preprocess — same pipeline as training (Resize 256 → CenterCrop 224 → Normalize)
    tensor = preprocess(pil_img)       # shape: (3, 224, 224)
    tensor = tensor.unsqueeze(0)       # add batch dimension → (1, 3, 224, 224)
    tensor = tensor.to(DEVICE)

    # C & D: Run model and get probabilities
    if model is not None:
        with torch.no_grad():          # no gradient calculation needed at inference
            logits = model(tensor)     # raw output scores: shape (1, 15)
        probs      = torch.softmax(logits, dim=1)[0]   # probabilities: shape (15,)
        class_idx  = probs.argmax().item()             # index of highest probability
        confidence = float(probs[class_idx])           # confidence score 0.0 → 1.0
    else:
        # MOCK mode — only active when model file is missing
        class_idx  = np.random.randint(0, len(CLASS_NAMES))
        confidence = round(float(np.random.uniform(0.70, 0.99)), 2)

    # E & F: Map index → class name → disease info
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
#  FLASK ROUTES
# ==================================================================

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main website page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint — receives image upload, returns JSON prediction.
    The JavaScript in index.html calls this endpoint automatically
    when the user clicks 'Analyze Plant'.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image field found in the request"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file was selected"}), 400
        image_bytes = file.read()
        result = predict_disease(image_bytes)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ==================================================================
#  START SERVER
# ==================================================================

if __name__ == "__main__":
    print("🌿 PlantMD server starting...")
    print("📡 Open browser at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
