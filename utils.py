# utils.py — updated (skin-first with confidence-based disease trigger)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(__file__)

DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "disease_model.h5")
SKIN_TYPE_MODEL_PATH = os.path.join(BASE_DIR, "skin_type_model.h5")

# ====== Replace these with your real class names if available ======
DISEASE_LABELS = [
    "akiec", "basal cell carcinoma", "benign keratosis-like lesions", "dermatofibroma",
    "melanocytic nevi", "pyogenic granulomas and hemorrhage", "melanoma"
]
SKIN_TYPE_LABELS = ["dry", "oily", "normal"]
# ====================================================================

# Confidence thresholds
SKIN_TRIGGER_THRESHOLD = 0.1     # Run disease model only if skin prediction ≥ this
DISEASE_FOUND_THRESHOLD = 0.6    # Accept disease prediction only if ≥ this confidence

_model_cache = {}

def _find_model_path(path_candidate):
    if os.path.exists(path_candidate):
        return path_candidate
    alt = os.path.join(BASE_DIR, os.path.basename(path_candidate))
    return alt if os.path.exists(alt) else None

def load_models():
    """Load and cache both models."""
    if 'disease' in _model_cache and 'skin' in _model_cache:
        return _model_cache['disease'], _model_cache['skin']

    dpath = _find_model_path(DISEASE_MODEL_PATH)
    spath = _find_model_path(SKIN_TYPE_MODEL_PATH)

    if dpath is None:
        raise FileNotFoundError(f"Disease model not found at: {DISEASE_MODEL_PATH}")
    if spath is None:
        raise FileNotFoundError(f"Skin-type model not found at: {SKIN_TYPE_MODEL_PATH}")

    print("Loading disease model from:", dpath)
    print("Loading skin-type model from:", spath)
    _model_cache['disease'] = load_model(dpath, compile=False)
    _model_cache['skin'] = load_model(spath, compile=False)
    return _model_cache['disease'], _model_cache['skin']

# ---------------------- Preprocessing ----------------------
def _to_rgb(pil_img):
    return pil_img.convert('RGB') if pil_img.mode != 'RGB' else pil_img

def preprocess_image_for_shape(pil_img, model_input_shape):
    """Resize and normalize image to fit model input shape."""
    img = _to_rgb(pil_img)
    target_h, target_w, target_c = 224, 224, 3
    try:
        shape = list(model_input_shape)
        if len(shape) == 4:
            _, h, w, c = shape
            target_h = int(h) if h else 224
            target_w = int(w) if w else 224
            target_c = int(c) if c else 3
    except Exception:
        pass

    img_resized = img.resize((target_w, target_h))
    arr = np.asarray(img_resized).astype('float32') / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    if arr.shape[2] == 1 and target_c == 3:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------- Notes & Suggestions ----------------------
def disease_short_note(label):
    notes = {
        "akiec": "Precancerous rough patches caused by sun damage. Can sometimes progress to squamous cell carcinoma if untreated. Dermatologist evaluation and sun protection are advised.",
        "basal cell carcinoma": "A slow-growing skin cancer often caused by long-term sun exposure. Usually not life-threatening but should be removed by a dermatologist to prevent local damage.",
        "benign keratosis-like lesions": "Harmless non-cancerous growths that may look warty or scaly. Usually no treatment needed unless they become irritated or for cosmetic reasons.",
        "dermatofibroma": "A small, firm bump under the skin—often from minor injury or insect bite. Benign and typically requires no treatment unless painful or changing.",
        "melanocytic nevi": "Common moles formed by clusters of pigment cells. Usually harmless, but any mole that changes in color, shape, or size should be checked for melanoma.",
        "pyogenic granulomas and hemorrhage": "Red, bleeding bumps that grow quickly after injury or irritation. Benign but may need removal if they bleed or don’t heal.",
        "melanoma": "A serious form of skin cancer that can spread rapidly. Early detection is critical—seek immediate dermatological evaluation if suspected."
    }
    return notes.get(label, "Consult a dermatologist for accurate diagnosis.")

def generate_suggestions(skin_label):
    base = {
        "dry": [
            "Use a hydrating cleanser",
            "Apply moisturizer twice daily",
            "Use sunscreen daily"
        ],
        "oily": [
            "Use an oil-free foaming cleanser",
            "Use non-comedogenic products",
            "Avoid heavy creams"
        ],
        "normal": [
            "Maintain a balanced skincare routine",
            "Use a gentle cleanser and moisturizer",
            "Use sunscreen daily"
        ]
    }
    return base.get(skin_label, ["Maintain good skin hygiene", "Use sunscreen daily"])

# ---------------------- Main Prediction Logic ----------------------
def predict_skin_then_disease(pil_img, disease_model, skin_model):
    """
    Flow:
    1. Run skin model first.
    2. If skin confidence >= SKIN_TRIGGER_THRESHOLD, run disease model.
    3. Return combined result depending on disease confidence.
    """
    # 1️⃣ Predict skin type
    s_input_shape = getattr(skin_model, 'input_shape', None)
    x_skin = preprocess_image_for_shape(pil_img, s_input_shape)
    spreds = skin_model.predict(x_skin)
    sprob = spreds[0] if spreds.ndim >= 2 else np.ravel(spreds)
    s_idx = int(np.argmax(sprob))
    s_conf = float(sprob[s_idx])
    skin_label = SKIN_TYPE_LABELS[s_idx] if s_idx < len(SKIN_TYPE_LABELS) else f"label_{s_idx}"
    suggestions = generate_suggestions(skin_label)

    # Skip disease detection if skin prediction is uncertain
    if s_conf < SKIN_TRIGGER_THRESHOLD:
        return {
            "disease_checked": False,
            "reason": f"Skin model confidence too low ({s_conf:.2f})",
            "skin_type_label": skin_label,
            "skin_type_confidence": s_conf,
            "suggestions": suggestions
        }

    # 2️⃣ Predict disease only if skin confidence is sufficient
    d_input_shape = getattr(disease_model, 'input_shape', None)
    x_disease = preprocess_image_for_shape(pil_img, d_input_shape)
    dpreds = disease_model.predict(x_disease)
    dprobs = dpreds[0] if dpreds.ndim >= 2 else np.ravel(dpreds)
    d_idx = int(np.argmax(dprobs))
    d_conf = float(dprobs[d_idx])
    disease_label = DISEASE_LABELS[d_idx] if d_idx < len(DISEASE_LABELS) else f"label_{d_idx}"

    # 3️⃣ Decide output based on disease confidence
    if d_conf >= DISEASE_FOUND_THRESHOLD:
        note = disease_short_note(disease_label)
        return {
            "disease_checked": True,
            "disease_found": True,
            "disease_label": disease_label,
            "disease_confidence": d_conf,
            "disease_note": note,
            "skin_type_label": skin_label,
            "skin_type_confidence": s_conf
        }
    else:
        return {
            "disease_checked": True,
            "disease_found": False,
            "disease_label": None,
            "disease_confidence": d_conf,
            "skin_type_label": skin_label,
            "skin_type_confidence": s_conf,
            "suggestions": suggestions
        }

# Backward compatibility
def predict_disease_then_skin_type(pil_img, disease_model, skin_model):
    return predict_skin_then_disease(pil_img, disease_model, skin_model)
