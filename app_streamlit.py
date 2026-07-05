import streamlit as st
from utils import load_models, predict_disease_then_skin_type
from PIL import Image
import io
import json
import os
import base64
import requests

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="DermaCare - Smart Dermatology Assistant",
    layout="centered",
    page_icon="🩺",
)

# ==========================================
# ASSET HELPERS
# ==========================================
def get_base64_image(path: str):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

# Update these paths to your real locations
logo_path = r"C:\Users\Admin\OneDrive\Desktop\Derma-Care\logo.png"
logo_white_path = r"C:\Users\Admin\OneDrive\Desktop\Derma-Care\logo_white.png"
bg_image_path = r"C:\Users\Admin\OneDrive\Desktop\Derma-Care\hospital_bg.jpg"

# Use white logo if present
if os.path.exists(logo_white_path):
    logo_path = logo_white_path

logo_base64 = get_base64_image(logo_path)
# bg_base64 is no longer needed for background

PRIMARY = "#0FB3A3"
ACCENT = "#2563EB"

# ==========================================
# GLOBAL CSS – CLEAN BACKGROUND (No image)
# ==========================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: #F8FAFC;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #1E2937;
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2.2rem;
        max-width: 1100px;
    }}

    /* Animations */
    @keyframes floatSoft {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-6px); }}
        100% {{ transform: translateY(0px); }}
    }}

    @keyframes fadeUp {{
        0% {{ opacity: 0; transform: translateY(14px); }}
        100% {{ opacity: 1; transform: translateY(0px); }}
    }}

    /* HEADER */
    .dc-hero {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 18px 20px;
        border-radius: 24px;
        background: white;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        animation: fadeUp 0.7s ease-out;
    }}

    .dc-hero-left {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}

    .dc-logo-wrap {{
        width: 68px;
        height: 68px;
        border-radius: 20px;
        background: radial-gradient(circle at 30% 20%, #F9FAFB, #0F172A);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 14px 32px rgba(15,23,42,0.85);
        animation: floatSoft 4.8s ease-in-out infinite;
    }}

    .dc-logo {{
        width: 54px;
        height: 54px;
        border-radius: 16px;
        object-fit: cover;
    }}

    .dc-title-text {{
        font-size: 30px;
        font-weight: 750;
        letter-spacing: 0.03em;
        color: #1E2937;
    }}

    .dc-subtitle-text {{
        font-size: 13px;
        color: #64748B;
        max-width: 360px;
    }}

    .dc-pill-row {{
        margin-top: 6px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}

    .dc-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 11px;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #475569;
    }}

    /* CARDS */
    .dc-card {{
        border-radius: 22px;
        background: white;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        padding: 18px 20px 20px 20px;
        animation: fadeUp 0.7s ease-out;
    }}

    .dc-section-title {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 4px;
        color: #1E2937;
    }}

    .dc-note-small {{
        font-size: 12px;
        color: #64748B;
        margin-bottom: 8px;
    }}

    .dc-suggestion {{
        padding: 8px 10px;
        border-radius: 10px;
        font-size: 13px;
        margin-top: 5px;
        background: linear-gradient(90deg, rgba(15, 179, 163, 0.08), rgba(37, 99, 235, 0.08));
        border: 1px solid rgba(15, 179, 163, 0.3);
        color: #1E2937;
    }}

    .dc-json-block .stCode {{
        border-radius: 16px !important;
        overflow: hidden;
    }}

    /* CHAT */
    .dc-chat-card {{
        border-radius: 22px;
        background: white;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
        padding: 16px 18px 20px 18px;
        margin-top: 14px;
        animation: fadeUp 0.8s ease-out;
    }}

    .dc-chat-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }}

    .dc-chat-left {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .dc-chat-avatar {{
        width: 30px;
        height: 30px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 30%, #4ADE80, #0EA5E9);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        color: #022c22;
    }}

    .dc-chat-title {{
        font-size: 14px;
        font-weight: 600;
        color: #1E2937;
    }}

    .dc-chat-subtitle {{
        font-size: 11px;
        color: #64748B;
    }}

    .dc-footer {{
        text-align: center;
        font-size: 11px;
        color: #64748B;
        margin-top: 18px;
        opacity: 0.85;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# HERO HEADER
# ==========================================
st.markdown(
    f"""
    <div class="dc-hero">
        <div class="dc-hero-left">
            <div class="dc-logo-wrap">
                {"<img src='data:image/png;base64," + logo_base64 + "' class='dc-logo'>" if logo_base64 else "🩺"}
            </div>
            <div>
                <div class="dc-title-text">DermaCare</div>
                <div class="dc-subtitle-text">
                    Hospital-grade AI assistant for preliminary skin screening and patient skin-care counseling.
                </div>
                <div class="dc-pill-row">
                    <div class="dc-pill">🔍 Image-based pre-screening</div>
                    <div class="dc-pill">🤖 Local AI chatbot (phi3:mini)</div>
                    <div class="dc-pill">🏥 OPD-friendly workflow</div>
                </div>
            </div>
        </div>
        <div class="dc-hero-right">
            <div class="dc-hero-tag">Smart Dermatology Support</div>
            <div>Designed for hospitals, clinics, and tele-dermatology teams.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# ==========================================
# MAIN LAYOUT – IMAGE + RESULTS
# ==========================================
left_col, right_col = st.columns([1, 1.3], vertical_alignment="top")

# LEFT – IMAGE UPLOAD
with left_col:
    st.markdown('<div class="dc-card">', unsafe_allow_html=True)
    st.markdown('<div class="dc-section-title">Patient Image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dc-note-small">Upload a clear, well-lit image of the affected skin area for AI analysis.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload skin image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT – RESULTS
with right_col:
    st.markdown('<div class="dc-card">', unsafe_allow_html=True)
    st.markdown('<div class="dc-section-title">AI Screening Result</div>', unsafe_allow_html=True)
    result_container = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# MODEL INFERENCE
# ==========================================
if uploaded:
    with st.spinner("Running dermatology screening model…"):
        model_disease, model_skin = load_models()
        result = predict_disease_then_skin_type(image, model_disease, model_skin)

    with result_container.container():
        st.markdown('<div class="dc-card">', unsafe_allow_html=True)

        if result.get("disease_found"):
            st.success(f"⚠️ Possible condition detected: **{result['disease_label']}**")
        else:
            st.info("✅ No obvious visible disease pattern detected in this image.")

        st.markdown("---")

        st.markdown(f"**Skin Type:** {result['skin_type_label']}")
        st.markdown(f"**Skin Type confidence:** {result['skin_type_confidence']:.2%}")

        if result.get("disease_found"):
            st.markdown(f"**Condition confidence:** {result['disease_confidence']:.2%}")
            if result.get("disease_note"):
                st.markdown("**Model note:**")
                st.write(result["disease_note"])

        if result.get("suggestions"):
            st.markdown("**Care suggestions:**")
            for s in result["suggestions"]:
                st.markdown(f"<div class='dc-suggestion'>• {s}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Full model output (JSON):**")
        st.markdown('<div class="dc-json-block">', unsafe_allow_html=True)
        st.code(json.dumps(result, indent=2), language="json")
        st.markdown("</div>", unsafe_allow_html=True)

        buf = io.BytesIO(json.dumps(result, indent=2).encode())
        st.download_button(
            "Download JSON report",
            data=buf,
            file_name="derma_result.json",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# AI CHAT ASSISTANT – FAST LOCAL COUNSELOR
# ==========================================
st.markdown("")
st.markdown('<div class="dc-chat-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="dc-chat-header">
        <div class="dc-chat-left">
            <div class="dc-chat-avatar">AI</div>
            <div>
                <div class="dc-chat-title">SkinCare Assistant</div>
                <div class="dc-chat-subtitle">General skin-care guidance · not a diagnosis</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

OLLAMA_MODEL = "phi3:mini"
OLLAMA_API_URL = "http://localhost:11434/api/chat"


def ask_ollama(history):
    system_message = {
        "role": "system",
        "content": (
            "You are a fast, concise, friendly skin-care assistant for a hospital. "
            "Give short, practical answers about daily routines, acne care, dry/oily skin, "
            "sun protection, and when to see a dermatologist. "
            "Never claim to give a confirmed diagnosis or prescribe medicines."
        ),
    }
    messages = [system_message] + history[-4:]

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=40,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
    except Exception as e:
        return (
            "⚠️ I couldn't reach the local AI model.\n\n"
            "Please check:\n"
            "• Ollama is installed and running\n"
            f"• Model `{OLLAMA_MODEL}` is pulled\n"
            "• Port 11434 is available\n\n"
            f"Technical details: {e}"
        )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Hi, I’m the DermaCare assistant. "
                "Ask me about routines for oily/dry skin, safe acne care, sunscreen usage, or general skin hygiene. "
                "For serious issues, always visit a dermatologist."
            ),
        }
    ]

# show history
for msg in st.session_state.chat_history:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a skin-care question…")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Preparing a quick response…"):
            reply_text = ask_ollama(st.session_state.chat_history)
            st.markdown(reply_text)
    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    """
    <div class="dc-footer">
        DermaCare is an assistive tool only and does not provide a medical diagnosis.
        Always consult a qualified dermatologist or healthcare professional for treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)