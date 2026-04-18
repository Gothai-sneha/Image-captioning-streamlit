import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Advanced Emotion Image Captioning")
st.title("Advanced Emotion-Aware Image Captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD BLIP MODEL
# =========================
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model

processor, model = load_model()

# =========================
# SIMPLE POS-STYLE VERB DETECTOR
# =========================
def ensure_action(caption):
    verbs = ["is", "are", "running", "playing", "walking", "jumping", "sitting"]

    if not any(v in caption.lower() for v in verbs):
        if "soccer" in caption or "football" in caption:
            return caption + " while playing with a ball"
        if "dog" in caption:
            return caption + " while playing"
        if "person" in caption or "man" in caption or "woman" in caption:
            return caption + " doing an activity"

    return caption

# =========================
# EMOTION DETECTION (LIGHTWEIGHT)
# =========================
def detect_emotion(caption):
    text = caption.lower()

    if any(w in text for w in ["run", "jump", "play", "catch"]):
        return "excited", 0.7
    elif any(w in text for w in ["smile", "laugh"]):
        return "happy", 0.75
    elif any(w in text for w in ["sit", "calm", "lake"]):
        return "peaceful", 0.65
    elif any(w in text for w in ["alone", "cry"]):
        return "sad", 0.6

    return "neutral", 0.5

# =========================
# EMOTION INJECTION
# =========================
def inject_emotion(caption, emotion):
    if emotion == "excited":
        return caption.replace("running", "running excitedly") \
                      .replace("playing", "playing energetically")
    elif emotion == "happy":
        return caption.replace("smiling", "smiling happily")
    elif emotion == "peaceful":
        return caption.replace("sitting", "sitting peacefully")
    elif emotion == "sad":
        return caption.replace("sitting", "sitting quietly")

    return caption

# =========================
# GENERATE CAPTION USING BLIP
# =========================
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.capitalize() + "."

# =========================
# UI
# =========================
file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):
        caption = generate_caption(img)

        # Ensure action (POS-style)
        caption = ensure_action(caption)

        # Emotion detection
        emotion, conf = detect_emotion(caption)

        # Inject emotion if meaningful
        if emotion != "neutral" and conf > 0.6:
            caption = inject_emotion(caption, emotion)

        st.success("Generated Caption:")
        st.write(caption)

        st.info(f"Emotion: {emotion} (confidence: {conf:.2f})")
