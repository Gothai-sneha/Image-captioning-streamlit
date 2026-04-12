import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
import re
from torchvision import models

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Emotion Enriched Image Captioning", layout="centered")
st.title("🖼️ Emotion-Enriched Image Caption Generator")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Encoder
# -----------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features.unsqueeze(1)

# -----------------------------
# Decoder with Attention
# -----------------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features, embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20):
        sampled_ids = []
        inputs = features

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)

        return sampled_ids

# -----------------------------
# Load vocabulary
# -----------------------------
with open("word_map.pkl", "rb") as f:
    word_map = pickle.load(f)

idx2word = {v: k for k, v in word_map.items()}

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, len(word_map)).to(device)

    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

encoder, decoder = load_models()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# -----------------------------
# Generate caption
# -----------------------------
def generate_caption(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)
        sampled_ids = decoder.sample(features)

    words = []
    for word_id in sampled_ids:
        word = idx2word.get(word_id, "<unk>")
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            words.append(word)

    caption = " ".join(words)
    caption = caption.capitalize()
    caption = re.sub(r"\s+", " ", caption).strip()

    return caption

# -----------------------------
# Emotion prediction
# -----------------------------
def predict_emotion(caption):
    caption = caption.lower()

    if any(word in caption for word in ["running", "jumping", "playing", "flying"]):
        return "excited"
    elif any(word in caption for word in ["sleeping", "resting", "sitting"]):
        return "peaceful"
    elif any(word in caption for word in ["crying", "alone", "dark"]):
        return "sad"
    else:
        return "happy"

# -----------------------------
# Human-like emotion integration
# -----------------------------
def enrich_caption(caption, emotion):
    caption = caption.rstrip(".")

    emotion_phrases = {
        "happy": "with a joyful vibe",
        "sad": "in a sorrowful moment",
        "excited": "in an excited moment",
        "peaceful": "in a peaceful moment"
    }

    phrase = emotion_phrases.get(emotion, f"in a {emotion} moment")
    return f"{caption} {phrase}."

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
        emotion = predict_emotion(caption)
        final_caption = enrich_caption(caption, emotion)

    st.subheader("✨ Generated Output")
    st.success(final_caption)

    st.subheader("🎭 Predicted Emotion")
    st.info(emotion.capitalize())
