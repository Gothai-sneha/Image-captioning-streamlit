import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from collections import Counter
import re
import os

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Emotion Enriched Captioning", layout="centered")
st.title("🖼️ Emotion Enriched Image Caption Generator")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Build Vocabulary from CSV
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "captions.csv")

def build_vocab(csv_file):
    df = pd.read_csv(csv_file)
    counter = Counter()

    for caption in df["caption"]:
        words = re.findall(r"\w+", str(caption).lower())
        counter.update(words)

    word_map = {
        "<pad>": 0,
        "<start>": 1,
        "<end>": 2,
        "<unk>": 3
    }

    idx = 4
    for word in counter.keys():
        word_map[word] = idx
        idx += 1

    return word_map

word_map = build_vocab(CSV_PATH)
idx2word = {v: k for k, v in word_map.items()}

# -----------------------------
# Encoder
# -----------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features.unsqueeze(1)

# -----------------------------
# Decoder
# -----------------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

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
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, len(word_map)).to(device)

    encoder.load_state_dict(torch.load(os.path.join(BASE_DIR, "encoder.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(BASE_DIR, "decoder.pth"), map_location=device))

    encoder.eval()
    decoder.eval()
    return encoder, decoder

encoder, decoder = load_models()

# -----------------------------
# Image Transform
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
# Generate Caption
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
    return caption

# -----------------------------
# Predict Emotion
# -----------------------------
def predict_emotion(caption):
    caption = caption.lower()

    if any(word in caption for word in ["running", "jumping", "playing"]):
        return "excited"
    elif any(word in caption for word in ["sleeping", "resting", "sitting"]):
        return "peaceful"
    elif any(word in caption for word in ["crying", "alone"]):
        return "sad"
    else:
        return "happy"

# -----------------------------
# Streamlit Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
        emotion = predict_emotion(caption)

        # EXACT COLAB FORMAT
        emotion_caption = f"{emotion.capitalize()} {caption} ."

    st.subheader("Generated Output")
    st.write(f'**Emotion Enriched Caption:** "{emotion_caption}"')
    st.write(f"**Predicted Emotion:** {emotion}")
