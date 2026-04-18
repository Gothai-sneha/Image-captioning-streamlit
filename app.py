import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import pickle
import os
import traceback

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotion Enriched Image Captioning")
st.title("Emotion Enriched Image Captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

BASE_DIR = os.path.dirname(__file__)

# =========================
# LOAD VOCAB (SAFE)
# =========================
def load_pickle(path):
    with open(os.path.join(BASE_DIR, path), "rb") as f:
        return pickle.load(f)

raw_vocab = load_pickle("vocab.pkl")

# Handle different vocab formats
if isinstance(raw_vocab, dict):
    stoi = raw_vocab.get("stoi")
    itos = raw_vocab.get("itos")
else:
    stoi = getattr(raw_vocab, "stoi", None)
    itos = getattr(raw_vocab, "itos", None)

if stoi is None or itos is None:
    st.error("Invalid vocab format. Missing stoi/itos.")
    st.stop()

class VocabWrap:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

vocab = VocabWrap(stoi, itos)

# =========================
# EMOTION LABELS
# =========================
emotion_classes = ["happy", "sad", "peaceful", "excited", "neutral"]

# =========================
# ENCODER
# =========================
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (B, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (B, 49, 2048)
        return features

# =========================
# ATTENTION
# =========================
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(hidden).unsqueeze(1)
        energy = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(energy)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

# =========================
# DECODER
# =========================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048):
        super().__init__()
        self.attention = Attention(encoder_dim, hidden_size, 256)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.emotion_fc = nn.Linear(encoder_dim, len(emotion_classes))
        self.dropout = nn.Dropout(0.5)

    def forward_step(self, word, encoder_out, h, c):
        emb = self.embedding(word)
        context, _ = self.attention(encoder_out, h)
        lstm_input = torch.cat([emb.squeeze(1), context], dim=1)
        h, c = self.lstm(lstm_input, (h, c))
        out = self.fc(self.dropout(h))
        return out, h, c

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(256, 512, len(vocab.stoi)).to(device)

    encoder.load_state_dict(torch.load(os.path.join(BASE_DIR, "encoder.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(BASE_DIR, "decoder.pth"), map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

try:
    encoder, decoder = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.text(traceback.format_exc())
    st.stop()

# =========================
# IMAGE PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# CAPTION CLEANING
# =========================
def refine_caption(c):
    words = [w for w in c.split() if w not in ["<unk>", "<pad>"]]
    sentence = " ".join(words).capitalize()
    return sentence + "."

# =========================
# EMOTION PREDICTION
# =========================
def predict_emotion(features):
    mean_feat = features.mean(dim=1)
    logits = decoder.emotion_fc(mean_feat)
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, 1)
    return emotion_classes[pred.item()], conf.item()

# =========================
# CAPTION GENERATION
# =========================
def generate_caption(image, max_len=20):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_out = encoder(image)

    h = torch.zeros(1, 512).to(device)
    c = torch.zeros(1, 512).to(device)

    word = torch.tensor([[vocab.stoi["<SOS>"]]]).to(device)

    result = []

    for _ in range(max_len):
        output, h, c = decoder.forward_step(word, encoder_out, h, c)
        predicted = output.argmax(1)

        token = vocab.itos.get(predicted.item(), "")

        if token == "<EOS>":
            break

        if token not in ["<SOS>", "<PAD>"]:
            result.append(token)

        word = predicted.unsqueeze(1)

    return " ".join(result), encoder_out

# =========================
# STREAMLIT UI
# =========================
file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):
        try:
            with st.spinner("Generating caption..."):

                raw, features = generate_caption(img)
                caption = refine_caption(raw)

                emotion, conf = predict_emotion(features)

                st.success("Generated Caption:")
                st.write(caption)

                st.info(f"Emotion: {emotion} (confidence: {conf:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")
            st.text(traceback.format_exc())
