import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import gdown
import os
import spacy
import subprocess

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotion Enriched Image Captioning")
st.title("Emotion Enriched Image Captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD SPACY MODEL (AUTO FIX)
# =========================
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# =========================
# DOWNLOAD MODELS
# =========================
ENCODER_FILE_ID = "1CYccQ7JxBCJL_unbXTgtwCe4dLwEENUb"
DECODER_FILE_ID = "1Sbu7VVU0kWH93l7z8-VCP8e4Y6f-IXYH"

if not os.path.exists("encoder.pth"):
    gdown.download(f"https://drive.google.com/uc?id={ENCODER_FILE_ID}", "encoder.pth", quiet=False)

if not os.path.exists("decoder.pth"):
    gdown.download(f"https://drive.google.com/uc?id={DECODER_FILE_ID}", "decoder.pth", quiet=False)

# =========================
# VOCAB
# =========================
class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

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
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images):
        features = self.resnet(images)
        return features.view(features.size(0), -1)

# =========================
# DECODER
# =========================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.emotion_fc = nn.Linear(2048, len(emotion_classes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, captions):
        features = encoder_out.unsqueeze(1).repeat(1, captions.size(1), 1)
        embeddings = self.embedding(captions)
        inputs = torch.cat((embeddings, features), dim=2)
        lstm_out, _ = self.lstm(inputs)
        return self.fc(self.dropout(lstm_out)), self.emotion_fc(encoder_out)

# =========================
# LOAD MODEL
# =========================
encoder = EncoderCNN().to(device)
decoder = DecoderRNN(256, 512, len(vocab)).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

encoder.eval()
decoder.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# CLEAN
# =========================
def clean_caption(c):
    return c.strip().replace(" .", ".").replace("..", ".").rstrip(".")

# =========================
# GRAMMAR FIX
# =========================
def refine_caption(caption):
    doc = nlp(caption)
    words = [t.text for t in doc if t.text not in ["<unk>", "<pad>"]]
    sent = " ".join(words).capitalize()
    return sent if sent.endswith(".") else sent + "."

# =========================
# EMOTION PREDICT
# =========================
def predict_emotion(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        f = encoder(image)
        logits = decoder.emotion_fc(f)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
    return emotion_classes[pred.item()], conf.item()

# =========================
# SMART EMOTION INJECTION
# =========================
def inject_emotion(caption, emotion):
    words = caption.split()

    mapping = {
        "excited": {"running": "running excitedly", "run": "run excitedly"},
        "happy": {"smiling": "smiling happily"},
        "sad": {"sitting": "sitting quietly"},
        "peaceful": {"sitting": "sitting peacefully"}
    }

    if emotion not in mapping:
        return caption

    new = []
    for w in words:
        replaced = False
        for k in mapping[emotion]:
            if k in w.lower():
                new.append(mapping[emotion][k])
                replaced = True
                break
        if not replaced:
            new.append(w)

    return " ".join(new)

# =========================
# ATTENTION WORDS
# =========================
def get_focus_words(caption):
    doc = nlp(caption)
    return [t.text for t in doc if t.pos_ in ["VERB", "ADV"]]

# =========================
# BEAM SEARCH
# =========================
def generate_caption(image, beam_width=3, max_len=20):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image)

    sequences = [[[], 0.0, None]]

    for _ in range(max_len):
        all_candidates = []
        for seq, score, hidden in sequences:
            word = torch.tensor([[vocab.stoi["<SOS>"] if len(seq)==0 else seq[-1]]]).to(device)
            emb = decoder.embedding(word)
            inp = torch.cat((emb, features.unsqueeze(1)), dim=2)
            out, hidden = decoder.lstm(inp, hidden)
            out = decoder.fc(out.squeeze(1))

            log_probs = torch.log_softmax(out, dim=1)
            topk = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                idx = topk.indices[0][i].item()
                prob = topk.values[0][i].item()
                all_candidates.append([seq+[idx], score-prob, hidden])

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    words = []
    for idx in sequences[0][0]:
        word = vocab.itos.get(idx, "")
        if word == "<EOS>":
            break
        if word not in ["<SOS>", "<PAD>"]:
            words.append(word)

    return " ".join(words)

# =========================
# UI
# =========================
file = st.file_uploader("Upload Image", ["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):
        raw = generate_caption(img)
        clean = clean_caption(raw)
        refined = refine_caption(clean)

        emotion, conf = predict_emotion(img)

        if conf > 0.6 and emotion != "neutral":
            refined = inject_emotion(refined, emotion)

        final = refined.rstrip(".") + "."
        focus = get_focus_words(final)

        st.success("Generated Caption:")
        st.write(final)

        st.info(f"Emotion: {emotion} (confidence: {conf:.2f})")
        st.write("Key Words:", focus)
