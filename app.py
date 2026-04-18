import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import pickle
import os
import gdown

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotion Enriched Image Captioning")
st.title("Emotion Enriched Image Captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# GOOGLE DRIVE MODEL DOWNLOAD
# =========================
ENCODER_FILE_ID = "1CYccQ7JxBCJL_unbXTgtwCe4dLwEENUb"
DECODER_FILE_ID = "1Sbu7VVU0kWH93l7z8-VCP8e4Y6f-IXYH"

def download_models():
    if not os.path.exists("encoder.pth"):
        with st.spinner("Downloading encoder model..."):
            gdown.download(
                f"https://drive.google.com/uc?id={ENCODER_FILE_ID}",
                "encoder.pth",
                quiet=False
            )

    if not os.path.exists("decoder.pth"):
        with st.spinner("Downloading decoder model..."):
            gdown.download(
                f"https://drive.google.com/uc?id={DECODER_FILE_ID}",
                "decoder.pth",
                quiet=False
            )

# =========================
# LOAD VOCAB
# =========================
@st.cache_resource
def load_vocab():
    with open("vocab.pkl", "rb") as f:
        return pickle.load(f)

vocab = load_vocab()

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
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource
def load_models():
    download_models()

    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(256, 512, len(vocab)).to(device)

    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

encoder, decoder = load_models()

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
# CLEAN CAPTION
# =========================
def refine_caption(c):
    c = c.strip().replace(" .", ".").replace("..", ".")
    words = [w for w in c.split() if w not in ["<unk>", "<pad>"]]
    sentence = " ".join(words).capitalize()
    return sentence

# =========================
# EMOTION PREDICTION
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
# HYBRID EMOTION REFINEMENT
# =========================
def refine_emotion(caption, model_emotion, confidence):
    text = caption.lower()

    if confidence > 0.6:
        return model_emotion

    if any(w in text for w in ["jump", "running", "play"]):
        return "excited"
    elif any(w in text for w in ["smile", "laugh"]):
        return "happy"
    elif any(w in text for w in ["sit", "calm"]):
        return "peaceful"
    elif any(w in text for w in ["alone", "cry"]):
        return "sad"

    return "neutral"

# =========================
# EMOTION INJECTION
# =========================
def inject_emotion(caption, emotion):
    if emotion == "excited":
        return caption.replace("jumping", "jumping excitedly") \
                      .replace("running", "running excitedly")
    elif emotion == "happy":
        return caption.replace("smiling", "smiling happily")
    elif emotion == "peaceful":
        return caption.replace("sitting", "sitting peacefully")
    elif emotion == "sad":
        return caption.replace("sitting", "sitting quietly")
    return caption

# =========================
# BEAM SEARCH (FIXED)
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

                output, new_hidden = decoder.lstm(inp, hidden)
                output = decoder.fc(output.squeeze(1))

                log_probs = torch.log_softmax(output, dim=1)
                topk = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    idx = topk.indices[0][i].item()
                    prob = topk.values[0][i].item()

                    h = (new_hidden[0].clone(), new_hidden[1].clone())
                    all_candidates.append([seq + [idx], score - prob, h])

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
# STREAMLIT UI
# =========================
file = st.file_uploader("Upload Image", ["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            raw = generate_caption(img)
            refined = refine_caption(raw)

            model_emotion, conf = predict_emotion(img)
            emotion = refine_emotion(refined, model_emotion, conf)

            if emotion != "neutral":
                refined = inject_emotion(refined, emotion)

            final = refined.strip()
            if not final.endswith("."):
                final += "."

        st.success("Generated Caption:")
        st.write(final)

        st.info(f"Emotion: {emotion} (confidence: {conf:.2f})")
