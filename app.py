import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import gdown
import os

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

if not os.path.exists("encoder.pth"):
    gdown.download(
        f"https://drive.google.com/uc?id={ENCODER_FILE_ID}",
        "encoder.pth",
        quiet=False
    )

if not os.path.exists("decoder.pth"):
    gdown.download(
        f"https://drive.google.com/uc?id={DECODER_FILE_ID}",
        "decoder.pth",
        quiet=False
    )

# =========================
# VOCAB CLASS (FOR PICKLE)
# =========================
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

# =========================
# LOAD VOCAB
# =========================
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# =========================
# EMOTION LABELS
# =========================
emotion_classes = ["happy", "sad", "peaceful", "excited", "neutral"]

# =========================
# ENCODER (NOTEBOOK MATCH)
# =========================
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features

# =========================
# DECODER (NOTEBOOK MATCH)
# =========================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            embed_size + encoder_dim,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.emotion_fc = nn.Linear(encoder_dim, len(emotion_classes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, captions):
        features = encoder_out.unsqueeze(1).repeat(1, captions.size(1), 1)
        embeddings = self.embedding(captions)
        inputs = torch.cat((embeddings, features), dim=2)
        lstm_out, _ = self.lstm(inputs)
        predictions = self.fc(self.dropout(lstm_out))
        emotion_pred = self.emotion_fc(encoder_out)
        return predictions, emotion_pred

# =========================
# LOAD MODEL
# =========================
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

encoder = EncoderCNN().to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))

encoder.eval()
decoder.eval()

# =========================
# IMAGE PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# CLEAN CAPTION
# =========================
def clean_caption(caption):
    caption = caption.strip()
    caption = caption.replace(" .", ".")
    caption = caption.replace("..", ".")
    if len(caption) == 0:
        return "No caption generated."
    return caption.capitalize() + "."

# =========================
# EMOTION POST-PROCESS
# =========================
def get_emotion_from_caption(caption):
    text = caption.lower()

    if any(word in text for word in ["smile", "laugh", "happy"]):
        return "happy"
    elif any(word in text for word in ["run", "jump", "play"]):
        return "excited"
    elif any(word in text for word in ["sit", "lake", "tree", "peace"]):
        return "peaceful"
    elif any(word in text for word in ["cry", "sad"]):
        return "sad"
    else:
        return "neutral"

# =========================
# BEAM SEARCH CAPTION
# =========================
def generate_caption(image, beam_width=3, max_len=20):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)

    sequences = [[[], 0.0, None]]

    for _ in range(max_len):
        all_candidates = []

        for seq, score, hidden in sequences:
            if len(seq) == 0:
                word = torch.tensor([[vocab.stoi["<SOS>"]]]).to(device)
            else:
                word = torch.tensor([[seq[-1]]]).to(device)

            embedding = decoder.embedding(word)
            feature_step = features.unsqueeze(1)

            lstm_input = torch.cat((embedding, feature_step), dim=2)

            output, hidden = decoder.lstm(lstm_input, hidden)
            output = decoder.fc(output.squeeze(1))

            log_probs = torch.log_softmax(output, dim=1)
            topk = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                word_idx = topk.indices[0][i].item()
                prob = topk.values[0][i].item()
                candidate = [seq + [word_idx], score - prob, hidden]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda x: x[1])
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]

    words = []
    for idx in best_seq:
        word = vocab.itos.get(idx, "")
        if word == "<EOS>":
            break
        if word not in ["<SOS>", "<PAD>"]:
            words.append(word)

    return " ".join(words)

# =========================
# STREAMLIT UI
# =========================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        raw_caption = generate_caption(image)
        final_caption = clean_caption(raw_caption)
        emotion = get_emotion_from_caption(final_caption)

        st.success(f"Caption: {final_caption}")
        st.info(f"Predicted Emotion: {emotion}")
