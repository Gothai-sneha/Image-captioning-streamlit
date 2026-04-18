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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotion Enriched Image Captioning")
st.title("Emotion Enriched Image Captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD SPACY MODEL
# =========================
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# =========================
# GOOGLE DRIVE MODEL DOWNLOAD
# =========================
ENCODER_FILE_ID = "1CYccQ7JxBCJL_unbXTgtwCe4dLwEENUb"
DECODER_FILE_ID = "1Sbu7VVU0kWH93l7z8-VCP8e4Y6f-IXYH"

if not os.path.exists("encoder.pth"):
    gdown.download(f"https://drive.google.com/uc?id={ENCODER_FILE_ID}", "encoder.pth", quiet=False)

if not os.path.exists("decoder.pth"):
    gdown.download(f"https://drive.google.com/uc?id={DECODER_FILE_ID}", "decoder.pth", quiet=False)

# =========================
# VOCAB CLASS
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
# ENCODER
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
# DECODER
# =========================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size + encoder_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Emotion head
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
    caption = caption.replace(" .", ".").replace("..", ".")
    caption = caption.rstrip(".")
    return caption

# =========================
# GRAMMAR REFINEMENT
# =========================
def refine_caption_grammar(caption):
    doc = nlp(caption)
    tokens = [token.text for token in doc if token.text not in ["<unk>", "<pad>"]]
    sentence = " ".join(tokens).capitalize()

    if not sentence.endswith("."):
        sentence += "."

    return sentence

# =========================
# EMOTION PREDICTION (MODEL)
# =========================
def predict_emotion(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)
        logits = decoder.emotion_fc(features)
        probs = torch.softmax(logits, dim=1)

        confidence, predicted = torch.max(probs, dim=1)

    return emotion_classes[predicted.item()], confidence.item()

# =========================
# EMOTION WORD INJECTION
# =========================
def emotion_word_injection(caption, emotion):
    words = caption.split()

    emotion_map = {
        "excited": {
            "run": "run excitedly",
            "running": "running excitedly",
            "jump": "jump energetically",
            "play": "play joyfully"
        },
        "happy": {
            "smile": "smile happily",
            "walk": "walk cheerfully",
            "play": "play happily"
        },
        "sad": {
            "sit": "sit quietly",
            "walk": "walk slowly"
        },
        "peaceful": {
            "stand": "stand calmly",
            "sit": "sit peacefully"
        }
    }

    if emotion not in emotion_map:
        return caption

    new_words = []
    for w in words:
        replaced = False
        for key in emotion_map[emotion]:
            if key in w:
                new_words.append(emotion_map[emotion][key])
                replaced = True
                break
        if not replaced:
            new_words.append(w)

    return " ".join(new_words)

# =========================
# ATTENTION-LIKE KEYWORDS
# =========================
def highlight_emotion_focus(caption):
    doc = nlp(caption)
    return [token.text for token in doc if token.pos_ in ["VERB", "ADV"]]

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

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

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
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        raw_caption = generate_caption(image)

        clean_cap = clean_caption(raw_caption)
        refined_cap = refine_caption_grammar(clean_cap)

        emotion, confidence = predict_emotion(image)

        if confidence > 0.6 and emotion != "neutral":
            refined_cap = emotion_word_injection(refined_cap, emotion)

        final_caption = refined_cap.rstrip(".") + "."

        focus_words = highlight_emotion_focus(final_caption)

        st.success("Generated Caption:")
        st.write(final_caption)

        st.info(f"Predicted Emotion: {emotion} (confidence: {confidence:.2f})")

        st.write("Key Action Words (Attention Focus):", focus_words)
