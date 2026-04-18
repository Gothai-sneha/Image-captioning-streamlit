
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotion Enriched Image Captioning")
st.title("Emotion Enriched Image Captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# VOCAB CLASS (FIX FOR PICKLE ERROR)
# =========================
class Vocabulary:
    def __init__(self):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

# =========================
# LOAD VOCAB
# =========================
# =========================
# SAFE VOCAB LOAD (FINAL FIX)
# =========================
class Vocabulary:
    def __init__(self):
        self.itos = {}
        self.stoi = {}

    def __len__(self):
        return len(self.itos)

def load_vocab():
    with open("vocab.pkl", "rb") as f:
        data = pickle.load(f)

    vocab = Vocabulary()

    # Case 1: saved as full object
    if hasattr(data, "__dict__"):
        vocab.__dict__.update(data.__dict__)

    # Case 2: saved as dict
    elif isinstance(data, dict):
        vocab.__dict__.update(data)

    return vocab

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
        resnet = models.resnet50(pretrained=True)
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

if not os.path.exists("encoder.pth"):
    st.error("encoder.pth NOT FOUND")
if not os.path.exists("decoder.pth"):
    st.error("decoder.pth NOT FOUND")

encoder.load_state_dict(torch.load("encoder.pth", map_location=device), strict=False)
decoder.load_state_dict(torch.load("decoder.pth", map_location=device), strict=False)

encoder.eval()
decoder.eval()

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
# REFINE CAPTION
# =========================
def refine_caption(c):
    words = [w for w in c.split() if w not in ["<unk>", "<pad>", "<SOS>"]]
    sentence = " ".join(words).strip()

    if " is " not in sentence:
        sentence = sentence.replace("a ", "a person is ", 1)

    sentence = sentence.capitalize()

    if not sentence.endswith("."):
        sentence += "."

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
# EMOTION REFINEMENT
# =========================
def refine_emotion(caption, model_emotion, confidence):
    text = caption.lower()

    if confidence > 0.5:
        return model_emotion

    if any(w in text for w in ["jump", "run", "play", "trick"]):
        return "excited"
    elif any(w in text for w in ["smile", "laugh"]):
        return "happy"
    elif any(w in text for w in ["sit", "lake", "calm"]):
        return "peaceful"
    elif any(w in text for w in ["alone", "cry"]):
        return "sad"

    return "neutral"

# =========================
# SIMPLE EMOTION ADD
# =========================
def add_emotion_simple(caption, emotion):
    caption = caption.rstrip(".")

    if emotion == "excited":
        return caption + " with excitement."
    elif emotion == "happy":
        return caption + " happily."
    elif emotion == "peaceful":
        return caption + " peacefully."
    elif emotion == "sad":
        return caption + " sadly."
    else:
        return caption + "."

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

            if len(seq) > 0 and seq[-1] == vocab.stoi["<EOS>"]:
                all_candidates.append((seq, score, hidden))
                continue

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
                all_candidates.append([seq+[idx], score - prob, new_hidden])

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    final_seq = sequences[0][0]

    words = []
    for idx in final_seq:
        word = vocab.itos.get(idx, "")
        if word == "<EOS>":
            break
        if word not in ["<SOS>", "<PAD>"]:
            words.append(word)

    return " ".join(words)

# =========================
# STREAMLIT UI
# =========================
file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):
        raw = generate_caption(img)
        refined = refine_caption(raw)

        model_emotion, conf = predict_emotion(img)
        emotion = refine_emotion(refined, model_emotion, conf)

        final = add_emotion_simple(refined, emotion)

        st.success("Generated Caption:")
        st.write(final)

        st.info(f"Emotion: {emotion} (confidence: {conf:.2f})")
```
