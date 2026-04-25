import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Emotion Enriched Image Captioning")

# =========================
# 🎨 UI STYLING (ADDED)
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #667eea, #764ba2);
}

/* Title */
h1 {
    color: white;
    text-align: center;
    font-weight: bold;
}

/* Text */
p {
    color: #f1f1f1;
    font-size: 18px;
}

/* Upload box */
.stFileUploader > div {
    background-color: #ffffff20;
    border: 2px dashed #ffffff80;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff20;
    color: white;
    border-radius: 12px;
}

/* Button */
.stButton > button {
    background-color: #ff4b5c;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #ff758c;
}

.stButton > button:active {
    background-color: #c9184a;
    transform: scale(0.95);
}
</style>
""", unsafe_allow_html=True)

st.title("Emotion Enriched Image Captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD VOCAB
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

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(256, 512, len(vocab)).to(device)

    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device), strict=False)

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
def clean_caption(caption):
    words = caption.split()

    cleaned = []
    for word in words:
        if not cleaned or cleaned[-1] != word:
            cleaned.append(word)

    sentence = " ".join(cleaned)

    parts = sentence.split(" of ")
    if len(parts) > 2:
        sentence = " of ".join(parts[:2])

    return sentence.strip().lower()

# =========================
# EMOTION DETECTION
# =========================
def get_emotion_from_caption(caption):
    text = caption.lower()

    if any(w in text for w in ["smile", "laugh", "happy"]):
        return "happy"

    if any(w in text for w in ["run", "jump", "race"]):
        return "excited"

    if any(w in text for w in ["play", "child", "dog", "ball"]):
        return "playful"

    if any(w in text for w in ["sit", "bench", "lake"]):
        return "peaceful"

    if any(w in text for w in ["cry", "sad", "alone", "lonely"]):
        return "sad"

    if any(w in text for w in ["rest", "sleep", "lying", "exhausted", "tired"]):
        return "tired"

    if "group of people" in text:
        return "happy"

    if "people" in text and "standing" in text:
        return "happy"

    return "neutral"

# =========================
# ENRICH CAPTION
# =========================
def enrich_caption_with_emotion(caption, emotion):

    caption = caption.strip().lower()

    if len(caption.split()) < 3:
        return "An image showing something."

    sentence = caption

    if " are " not in sentence:
        sentence = sentence.replace(" standing", " is standing") \
                           .replace(" running", " is running") \
                           .replace(" playing", " is playing") \
                           .replace(" sitting", " is sitting") \
                           .replace(" lying", " is lying")

    if emotion == "tired":
        return sentence.strip().rstrip(".").capitalize() + " looking tired."

    emotion_map = {
        "happy": "happily",
        "excited": "excitedly",
        "playful": "playfully",
        "peaceful": "peacefully",
        "sad": "sadly",
        "neutral": ""
    }

    emotion_word = emotion_map.get(emotion, "")

    if emotion_word:
        if "standing" in sentence:
            sentence = sentence.replace("standing", f"standing {emotion_word}")
        elif "running" in sentence:
            sentence = sentence.replace("running", f"running {emotion_word}")
        elif "playing" in sentence:
            sentence = sentence.replace("playing", f"playing {emotion_word}")
        elif "sitting" in sentence:
            sentence = sentence.replace("sitting", f"sitting {emotion_word}")
        elif "lying" in sentence:
            sentence = sentence.replace("lying", f"lying {emotion_word}")
        else:
            sentence += f" {emotion_word}"

    return sentence.strip().rstrip(".").capitalize() + "."

# =========================
# BEAM SEARCH (UNCHANGED)
# =========================
def generate_caption(image, encoder, decoder, beam_width=3, max_len=20):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)
        sequences = [[[], 0.0, None]]

        for _ in range(max_len):
            all_candidates = []

            for seq, score, hidden in sequences:

                if len(seq) > 0 and seq[-1] == vocab.stoi["<EOS>"]:
                    all_candidates.append([seq, score, hidden])
                    continue

                word = torch.tensor([
                    [vocab.stoi["<SOS>"] if len(seq) == 0 else seq[-1]]
                ]).to(device)

                emb = decoder.embedding(word)
                inp = torch.cat((emb, features.unsqueeze(1)), dim=2)

                output, new_hidden = decoder.lstm(inp, hidden)
                output = decoder.fc(output.squeeze(1))

                log_probs = torch.log_softmax(output, dim=1)
                topk = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    idx = topk.indices[0][i].item()
                    prob = topk.values[0][i].item()
                    all_candidates.append([seq + [idx], score + prob, new_hidden])

            sequences = sorted(
                all_candidates,
                key=lambda x: x[1] / len(x[0]),
                reverse=True
            )[:beam_width]

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

# 📁 Upload Title (ADDED)
st.markdown('<h3 style="color:white;">📁 Upload Image</h3>', unsafe_allow_html=True)

file = st.file_uploader("", ["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):

        raw_caption = generate_caption(img, encoder, decoder)
        caption = clean_caption(raw_caption)
        emotion = get_emotion_from_caption(caption)
        final_caption = enrich_caption_with_emotion(caption, emotion)

        emoji_map = {
            "happy": "😊",
            "sad": "😢",
            "excited": "🤩",
            "playful": "😄",
            "peaceful": "😌",
            "tired": "😴",
            "neutral": "😐"
        }

        emoji = emoji_map.get(emotion, "")

        # 🎨 OUTPUT CARD (REPLACED SUCCESS/INFO)
       st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.12);
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
">

    <h2 style="
        color:#ffd54f;
        font-weight:bold;
        margin-bottom:15px;
    ">
        Emotion Enriched Caption
    </h2>

    <div style="
        background:white;
        padding:15px;
        border-radius:10px;
        margin-bottom:15px;
    ">
        <span style="
            color:black;
            font-size:18px;
            font-weight:bold;
        ">
            {final_caption}
        </span>
    </div>

    <div style="
        background: rgba(0,255,204,0.15);
        padding:10px;
        border-radius:10px;
    ">
        <span style="
            color:#00ffcc;
            font-size:18px;
            font-weight:bold;
        ">
            Predicted Emotion: {emotion} {emoji}
        </span>
    </div>

</div>
""", unsafe_allow_html=True)
