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
# CLEAN + STRUCTURE CAPTION
# =========================
def refine_caption(caption):
    caption = caption.lower()

    words = [w for w in caption.split() if w not in ["<unk>", "<pad>", "<sos>"]]

    if not words:
        return "An image."

    sentence = " ".join(words)

    # Add proper grammar (is/are)
    if sentence.startswith(("a ", "the ", "one ")):
        sentence = sentence.replace(" running", " is running")
        sentence = sentence.replace(" playing", " is playing")
        sentence = sentence.replace(" sitting", " is sitting")
        sentence = sentence.replace(" standing", " is standing")

    if sentence.startswith(("two ", "three ", "many ")):
        sentence = sentence.replace(" running", " are running")
        sentence = sentence.replace(" playing", " are playing")
        sentence = sentence.replace(" sitting", " are sitting")

    # Remove duplicate words
    cleaned = []
    for w in sentence.split():
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)

    sentence = " ".join(cleaned).strip()

    if len(sentence.split()) < 3:
        sentence = "An image showing something"

    return sentence.capitalize() + "."

# =========================
# EMOTION DETECTION
# =========================
def detect_emotion(caption):
    text = caption.lower()

    if "smiling" in text or "laughing" in text:
        return "happy"
    elif "running" in text or "playing" in text or "trick" in text:
        return "excited"
    elif "sitting" in text and ("lake" in text or "bench" in text):
        return "peaceful"
    elif "alone" in text or "crying" in text:
        return "sad"

    return None

# =========================
# INSERT EMOTION INTO SENTENCE
# =========================
def inject_emotion(caption, emotion):
    if emotion is None:
        return caption

    caption = caption.rstrip(".")

    if " is running" in caption:
        caption = caption.replace(" is running", f" is running {emotion}ly")
    elif " are running" in caption:
        caption = caption.replace(" are running", f" are running {emotion}ly")
    elif " is playing" in caption:
        caption = caption.replace(" is playing", f" is playing {emotion}ly")
    elif " are playing" in caption:
        caption = caption.replace(" are playing", f" are playing {emotion}ly")
    else:
        caption = caption + f" {emotion}ly"

    return caption + "."

# =========================
# BEAM SEARCH
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
file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("Generate Caption"):
        raw = generate_caption(img, encoder, decoder)
        refined = refine_caption(raw)

        emotion = detect_emotion(refined)
        final = inject_emotion(refined, emotion)

        st.success("Generated Caption:")
        st.write(final)

        if emotion:
            st.info(f"Predicted Emotion: {emotion}")
