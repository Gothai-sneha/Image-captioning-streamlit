
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
# CLEAN CAPTION (COLAB STYLE)
# =========================
def clean_caption(caption):
    words = caption.split()
    cleaned = []

    for word in words:
        if len(cleaned) == 0 or cleaned[-1] != word:
            cleaned.append(word)

    caption = " ".join(cleaned)
    caption = caption.strip().rstrip(".").lower()

    return caption

# =========================
# EMOTION FROM CAPTION (COLAB STYLE)
# =========================
def get_emotion_from_caption(caption):
    text = caption.lower()

    if any(w in text for w in ["run", "jump", "race", "bike"]):
        return "excited"

    elif any(w in text for w in ["play", "child", "dog", "ball"]):
        return "playful"

    elif any(w in text for w in ["sit", "bench", "lake", "water"]):
        return "peaceful"

    elif any(w in text for w in ["smile", "laugh"]):
        return "happy"

    elif any(w in text for w in ["cry", "sad"]):
        return "sad"

    return "neutral"

# =========================
# ENRICH CAPTION (COLAB STYLE)
# =========================
def enrich_caption_with_emotion(caption, emotion):

    if emotion == "neutral":
        return caption.capitalize() + "."

    words = caption.split()

    if len(words) == 0:
        return "An image."

    if words[0] in ["a", "an", "the"]:
        enriched = f"{words[0]} {emotion} " + " ".join(words[1:])
    else:
        enriched = f"{emotion} {caption}"

    return enriched.capitalize() + "."

# =========================
# BEAM SEARCH (SAME AS YOUR CODE)
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

        raw_caption = generate_caption(img, encoder, decoder)

        caption = clean_caption(raw_caption)

        emotion = get_emotion_from_caption(caption)

        final_caption = enrich_caption_with_emotion(caption, emotion)

        st.success("Emotion Enriched Caption:")
        st.write(f'"{final_caption}"')

        st.info(f"Predicted Emotion: {emotion}")

