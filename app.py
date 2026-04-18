import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pickle
import torch.nn.functional as F

# =========================
# LOAD DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD VOCAB
# =========================
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# =========================
# MODEL CLASSES (SAME AS COLAB)
# =========================
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        features = features.unsqueeze(1).repeat(1, captions.size(1), 1)
        embeddings = self.embedding(captions)
        inputs = torch.cat((embeddings, features), dim=2)
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        return outputs


# =========================
# LOAD MODELS
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
# IMAGE TRANSFORM
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
# BEAM SEARCH
# =========================
def generate_caption(image, beam_width=3, max_len=20):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image)

    beams = [([vocab.stoi["<SOS>"]], 0.0, None)]
    completed = []

    for _ in range(max_len):
        all_candidates = []

        for seq, score, hidden in beams:
            last_word = seq[-1]

            if last_word == vocab.stoi["<EOS>"]:
                completed.append((seq, score))
                continue

            word_tensor = torch.tensor([[last_word]]).to(device)
            embedding = decoder.embedding(word_tensor)

            feature_expand = features.unsqueeze(1)
            lstm_input = torch.cat((embedding, feature_expand), dim=2)

            output, hidden_new = decoder.lstm(lstm_input, hidden)
            output = decoder.fc(output.squeeze(1))

            log_probs = F.log_softmax(output, dim=1)
            top_probs, top_idxs = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                next_word = top_idxs[0][i].item()
                next_score = score + top_probs[0][i].item()

                new_seq = seq + [next_word]
                all_candidates.append((new_seq, next_score, hidden_new))

        if len(all_candidates) == 0:
            break

        beams = sorted(all_candidates, key=lambda x: x[1]/len(x[0]), reverse=True)[:beam_width]

    best_seq = beams[0][0]

    words = []
    for idx in best_seq:
        word = vocab.itos[idx]
        if word == "<EOS>":
            break
        if word not in ["<SOS>", "<PAD>"]:
            words.append(word)

    return " ".join(words)

# =========================
# CLEAN + EMOTION
# =========================
def clean_caption(caption):
    words = caption.split()
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1] != w:
            cleaned.append(w)
    return " ".join(cleaned)

def get_emotion(caption):
    caption = caption.lower()
    if any(w in caption for w in ["run","jump","race"]):
        return "excited"
    if any(w in caption for w in ["play","dog","ball"]):
        return "playful"
    if any(w in caption for w in ["sit","bench"]):
        return "peaceful"
    return "neutral"

def enrich_caption(caption, emotion):
    if emotion == "neutral":
        return caption.capitalize() + "."
    return f"{emotion.capitalize()} {caption}."

# =========================
# STREAMLIT UI
# =========================
st.title("Emotion-Based Image Caption Generator")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    caption = generate_caption(image)
    caption = clean_caption(caption)

    emotion = get_emotion(caption)
    final_caption = enrich_caption(caption, emotion)

    st.subheader("Generated Caption")
    st.write(final_caption)

    st.subheader("Emotion")
    st.write(emotion)
