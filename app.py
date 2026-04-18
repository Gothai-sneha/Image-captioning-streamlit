import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import pickle
import random

# -------------------------------
# Load Models & Tokenizer
# -------------------------------
@st.cache_resource
def load_models():
    encoder = tf.keras.models.load_model("encoder_model.h5")
    decoder = tf.keras.models.load_model("decoder_model.h5")
    emotion_model = tf.keras.models.load_model("emotion_model.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    return encoder, decoder, emotion_model, tokenizer

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Beam Search Caption Generator
# -------------------------------
def beam_search_caption(model, image_features, tokenizer, max_length=34, beam_width=3):
    start = tokenizer.word_index['startseq']
    end = tokenizer.word_index['endseq']

    sequences = [[[], 0.0]]

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:

            if len(seq) > 0 and seq[-1] == end:
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([image_features, padded], verbose=0)[0]

            top_k = np.argsort(preds)[-beam_width:]

            for idx in top_k:
                prob = preds[idx]
                new_seq = seq + [idx]
                new_score = score - np.log(prob + 1e-10)
                all_candidates.append((new_seq, new_score))

        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]

    caption = []
    for idx in best_seq:
        word = tokenizer.index_word.get(idx)
        if word == 'endseq':
            break
        if word != 'startseq':
            caption.append(word)

    return ' '.join(caption)

# -------------------------------
# Emotion Prediction
# -------------------------------
def predict_emotion(model, image_features):
    preds = model.predict(image_features)[0]
    emotions = ["happy", "sad", "angry", "excited", "neutral"]

    idx = np.argmax(preds)
    return emotions[idx], float(preds[idx])

# -------------------------------
# Emotion Injection
# -------------------------------
emotion_map = {
    "happy": ["happily", "joyfully"],
    "excited": ["excitedly", "energetically"],
    "sad": ["sadly"],
    "angry": ["angrily"],
    "neutral": []
}


def inject_emotion(caption, emotion, confidence):
    if confidence < 0.4 or emotion == "neutral":
        return caption

    words = caption.split()
    emo_words = emotion_map.get(emotion, [])

    if not emo_words:
        return caption

    word = random.choice(emo_words)
    insert_pos = min(2, len(words))
    words.insert(insert_pos, word)

    return " ".join(words)

# -------------------------------
# Caption Enhancement
# -------------------------------
def enhance_caption(caption):
    caption = caption.replace("is doing", "performs")
    caption = caption.replace("a person", "someone")
    caption = caption.replace("a man", "a man")
    caption = caption.replace("a woman", "a woman")
    caption = caption.replace("a dog", "a dog")
    caption = caption.replace("a cat", "a cat")
    return caption.capitalize()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Emotion-Aware Image Caption Generator")

encoder, decoder, emotion_model, tokenizer = load_models()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    features = encoder.predict(processed)

    # Generate Caption
    caption = beam_search_caption(decoder, features, tokenizer)

    # Predict Emotion
    emotion, confidence = predict_emotion(emotion_model, features)

    # Inject Emotion
    final_caption = inject_emotion(caption, emotion, confidence)

    # Enhance
    final_caption = enhance_caption(final_caption)

    st.subheader("Generated Caption:")
    st.write(final_caption)

    st.subheader("Emotion:")
    st.write(f"{emotion} (confidence: {confidence:.2f})")
