import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import gdown
import os
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    TimeDistributed, GlobalAveragePooling2D, Bidirectional, GRU,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

st.set_page_config(page_title="Deepfake Detection (Xception)", layout="centered")
st.title("🎭 Deepfake Detection (Xception + Bi-GRU)")
st.write("Upload a video to detect whether it's **REAL** or **FAKE** using the trained Xception model.")

# --- Constants ---
SEQ_LEN = 10
FRAME_SIZE = (128, 128)
WEIGHTS_PATH = "deepfake_detection_xception_best.weights.h5"

# --- Step 1: Download model weights from Google Drive if not present ---
@st.cache_resource
def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        url = "https://drive.google.com/uc?id=170Wvn04Do_etoN6Uh6tblAgLKYeAzmUk"
        st.info("📥 Downloading model weights from Google Drive...")
        gdown.download(url, WEIGHTS_PATH, quiet=False)
        st.success("✅ Weights downloaded successfully.")
    else:
        st.info("✅ Weights already available locally.")
    return WEIGHTS_PATH


# --- Step 2: Build the Xception model ---
@st.cache_resource
def build_xception_model(input_shape=(SEQ_LEN, 128, 128, 3)):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model = Sequential([
        TimeDistributed(base_model, input_shape=input_shape),
        TimeDistributed(GlobalAveragePooling2D()),
        BatchNormalization(),
        Bidirectional(GRU(128, return_sequences=False, kernel_regularizer=l2(0.005))),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model


# --- Step 3: Load model with downloaded weights ---
@st.cache_resource
def load_model():
    weights_path = download_weights()
    model = build_xception_model()
    model.load_weights(weights_path)
    return model


model = load_model()


# --- Helper: Extract frame sequences ---
def extract_frame_sequences(video_path, sequence_length=SEQ_LEN, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames, sequences = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
        if len(frames) == sequence_length:
            sequences.append(np.array(frames))
            frames = []
    cap.release()
    return np.array(sequences)


# --- Streamlit UI ---
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])
if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name
    st.video(video_path)

    st.info("⏳ Extracting frames and running predictions...")
    sequences = extract_frame_sequences(video_path)
    if len(sequences) == 0:
        st.error("No valid 10-frame sequences found in this video.")
    else:
        preds = model.predict(sequences)
        avg_pred = np.mean(preds, axis=0)
        label = "FAKE" if avg_pred[1] > avg_pred[0] else "REAL"
        confidence = float(max(avg_pred))

        st.subheader(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")
        st.success("✅ Analysis complete!")

st.caption("Model: Xception + Bi-GRU | Framework: TensorFlow + Streamlit")
