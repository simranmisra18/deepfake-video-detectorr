import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import gdown
import os
import gc
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    TimeDistributed, GlobalAveragePooling2D, Bidirectional, GRU,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# --------------------------
# APP CONFIG
# --------------------------
st.set_page_config(page_title="Deepfake Detection (Xception)", layout="centered")
st.title("🎭 Deepfake Detection (Xception + Bi-GRU)")
st.write("Upload a video to detect whether it's **REAL** or **FAKE** using the trained Xception model.")

# --------------------------
# CONSTANTS
# --------------------------
SEQ_LEN = 10
FRAME_SIZE = (128, 128)
MAX_VIDEO_SIZE_MB = 50
MAX_SEQUENCES = 20
WEIGHTS_PATH = "deepfake_detection_xception_best.weights.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=170Wvn04Do_etoN6Uh6tblAgLKYeAzmUk"

# --------------------------
# DOWNLOAD WEIGHTS IF NEEDED
# --------------------------
@st.cache_resource
def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        st.info("📥 Downloading model weights from Google Drive...")
        gdown.download(GDRIVE_URL, WEIGHTS_PATH, quiet=False)
        st.success("✅ Weights downloaded successfully.")
    else:
        st.info("✅ Weights already available locally.")
    return WEIGHTS_PATH

# --------------------------
# BUILD MODEL ARCHITECTURE
# --------------------------
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

# --------------------------
# LOAD MODEL WITH WEIGHTS
# --------------------------
@st.cache_resource
def load_model():
    weights_path = download_weights()
    model = build_xception_model()
    model.load_weights(weights_path)
    return model

model = load_model()

# --------------------------
# FRAME EXTRACTION (Optimized)
# --------------------------
def extract_frame_sequences(video_path, sequence_length=SEQ_LEN, frame_size=FRAME_SIZE, max_sequences=MAX_SEQUENCES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // (sequence_length * max_sequences), 1)

    frames, sequences = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, frame_size)
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
            if len(frames) == sequence_length:
                sequences.append(np.array(frames))
                frames = []
        frame_idx += 1

    cap.release()
    return np.array(sequences)

# --------------------------
# VIDEO UPLOAD + PREDICTION
# --------------------------
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    # 1️⃣ Limit file size
    if uploaded_video.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        st.error(f"🚫 Video too large! Please upload a clip under {MAX_VIDEO_SIZE_MB} MB (~20–30 seconds).")
        st.stop()

    # 2️⃣ Save video temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name
    st.video(video_path)

    # 3️⃣ Extract frames efficiently
    st.info("⏳ Extracting frames and preparing for analysis...")
    sequences = extract_frame_sequences(video_path)

    if len(sequences) == 0:
        st.error("No valid 10-frame sequences found in this video.")
    else:
        st.success(f"✅ Extracted {len(sequences)} sequences for prediction.")
        st.info("🧠 Running predictions...")
        progress = st.progress(0)
        preds = []

        # 4️⃣ Predict with progress bar
        for i, seq in enumerate(sequences):
            pred = model.predict(np.expand_dims(seq, axis=0), verbose=0)
            preds.append(pred)
            progress.progress((i + 1) / len(sequences))

        # 5️⃣ Average results
        avg_pred = np.mean(preds, axis=0)
        label = "FAKE" if avg_pred[0][1] > avg_pred[0][0] else "REAL"
        confidence = float(max(avg_pred[0]))

        st.subheader(f"🎯 Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        # 6️⃣ Cleanup memory
        K.clear_session()
        gc.collect()
        st.success("✅ Analysis complete and resources cleared.")

st.caption("Model: Xception + Bi-GRU | Optimized for Streamlit Cloud")
