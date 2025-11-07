import io
import os
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from tensorflow.keras import layers

# =========================================
# 기본 설정
# =========================================
st.set_page_config(page_title="MNIST Recognizer", page_icon="🧮", layout="centered")
st.title("🧮 MNIST Recognizer")

MODEL_PATH = "model.h5"
CANVAS_SIZE = 192

# =========================================
# 모델 생성 / 로드
# =========================================
@st.cache_resource
def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        st.success(f"Loaded model from {MODEL_PATH}")
    else:
        # 입력: 784 (28×28)
        model = keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        model.save(MODEL_PATH)
        st.warning("No pre-trained weights found. Created and saved an untrained model.")
    return model

model = load_or_create_model()

# =========================================
# 캔버스
# =========================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Draw a digit")
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas"
    )

# =========================================
# 전처리 함수
# =========================================
def preprocess(img_rgba: np.ndarray):
    """RGBA → 28×28 흑백 → 784 벡터 (0~1 정규화)"""
    img_rgba = img_rgba.astype(np.uint8)
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
    img28 = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_norm = img28.astype("float32") / 255.0
    x_input = img_norm.reshape(1, 784)  # flatten to (1, 784)
    return x_input, img28

# =========================================
# 예측
# =========================================
if canvas.image_data is not None:
    x_input, img28 = preprocess(canvas.image_data)
    preview = cv2.resize(img28, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

    with col2:
        st.subheader("Preview")
        st.image(preview, clamp=True, caption="28×28 Preview", width=CANVAS_SIZE)

        y = model.predict(x_input, verbose=0).squeeze()
        pred = int(np.argmax(y))

        st.write(f"## Predicted: **{pred}**")
        st.bar_chart(y)
else:
    st.info("Draw a digit on the left to predict.")
