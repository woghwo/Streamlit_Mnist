🧮 MNIST Recognizer

Streamlit + TensorFlow 기반의 실시간 손글씨 숫자 인식 웹 애플리케이션

<p align="center"> <img src="https://github.com/yourusername/mnist-recognizer/assets/demo.gif" width="480" alt="demo preview"/> </p>
🚀 프로젝트 개요

이 프로젝트는 사용자가 캔버스에 숫자를 직접 그리면,
MNIST 데이터셋으로 학습된 신경망(MLP) 모델이 해당 숫자를 예측해주는 웹 애플리케이션입니다.

항목	내용
Framework	Streamlit
Model	TensorFlow / Keras (Multi-Layer Perceptron)
Dataset	MNIST (28×28 grayscale digits)
기능 요약	🎨 자유롭게 숫자 입력 → 🧠 실시간 예측 → 📊 확률 분포 시각화
🗂 프로젝트 구조
📁 mnist-recognizer/
│
├── app.py                # Streamlit 웹 앱 메인 코드
├── train_model.py        # MNIST 데이터셋 학습 코드
├── model.h5              # 학습된 모델 (자동 생성)
│
├── requirements.txt      # 의존성 패키지 목록
├── README.md             # 프로젝트 설명
└── assets/
    ├── demo.gif          # 데모 애니메이션 (선택)
    └── example1.png      # 예시 이미지


requirements.txt 예시

streamlit
tensorflow
opencv-python
streamlit-drawable-canvas
numpy

💡 주요 기능

🎨 Canvas 입력 : 마우스로 직접 숫자(0~9) 입력

⚙️ 전처리 : RGBA → Grayscale → 28×28 → Flatten → Normalize

🧠 예측 : 학습된 MLP 모델로 예측 수행

📊 시각화 : bar chart를 통해 예측 확률 분포 표시

🧠 모델 학습 (train_model.py)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

# 훈련/테스트셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)) / 255.0
x_test = np.reshape(x_test, (-1, 784)) / 255.0

# 모델 정의
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
model.save("model.h5")
print("✅ Saved model.h5")

🌐 실행 방법
1️⃣ 의존성 설치
pip install -r requirements.txt

2️⃣ 모델 학습
python train_model.py

3️⃣ Streamlit 앱 실행
streamlit run app.py

🧩 Streamlit App (app.py)
import io
import os
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="MNIST Recognizer", page_icon="🧮", layout="centered")
st.title("🧮 MNIST Recognizer")

MODEL_PATH = "model.h5"
CANVAS_SIZE = 192

@st.cache_resource
def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        st.success(f"Loaded model from {MODEL_PATH}")
    else:
        model = keras.Sequential([
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        model.save(MODEL_PATH)
        st.warning("No pre-trained weights found. Created an untrained model.")
    return model

model = load_or_create_model()

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

def preprocess(img_rgba: np.ndarray):
    img_rgba = img_rgba.astype(np.uint8)
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
    img28 = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_norm = img28.astype("float32") / 255.0
    x_input = img_norm.reshape(1, 784)
    return x_input, img28

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

🌍 Streamlit Cloud 배포 (선택)

GitHub 저장소를 public으로 설정

Streamlit Cloud
 접속

New app → GitHub repo 연결 → branch와 파일(app.py) 선택

자동 빌드 후 웹에서 즉시 실행 🎉

💡 예시 URL:
👉 https://mnist-recognizer.streamlit.app

📊 결과 예시
입력 (Canvas)	예측 결과

	Predicted: 3
📚 참고 자료

Streamlit Documentation

TensorFlow MNIST Tutorial

streamlit-drawable-canvas
