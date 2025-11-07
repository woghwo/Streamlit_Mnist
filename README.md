# MNIST Recognizer

간단한 Streamlit 기반 손글씨 숫자(0–9) 인식 데모입니다.  
두 파일이 포함되어 있습니다:

- `train.py` — MNIST 데이터셋으로 간단한 다층 퍼셉트론(MLP) 모델을 학습하고 `model.h5`로 저장합니다.
- `inference_server.py` — Streamlit 앱으로, 좌측 캔버스에 손으로 숫자를 그리면 모델이 예측값을 보여줍니다. 모델 파일이 없으면 동일한 구조의 빈(학습되지 않은) 모델을 생성하여 저장합니다.

---

## 빠른 시작

1. (권장) 가상환경 생성 및 활성화
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   ```

2. 필요한 패키지 설치
   ```bash
   pip install streamlit streamlit-drawable-canvas tensorflow numpy opencv-python
   ```

3. 모델을 학습하고 저장하려면
   ```bash
   python train.py
   ```
   - 기본 설정: 5 에포크, 배치 크기 128, 검증 데이터 10% 사용.
   - 학습이 끝나면 `model.h5`가 생성됩니다.

4. Streamlit 앱 실행 (모델이 없으면 앱이 빈 모델을 생성하고 저장합니다)
   ```bash
   streamlit run inference_server.py
   ```
   브라우저가 자동으로 열리지 않으면 출력된 로컬 URL을 복사해서 접속하세요 (예: http://localhost:8501).

---

## 파일 설명

- train.py
  - TensorFlow/Keras로 간단한 MLP 모델 정의 및 MNIST 학습 코드입니다.
  - 학습된 모델을 `model.h5`로 저장합니다.
     
 

- inference_server.py
  - Streamlit UI: 좌측에 drawable canvas, 우측에 28×28 미리보기와 예측 결과(확률 바 차트).
  - 모델 로드/생성: `MODEL_PATH = "model.h5"`가 있으면 로드하고, 없으면 동일 구조의 untrained 모델을 생성 후 저장합니다.
  - 전처리: 캔버스의 RGBA 이미지를 그레이스케일로 변환 → 28×28로 리사이즈 → [0,1] 정규화 → (1, 784) 형태의 입력으로 변환합니다.
  - 예측 출력: softmax 확률과 argmax(예측 숫자)를 표시합니다.
 
<img width="741" height="875" alt="image" src="https://github.com/user-attachments/assets/7386578e-b4ec-4492-8e75-805a513440e8" />




---

## 모델 및 전처리 상세

- 모델 구조
  - 입력: 784 (28×28 이미지를 flatten)
  - Dense(256, relu) → Dense(512, relu) → Dense(10, softmax)

- 전처리 (inference_server.py)
  - 캔버스에서 RGBA를 받아 cv2.COLOR_RGBA2GRAY로 그레이스케일 변환
  - cv2.resize(..., (28, 28), interpolation=cv2.INTER_AREA)
  - 0–255 → 0.0–1.0로 정규화
  - (1, 784) 형태로 모델에 입력

- 주의: 캔버스 배경/선 색상 설정(현재 배경 검정, 선 흰색)을 유지해야 학습 데이터(MNIST; 흰 글자/검은 배경 반전 여부)를 고려한 결과가 더 안정적입니다.(MNIST는 흰 배경에 검은 글자가 기본이므로 필요 시 색상 반전을 고려해야 합니다.)

