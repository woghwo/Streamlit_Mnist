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
