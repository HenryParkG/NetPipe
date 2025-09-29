import streamlit as st
from tensorflow.keras import layers, models
import tempfile
import os
import tensorflow as tf
import io

st.title("TensorFlow CNN Designer GUI")

# -----------------------------
# 1. 네트워크 구성
# -----------------------------
layers_list = []

num_layers = st.slider("Number of Conv layers", 1, 5, 2)
for i in range(num_layers):
    filters = st.number_input(f"Filters in layer {i+1}", 8, 512, 32)
    kernel = st.number_input(f"Kernel size for layer {i+1}", 1, 7, 3)
    layers_list.append((filters, kernel))

# -----------------------------
# 2. Build & Download Model
# -----------------------------
if st.button("Build & Download Model"):
    model = models.Sequential()
    model.add(layers.Input(shape=(64,64,3)))
    for f, k in layers_list:
        model.add(layers.Conv2D(f, (k,k), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    # 모델 summary 표시
    st.text("Model Summary:")
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    st.text(stream.getvalue())

    # 임시 파일에 저장
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name
    model.save(tmp_path)  # 실제 파일 경로로 저장

    # Streamlit 다운로드 버튼
    with open(tmp_path, "rb") as f:
        st.download_button(
            label="Download Model (.h5)",
            data=f,
            file_name="model.h5",
            mime="application/octet-stream"
        )

    # 임시 파일 삭제
    os.remove(tmp_path)
