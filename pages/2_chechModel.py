# visualize_model.py
import streamlit as st
from ultralytics import YOLO
import tempfile
import os

st.title("YOLO Model Visualizer")

# 1. 모델 파일 업로드
model_file = st.file_uploader("Upload a YOLOv8 .pt model", type=["pt"])
if model_file:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name

    try:
        # 2. 모델 로드
        model = YOLO(tmp_path)
        st.success("Model loaded successfully!")

        # 3. 모델 정보 표시
        st.subheader("Model Summary")
        st.text(model.model)

        # 4. 레이어별 정보
        st.subheader("Layer Details")
        for i, layer in enumerate(model.model.model):
            st.write(f"Layer {i}: {layer}")

        # 5. 클래스 정보
        st.subheader("Classes")
        st.write(model.names)

    except Exception as e:
        st.error(f"Error loading model: {e}")

    finally:
        os.remove(tmp_path)
