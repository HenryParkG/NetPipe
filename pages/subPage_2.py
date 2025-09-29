import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import io
import os
import tempfile

import graphviz
import pydot

st.title("Load & Visualize Keras Model")

# 모델 업로드
model_file = st.file_uploader("Upload Keras Model (.h5 or .keras)", type=["h5","keras"])

if model_file is not None:
    try:
        # 업로드 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.name)[1]) as tmp:
            tmp.write(model_file.read())
            tmp_path = tmp.name

        # 임시 파일 경로로 모델 로드
        model = tf.keras.models.load_model(tmp_path)
        st.success("Model loaded successfully!")

        # 모델 summary
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_str = stream.getvalue()
        stream.close()
        st.text(summary_str)

        # 모델 시각화
        # try:
        #     plot_path = "model_plot.png"
        #     plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        #     if os.path.exists(plot_path):
        #         st.image(plot_path, caption="Model Architecture", use_column_width=True)
        #         os.remove(plot_path)
        #     else:
        #         st.warning("Failed to generate model plot.")
        # except Exception as e:
        #     st.error(f"Error plotting model: {e}")

        # 임시 파일 삭제
        os.remove(tmp_path)

    except Exception as e:
        st.error(f"Error loading model: {e}")
