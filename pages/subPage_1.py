# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import io
from pathlib import Path
from tkinter import Tk, filedialog
from tensorflow.keras.utils import plot_model
import os

st.title("TensorFlow CNN Designer GUI")

# 네트워크 구성
layers_list = []

num_layers = st.slider("Number of Conv layers", 1, 5, 2)
for i in range(num_layers):
    filters = st.number_input(f"Filters in layer {i+1}", 8, 512, 32)
    kernel = st.number_input(f"Kernel size for layer {i+1}", 1, 7, 3)
    layers_list.append((filters, kernel))

# Build model 버튼
if st.button("Build Model"):
    model = models.Sequential()
    model.add(layers.Input(shape=(64,64,3)))
    for f, k in layers_list:
        model.add(layers.Conv2D(f, (k,k), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    
    st.write(model.summary())
    
    # 모델 summary를 문자열로 캡쳐


    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    
    st.text(summary_str)
    
    # 모델 저장 (사용자가 폴더 및 파일 이름 선택)
    st.write("Enter the file path below to save the model.")
    save_path = st.text_input("File path to save the model (e.g., 'model.h5')", value="./model.h5")
    if st.button("Save Model"):
        if save_path.strip():
            try:
                absolute_save_path = os.path.abspath(save_path.strip())
                model.save(absolute_save_path)
                st.success(f"Model saved successfully at {absolute_save_path}")
            except Exception as e:
                st.error(f"Error saving model: {e}")
        else:
            st.warning("Please provide a valid file path.")

    # Display model architecture as a plot
    try:
        st.write("Visualize the model architecture:")
        plot_path = "model_plot.png"
        plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Model Architecture", use_column_width=True)
            os.remove(plot_path)  # Clean up the generated file
        else:
            st.warning("Failed to generate model plot.")
    except ImportError:
        st.error("Failed to import required libraries for plotting. Ensure pydot and graphviz are installed.")