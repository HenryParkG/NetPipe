# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models

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
