# train_yolo_stop.py
import streamlit as st
from ultralytics import YOLO
import os
import yaml
import tempfile
import threading

st.title("Train YOLO Model with Stop Button")

# -------------------------
# 모델 선택 / 업로드
# -------------------------
default_models = ["yolov5n.pt", "yolov5s.pt", "yolov8n.pt", "yolov8s.pt"]
selected_model = st.selectbox("Select a default YOLO model", default_models)
uploaded_model = st.file_uploader("Or upload your own YOLO .pt model", type=["pt"])

model_path = selected_model
if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        tmp_file.write(uploaded_model.getbuffer())
        model_path = tmp_file.name

st.info(f"Using model: {model_path}")

# -------------------------
# 데이터셋 경로
# -------------------------
train_path = st.text_input("Train folder path")
val_path = st.text_input("Validation folder path")
model_save_path = st.text_input("Save trained model path (.pt)", value="./yolo_model.pt")

# -------------------------
# 로그 영역
# -------------------------
log_area = st.empty()
stop_flag = threading.Event()  # 중지 플래그

# -------------------------
# 데이터셋 YAML 생성
# -------------------------
def create_yaml(train, val, num_classes=1, names=["chip"]):
    data_dict = {
        'train': train,
        'val': val,
        'nc': num_classes,
        'names': names
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        yaml.dump(data_dict, f)
        return f.name

yaml_path = None
if train_path and val_path:
    yaml_path = create_yaml(train_path, val_path)

# -------------------------
# 하이퍼파라미터
# -------------------------
st.subheader("Training Hyperparameters")
epochs = st.number_input("Epochs", 1, 500, 50)
batch_size = st.number_input("Batch size", 1, 128, 16)
learning_rate = st.number_input("Learning rate", 1e-5, 1e-1, 0.001, format="%.5f")

# -------------------------
# 학습 스레드 함수
# -------------------------
def train_yolo():
    try:
        model = YOLO(model_path)
        for epoch in range(epochs):
            if stop_flag.is_set():
                log_area.text("Training stopped by user!")
                return

            model.train(
                data=yaml_path,
                epochs=1,              # 1 epoch씩 반복하며 중지 체크
                batch=batch_size,
                lr0=learning_rate,
                project="runs/train_streamlit",
                name="yolo_training",
                exist_ok=True,
                verbose=True
            )
            log_area.text(f"Epoch {epoch+1}/{epochs} finished...")

        model.save(model_save_path)
        log_area.text(f"Training finished!\nSaved at {model_save_path}")

    except Exception as e:
        log_area.text(f"Training failed: {e}")

# -------------------------
# 버튼
# -------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Training"):
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            st.error("Invalid folder path!")
        elif not yaml_path:
            st.error("Failed to generate dataset YAML.")
        else:
            stop_flag.clear()
            t = threading.Thread(target=train_yolo)
            t.start()

with col2:
    if st.button("Stop Training"):
        stop_flag.set()
        log_area.text("Stop requested... waiting for current epoch to finish.")
