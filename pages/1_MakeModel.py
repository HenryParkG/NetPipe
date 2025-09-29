import streamlit as st
import yaml
from ultralytics import YOLO
import tempfile

st.title("Custom YOLOv8 Model Builder (Latest ultralytics)")

# 1️⃣ 클래스 수 입력
num_classes = st.number_input("Number of classes", min_value=1, max_value=100, value=3)

# 2️⃣ 레이어 구성
st.subheader("Add Layers to Backbone / Head")

if 'layers' not in st.session_state:
    st.session_state.layers = []

# 레이어 추가 UI
layer_type = st.selectbox("Layer type", ["Conv", "C3", "Detect"])
out_channels = st.number_input("Output channels / filters", min_value=1, max_value=1024, value=64)
kernel_size = st.number_input("Kernel size (Conv only)", min_value=1, max_value=7, value=3)
stride = st.number_input("Stride (Conv only)", min_value=1, max_value=3, value=1)

if st.button("Add Layer"):
    layer = {
        "type": layer_type,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride
    }
    st.session_state.layers.append(layer)

st.write("Current Layers:", st.session_state.layers)

if st.button("Clear Layers"):
    st.session_state.layers = []

# 3️⃣ 모델 생성 및 저장
if st.button("Build YOLOv8 Model"):
    # YAML 생성
    yaml_dict = {
        "nc": num_classes,
        "depth_multiple": 0.33,
        "width_multiple": 0.25,
        "backbone": [],
        "head": []
    }

    backbone_layers = []
    head_layers = []

    for l in st.session_state.layers:
        if l["type"] == "Detect":
            # Detect 레이어는 반드시 마지막
            head_layers.append([-1, 1, "Detect", [num_classes]])
        elif l["type"] == "Conv":
            # Conv는 [out_channels, kernel, stride]
            backbone_layers.append([-1, 1, "Conv", [l["out_channels"], l["kernel_size"], l["stride"]]])
        elif l["type"] == "C3":
            # C3는 int만 전달
            backbone_layers.append([-1, 1, "C3", l["out_channels"]])

    yaml_dict["backbone"] = backbone_layers
    yaml_dict["head"] = head_layers if head_layers else [[-1, 1, "Detect", [num_classes]]]

    # 임시 YAML 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(yaml_dict, f)
        yaml_path = f.name

    # YOLO 객체로 불러오기
    try:
        model = YOLO(yaml_path)
    except Exception as e:
        st.error(f"Error building model: {e}")
        st.stop()

    # 저장
    save_path = st.text_input("Save path for .pt file", value="./custom_yolo.pt")
    if save_path:
        model.save(save_path)
        st.success(f"YOLOv8 model saved at {save_path}")
        st.download_button(
            label="Download YOLO Model",
            data=open(save_path, "rb").read(),
            file_name="custom_yolo.pt",
            mime="application/octet-stream"
        )
