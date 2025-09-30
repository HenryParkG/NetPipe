# Description: Universal PyTorch Model Visualizer (Graph)
import streamlit as st
import torch
import torch.nn as nn
import tempfile
import os
from graphviz import Digraph

st.title("Universal PyTorch Model Visualizer (Graph)")

# YOLO 전용 로더
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 1. 모델 파일 업로드
model_file = st.file_uploader("Upload a PyTorch .pt/.pth model", type=["pt", "pth"])
input_shape_text = st.text_input("Example input shape (comma-separated, e.g. 1,3,224,224)")
max_depth = st.slider("Max Depth for visualization", 1, 10, 5)

if model_file and input_shape_text:
    input_shape = tuple(int(x) for x in input_shape_text.split(","))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name

    model = None
    try:
        # 2. YOLO 체크
        if YOLO_AVAILABLE:
            try:
                model = YOLO(tmp_path)
                st.success("YOLO model loaded successfully!")
            except Exception:
                pass

        # 3. 일반 PyTorch nn.Module 로드
        if model is None:
            try:
                with torch.serialization.safe_globals([DetectionModel] if YOLO_AVAILABLE else []):
                    loaded = torch.load(tmp_path, weights_only=False)
                if isinstance(loaded, nn.Module):
                    model = loaded
                elif isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], nn.Module):
                    model = loaded["model"]
                else:
                    st.error("Uploaded file does not contain a valid nn.Module.")
            except Exception as e:
                st.error(f"Error loading model: {e}")

        if model is not None:
            model.eval()
            st.subheader("Model Graph")

            # 4. Graphviz 그래프 생성
            dot = Digraph(comment='PyTorch Model', graph_attr={'rankdir': 'LR'})

            def add_nodes(module, parent_name=None, prefix="", current_depth=0):
                if current_depth > max_depth:
                    return
                for name, child in module.named_children():
                    node_name = f"{prefix}{name}"
                    # 컨테이너 레이어 처리
                    if isinstance(child, (nn.Sequential, nn.ModuleList)) or child.__class__.__name__ in ['Concat', 'Detect']:
                        out_shape = "Container / ?"
                    else:
                        try:
                            with torch.no_grad():
                                dummy_input = torch.zeros(*input_shape)
                                out = child(dummy_input)
                                out_shape = tuple(out.shape)
                        except:
                            out_shape = "?"
                    dot.node(node_name, f"{name}\n{child.__class__.__name__}\n{out_shape}", shape='box', style='rounded', fontsize='10')
                    if parent_name:
                        dot.edge(parent_name, node_name)
                    # 재귀 호출
                    add_nodes(child, node_name, prefix=node_name+".", current_depth=current_depth+1)

            add_nodes(model)
            st.graphviz_chart(dot)

    finally:
        os.remove(tmp_path)
