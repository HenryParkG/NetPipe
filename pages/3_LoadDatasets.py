# Description: 이미지와 YOLO 라벨 파일을 업로드하고, 그리드 형태로 미리보기
import streamlit as st
from PIL import Image, ImageDraw
import math
import io

st.set_page_config(
    page_title="Dataset Upload & YOLO Label Preview",
    layout="wide",      # 좌우 여백 최소화
    initial_sidebar_state="auto"
)

st.title("Dataset Upload & YOLO Label Preview")


# 1. 이미지 업로드
uploaded_images = st.file_uploader(
    "Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True
)

# 2. 라벨 파일 업로드 (YOLO 형식)
uploaded_labels = st.file_uploader(
    "Upload corresponding YOLO label files (same order as images)", 
    type=["txt"], accept_multiple_files=True
)

if uploaded_images:
    num_images = len(uploaded_images)
    cols_count = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols_count)

    st.subheader("Grid Preview with YOLO Labels")

    for r in range(rows):
        cols = st.columns(cols_count)
        for c in range(cols_count):
            idx = r*cols_count + c
            if idx < num_images:
                img = Image.open(uploaded_images[idx]).convert("RGB")
                draw = ImageDraw.Draw(img)
                
                # YOLO 라벨이 존재하면 오버레이
                if uploaded_labels and idx < len(uploaded_labels):
                    label_file = uploaded_labels[idx]
                    lines = label_file.read().decode("utf-8").splitlines()
                    w, h = img.size
                    for line in lines:
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        class_id, x_center, y_center, width, height = map(float, parts)
                        # YOLO 좌표 변환
                        x1 = (x_center - width/2) * w
                        y1 = (y_center - height/2) * h
                        x2 = (x_center + width/2) * w
                        y2 = (y_center + height/2) * h
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                        draw.text((x1, y1), str(int(class_id)), fill="red")
                
                cols[c].image(img, use_container_width=True)
