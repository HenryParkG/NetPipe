# Description: 데이터셋 증강 웹앱
import streamlit as st
import zipfile
import os
import tempfile
from PIL import Image, ImageEnhance
import io

st.title("Dataset Augmentation")

# 1. 데이터셋 업로드 (zip)
uploaded_file = st.file_uploader("Upload dataset ZIP", type=["zip"])
if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # ZIP 압축 해제
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # 이미지 파일 리스트
        img_extensions = (".png", ".jpg", ".jpeg")
        img_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(tmpdir) 
                     for f in filenames if f.lower().endswith(img_extensions)]
        st.write(f"{len(img_files)} images loaded.")

        # 2. 증강 옵션 선택
        st.subheader("Select Augmentation Options")
        rotate = st.checkbox("Rotate 90°")
        flip_h = st.checkbox("Horizontal Flip")
        flip_v = st.checkbox("Vertical Flip")
        brightness = st.checkbox("Increase Brightness")
        contrast = st.checkbox("Increase Contrast")

        if st.button("Create Augmented Dataset"):
            augmented_dir = os.path.join(tmpdir, "augmented")
            os.makedirs(augmented_dir, exist_ok=True)

            for img_path in img_files:
                img = Image.open(img_path)
                basename = os.path.basename(img_path).split('.')[0]

                augmented_imgs = [img]

                # 회전
                if rotate:
                    augmented_imgs.append(img.rotate(90))
                # 수평 뒤집기
                if flip_h:
                    augmented_imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
                # 수직 뒤집기
                if flip_v:
                    augmented_imgs.append(img.transpose(Image.FLIP_TOP_BOTTOM))
                # 밝기
                if brightness:
                    enhancer = ImageEnhance.Brightness(img)
                    augmented_imgs.append(enhancer.enhance(1.5))
                # 대비
                if contrast:
                    enhancer = ImageEnhance.Contrast(img)
                    augmented_imgs.append(enhancer.enhance(1.5))

                # 저장
                for i, aug_img in enumerate(augmented_imgs):
                    save_path = os.path.join(augmented_dir, f"{basename}_aug{i}.png")
                    aug_img.save(save_path)

            # ZIP 생성
            zip_out_path = os.path.join(tmpdir, "augmented_dataset.zip")
            with zipfile.ZipFile(zip_out_path, 'w') as zipf:
                for root, _, files in os.walk(augmented_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=file)

            # 다운로드 링크 제공
            st.success("Augmented dataset created!")
            with open(zip_out_path, "rb") as f:
                st.download_button(
                    label="Download Augmented Dataset ZIP",
                    data=f,
                    file_name="augmented_dataset.zip",
                    mime="application/zip"
                )
