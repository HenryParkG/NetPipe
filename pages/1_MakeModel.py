import streamlit as st
import yaml
from pathlib import Path
import json
import tempfile
import os

st.set_page_config(page_title="YOLOv11 Network Builder", layout="wide")

# 세션 상태 초기화
if 'layers' not in st.session_state:
    st.session_state.layers = []
if 'model_config' not in st.session_state:
    st.session_state.model_config = {
        'nc': 80,  # number of classes
        'scales': {
            'n': [0.50, 0.25, 1024],
            's': [0.50, 0.50, 1024],
            'm': [0.50, 1.00, 512],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.50, 512]
        }
    }

# YOLO 레이어 타입 정의 (YOLOv11 기준)
LAYER_TYPES = {
    'Conv': 'Convolution Layer',
    'C3k2': 'C3k2 (CSPNet with 2 convolutions and kernel size flexibility)',
    'C2PSA': 'C2 with Polarized Self-Attention',
    'C2f': 'C2f (CSPNet with 2 convolutions - legacy)',
    'C3': 'C3 (CSPNet with 3 convolutions - legacy)',
    'SPPF': 'Spatial Pyramid Pooling - Fast',
    'Upsample': 'Upsampling Layer',
    'Concat': 'Concatenation Layer',
    'Detect': 'Detection Head',
    'nn.Upsample': 'PyTorch Upsample',
    'PSA': 'Polarized Self-Attention',
}

def create_yolo11_template():
    """YOLOv11n 공식 템플릿 생성"""
    return {
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
            [-1, 2, 'C3k2', [256, False, 0.25]],
            [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
            [-1, 2, 'C3k2', [512, False, 0.25]],
            [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
            [-1, 2, 'C3k2', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
            [-1, 2, 'C3k2', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
            [-1, 2, 'C2PSA', [1024]],  # 10
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 2, 'C3k2', [512, False]],  # 13
            
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 2, 'C3k2', [256, False]],  # 16 (P3/8-small)
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 13], 1, 'Concat', [1]],  # cat head P4
            [-1, 2, 'C3k2', [512, False]],  # 19 (P4/16-medium)
            
            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],  # cat head P5
            [-1, 2, 'C3k2', [1024, True]],  # 22 (P5/32-large)
            
            [[16, 19, 22], 1, 'Detect', ['nc']],  # Detect(P3, P4, P5)
        ]
    }

def add_layer(layer_type, from_layer, number, args):
    """레이어 추가"""
    layer = {
        'from': from_layer,
        'number': number,
        'type': layer_type,
        'args': args
    }
    st.session_state.layers.append(layer)

def remove_layer(index):
    """레이어 제거"""
    if 0 <= index < len(st.session_state.layers):
        st.session_state.layers.pop(index)

def generate_yaml_config():
    """YAML 설정 파일 생성 (YOLOv11 형식)"""
    config = {
        'nc': st.session_state.model_config['nc'],
        'scales': st.session_state.model_config['scales'],
        'backbone': [],
        'head': []
    }
    
    # 레이어를 backbone과 head로 분류
    in_head = False
    for layer in st.session_state.layers:
        layer_def = [
            layer['from'],
            layer['number'],
            layer['type'],
            layer['args']
        ]
        
        # Detect 레이어나 첫 Upsample이 나타나면 head로 전환
        if layer['type'] in ['Detect', 'nn.Upsample'] and not in_head:
            in_head = True
        
        if in_head:
            config['head'].append(layer_def)
        else:
            config['backbone'].append(layer_def)
    
    return config

def save_model_config(filename):
    """모델 설정 저장"""
    config = generate_yaml_config()
    
    # YAML 헤더 추가
    yaml_header = """# Ultralytics YOLO11 Custom Model
# Custom architecture based on YOLOv11

# Parameters
"""
    
    yaml_str = yaml_header + yaml.dump(config, default_flow_style=False, sort_keys=False)
    return yaml_str

def build_pt_model(yaml_content, output_name):
    """YAML 설정을 기반으로 .pt 모델 파일 생성"""
    try:
        from ultralytics import YOLO
        
        # 임시 YAML 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_yaml_path = f.name
        
        # YAML로부터 모델 생성 (가중치 초기화)
        model = YOLO(temp_yaml_path)
        
        # .pt 파일로 저장
        temp_pt_path = temp_yaml_path.replace('.yaml', '.pt')
        model.model.save(temp_pt_path)
        
        # 저장된 .pt 파일 읽기
        with open(temp_pt_path, 'rb') as f:
            pt_data = f.read()
        
        # 임시 파일 삭제
        os.unlink(temp_yaml_path)
        os.unlink(temp_pt_path)
        
        return pt_data, None
        
    except ImportError:
        return None, "ultralytics 패키지가 설치되어 있지 않습니다. 'pip install ultralytics' 명령으로 설치해주세요."
    except Exception as e:
        return None, f"모델 빌드 중 오류 발생: {str(e)}"

# 메인 UI
st.title("🚀 YOLOv11 Network Builder")
st.markdown("YOLOv11 네트워크 구조를 시각적으로 수정하고 빌드하세요")

# 사이드바 - 모델 설정
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    st.session_state.model_config['nc'] = st.number_input(
        "Number of Classes (nc)",
        min_value=1,
        value=st.session_state.model_config['nc'],
        help="탐지할 클래스 개수"
    )
    
    st.subheader("Model Scales")
    scale_type = st.selectbox(
        "Select Scale",
        options=['n', 's', 'm', 'l', 'x'],
        format_func=lambda x: {
            'n': 'Nano (2.6M params, 6.6 GFLOPs)',
            's': 'Small (9.4M params, 21.7 GFLOPs)',
            'm': 'Medium (20.1M params, 68.5 GFLOPs)',
            'l': 'Large (25.3M params, 87.6 GFLOPs)',
            'x': 'XLarge (56.9M params, 196.0 GFLOPs)'
        }[x]
    )
    
    scale_values = st.session_state.model_config['scales'][scale_type]
    st.info(f"""**Scale Parameters:**
- Depth: {scale_values[0]}
- Width: {scale_values[1]}
- Max Channels: {scale_values[2]}""")
    
    st.divider()
    
    if st.button("📥 Load YOLOv11n Template", use_container_width=True):
        template = create_yolo11_template()
        st.session_state.layers = []
        
        for layer in template['backbone']:
            st.session_state.layers.append({
                'from': layer[0],
                'number': layer[1],
                'type': layer[2],
                'args': layer[3]
            })
        
        for layer in template['head']:
            st.session_state.layers.append({
                'from': layer[0],
                'number': layer[1],
                'type': layer[2],
                'args': layer[3]
            })
        
        st.success("✅ YOLOv11n template loaded!")
        st.rerun()
    
    if st.button("🗑️ Clear All Layers", use_container_width=True):
        st.session_state.layers = []
        st.rerun()

# 메인 영역
tab1, tab2, tab3 = st.tabs(["📝 Layer Editor", "🔍 Network Viewer", "💾 Export"])

with tab1:
    st.header("Add New Layer")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        layer_type = st.selectbox(
            "Layer Type",
            options=list(LAYER_TYPES.keys()),
            format_func=lambda x: f"{x}"
        )
        st.caption(LAYER_TYPES[layer_type])
    
    with col2:
        from_layer = st.text_input(
            "From Layer",
            value="-1",
            help="이전 레이어 인덱스 (-1: 직전 레이어, [4, 6]: 여러 레이어)"
        )
    
    with col3:
        number = st.number_input(
            "Number/Repeat",
            min_value=1,
            value=1,
            help="레이어 반복 횟수"
        )
    
    with col4:
        # 레이어 타입별 기본 args 제공
        default_args = {
            'Conv': '[64, 3, 2]',
            'C3k2': '[256, False, 0.25]',
            'C2PSA': '[1024]',
            'SPPF': '[1024, 5]',
            'nn.Upsample': '[None, 2, "nearest"]',
            'Concat': '[1]',
            'Detect': '["nc"]',
        }
        args_input = st.text_input(
            "Arguments",
            value=default_args.get(layer_type, '[64, 3, 2]'),
            help="레이어 파라미터 (리스트 형식)"
        )
    
    if st.button("➕ Add Layer", use_container_width=True):
        try:
            # from_layer 파싱
            if from_layer.startswith('['):
                from_parsed = json.loads(from_layer)
            else:
                from_parsed = int(from_layer)
            
            # args 파싱
            args = json.loads(args_input)
            
            add_layer(layer_type, from_parsed, number, args)
            st.success(f"✅ {layer_type} layer added!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    st.header("Current Layers")
    
    if not st.session_state.layers:
        st.info("No layers added yet. Add layers or load a template.")
    else:
        for idx, layer in enumerate(st.session_state.layers):
            # 레이어 섹션 구분
            section = "🔷 Backbone" if layer['type'] not in ['Detect', 'nn.Upsample'] or idx > 10 else "🔶 Head"
            if idx > 0 and st.session_state.layers[idx-1]['type'] not in ['Detect', 'nn.Upsample'] and layer['type'] in ['Detect', 'nn.Upsample']:
                st.markdown("---")
                st.subheader("🔶 Head Section")
            
            with st.expander(f"Layer {idx}: {layer['type']} - {layer['args']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**From:** `{layer['from']}`")
                    st.write(f"**Number:** `{layer['number']}`")
                    st.write(f"**Type:** `{layer['type']}`")
                    st.write(f"**Args:** `{layer['args']}`")
                
                with col2:
                    if st.button("🗑️", key=f"del_{idx}", help="Delete layer"):
                        remove_layer(idx)
                        st.rerun()

with tab2:
    st.header("Network Structure Viewer")
    
    if not st.session_state.layers:
        st.info("No network structure to display. Add layers first.")
    else:
        # Backbone과 Head 구분
        backbone_layers = []
        head_layers = []
        in_head = False
        
        for idx, layer in enumerate(st.session_state.layers):
            if layer['type'] in ['Detect', 'nn.Upsample'] and not in_head:
                in_head = True
            
            if in_head:
                head_layers.append((idx, layer))
            else:
                backbone_layers.append((idx, layer))
        
        # Backbone 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔷 Backbone")
            for idx, layer in backbone_layers:
                layer_str = f"[{layer['from']}, {layer['number']}, {layer['type']}, {layer['args']}]"
                st.code(f"{idx}: {layer_str}", language="python")
        
        with col2:
            st.subheader("🔶 Head")
            if head_layers:
                for idx, layer in head_layers:
                    layer_str = f"[{layer['from']}, {layer['number']}, {layer['type']}, {layer['args']}]"
                    st.code(f"{idx}: {layer_str}", language="python")
            else:
                st.info("No head layers defined")
        
        st.divider()
        
        # 통계
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Layers", len(st.session_state.layers))
        col2.metric("Backbone Layers", len(backbone_layers))
        col3.metric("Head Layers", len(head_layers))

with tab3:
    st.header("Export Model Configuration")
    
    if not st.session_state.layers:
        st.warning("⚠️ No layers to export. Add layers first.")
    else:
        filename = st.text_input(
            "Model Name",
            value="yolov11_custom",
            help="Enter filename for the model (without extension)"
        )
        
        export_format = st.radio(
            "Export Format",
            options=["YAML only", "YAML + PT model"],
            help="PT 모델은 ultralytics 패키지가 필요합니다"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate YAML", use_container_width=True):
                try:
                    yaml_content = save_model_config(f"{filename}.yaml")
                    
                    st.success("✅ YAML configuration generated!")
                    
                    st.subheader("Preview")
                    st.code(yaml_content, language="yaml")
                    
                    st.download_button(
                        label="💾 Download YAML",
                        data=yaml_content,
                        file_name=f"{filename}.yaml",
                        mime="text/yaml",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating YAML: {str(e)}")
        
        with col2:
            if export_format == "YAML + PT model":
                if st.button("🏗️ Build & Export PT Model", use_container_width=True):
                    with st.spinner("Building PyTorch model... This may take a moment."):
                        try:
                            yaml_content = save_model_config(f"{filename}.yaml")
                            pt_data, error = build_pt_model(yaml_content, filename)
                            
                            if error:
                                st.error(f"❌ {error}")
                                st.info("💡 YAML 파일만 다운로드하고, 별도로 모델을 빌드할 수 있습니다.")
                            else:
                                st.success("✅ PT model built successfully!")
                                
                                col_yaml, col_pt = st.columns(2)
                                
                                with col_yaml:
                                    st.download_button(
                                        label="💾 Download YAML",
                                        data=yaml_content,
                                        file_name=f"{filename}.yaml",
                                        mime="text/yaml",
                                        use_container_width=True
                                    )
                                
                                with col_pt:
                                    st.download_button(
                                        label="💾 Download PT Model",
                                        data=pt_data,
                                        file_name=f"{filename}.pt",
                                        mime="application/octet-stream",
                                        use_container_width=True
                                    )
                                
                        except Exception as e:
                            st.error(f"❌ Error building model: {str(e)}")
        
        st.divider()
        
        st.subheader("📚 Usage Instructions")
        
        tab_yaml, tab_pt, tab_train = st.tabs(["Using YAML", "Using PT Model", "Training"])
        
        with tab_yaml:
            st.markdown("""
            ### YAML 파일을 사용한 모델 생성
            
            ```python
            from ultralytics import YOLO
            
            # YAML 설정 파일로부터 모델 생성
            model = YOLO('yolov11_custom.yaml')
            
            # 모델 학습
            results = model.train(
                data='coco128.yaml',  # 데이터셋 설정
                epochs=100,
                imgsz=640,
                batch=16
            )
            ```
            
            ✅ YAML 파일은 네트워크 구조만 정의하며, 학습 시 가중치가 랜덤 초기화됩니다.
            """)
        
        with tab_pt:
            st.markdown("""
            ### PT 모델 파일 사용
            
            ```python
            from ultralytics import YOLO
            
            # 빌드된 PT 모델 로드 (초기화된 가중치 포함)
            model = YOLO('yolov11_custom.pt')
            
            # 바로 학습 가능
            results = model.train(
                data='coco128.yaml',
                epochs=100,
                imgsz=640
            )
            
            # 또는 예측 (학습 후)
            results = model.predict('image.jpg')
            ```
            
            ✅ PT 파일은 네트워크 구조 + 초기화된 가중치를 포함합니다.
            """)
        
        with tab_train:
            st.markdown("""
            ### 커스텀 데이터셋으로 학습하기
            
            1. **데이터셋 YAML 생성** (`data.yaml`):
            ```yaml
            path: /path/to/dataset
            train: images/train
            val: images/val
            
            nc: 3  # number of classes
            names: ['class1', 'class2', 'class3']
            ```
            
            2. **학습 실행**:
            ```python
            from ultralytics import YOLO
            
            model = YOLO('yolov11_custom.yaml')
            
            results = model.train(
                data='data.yaml',
                epochs=100,
                imgsz=640,
                batch=16,
                device=0,  # GPU 번호
                workers=8,
                patience=50
            )
            ```
            
            3. **검증 및 예측**:
            ```python
            # 검증
            metrics = model.val()
            
            # 예측
            results = model.predict('test.jpg', save=True)
            ```
            """)
        
        st.info("""
        💡 **Requirements**: 
        - `pip install ultralytics`
        - PyTorch 설치 필요
        - CUDA (GPU 학습 시)
        """)

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>YOLOv11 Network Builder</strong> | Built with Streamlit</p>
    <p>📝 Based on Ultralytics YOLOv11 Architecture</p>
    <p>⚠️ Validate your network architecture before training</p>
</div>
""", unsafe_allow_html=True)