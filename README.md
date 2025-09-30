# NetPipe

**Network Customization Pipeline Development**  
NetPipe는 딥러닝 모델을 쉽게 생성, 검증, 학습할 수 있는 파이프라인입니다.  

---

## 기능 페이지

1. **MakeModel** – 모델 생성  
   - 원하는 네트워크 구조를 GUI에서 정의하고 생성할 수 있습니다.

2. **CheckModel** – 모델 구조 확인  
   - 생성된 모델의 레이어 구성, 파라미터 수 등을 시각적으로 확인합니다.

3. **LoadDatasets** – 데이터셋 검사  
   - 학습/검증 데이터셋을 불러와 확인하고 전처리 상태를 점검할 수 있습니다.

4. **TrainModel** – 모델 학습  
   - 설정된 하이퍼파라미터로 모델을 학습하고, 학습 로그와 결과를 확인할 수 있습니다.

---

## 사용법

1. GitHub 저장소 클론

```bash
git clone https://github.com/HenryParkG/NetPipe.git
cd NetPipe
python -m venv npvenv
# Windows
.\npvenv\Scripts\activate
# macOS / Linux
source npvenv/bin/activate
pip install -r requirements.txt
streamlit run main.py
