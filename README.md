# Face Recognition Dashboard (얼굴 인식 대시보드)

이 프로젝트는 Python, OpenCV, Streamlit, 그리고 Facenet-PyTorch를 이용한 실시간 얼굴 인식 및 신원 확인 웹 애플리케이션입니다.

## 🌟 주요 기능
- **실시간 얼굴 탐지**: 웹캠을 통해 실시간으로 얼굴을 탐지합니다.
- **인공지능 얼굴 인식**: 사전에 딥러닝 모델(`InceptionResnetV1`)을 사용하여 얼굴 특징을 분석하고 누구인지 식별합니다.
- **얼굴 등록 및 관리**:
    - 사이드바에서 이름을 입력하고 사진을 찍어 간편하게 얼굴을 등록할 수 있습니다.
    - 등록된 얼굴 데이터를 관리하고 삭제할 수 있습니다.
- **한국어 지원**: 모든 UI가 직관적인 한국어로 제공됩니다.

## 🛠️ 설치 방법

이 프로젝트를 로컬 환경에서 실행하려면 다음 단계(Windows 기준)를 따르세요.

### 1. 필수 프로그램 설치
- [Python 3.10 이상](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

### 2. 프로젝트 클론 (다운로드)
```bash
git clone https://github.com/Iamhso/face.git
cd face
```

### 3. 가상환경 생성 및 활성화 (권장)
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 4. 라이브러리 설치
```bash
pip install -r requirements.txt
```
*(PyTorch 용량이 커서 설치에 시간이 조금 걸릴 수 있습니다.)*

## 🚀 실행 방법

설치가 완료되면 아래 명령어로 대시보드를 실행하세요.

```bash
streamlit run src/main.py
```
실행 후 브라우저가 자동으로 열리며 `http://localhost:8501`로 접속됩니다.

## 💡 사용 가이드

1.  **카메라 시작**: 메인 화면의 "카메라 시작" 박스를 체크하세요.
2.  **얼굴 인식**: 카메라에 얼굴을 비추면 녹색 박스와 함께 인식된 이름이 뜹니다. (처음엔 `Unknown`으로 뜸)
3.  **얼굴 등록**:
    - 왼쪽 사이드바의 **"새 얼굴 등록"** 섹션으로 이동합니다.
    - 이름을 입력하고, **카메라에 한 명의 얼굴만 나오게 한 뒤** `[얼굴 등록]` 버튼을 누르세요.
    - 등록 완료 메시지가 뜨면 이제부터 해당 이름으로 인식됩니다.
4.  **얼굴 삭제**:
    - 사이드바의 **"등록된 얼굴 관리"** 섹션에서 삭제할 이름을 선택하세오.
    - `[삭제]` 버튼을 누르고, 확인 메시지에서 `[✔️ 예]`를 클릭하면 삭제됩니다.

## ⚠️ 트러블슈팅

- **카메라가 안 켜져요**: 다른 프로그램(Zoom, Discord 등)이 카메라를 사용 중인지 확인하고 꺼주세요.
- **한글이 깨져요**: 윈도우 기본 폰트(`malgun.ttf`)를 사용하므로 윈도우 외 환경에서는 `src/detector.py`의 폰트 경로 수정이 필요할 수 있습니다.
- **`AttributeError` 오류**: 서버가 이전 코드를 기억하고 있어서 그렇습니다. 터미널에서 `Ctrl+C`로 서버를 끄고 다시 시작하세요.

## 📂 프로젝트 구조
```
face/
├── src/
│   ├── main.py         # 메인 웹 애플리케이션 (Streamlit)
│   ├── camera.py       # 카메라 제어 모듈
│   ├── detector.py     # 얼굴 탐지 및 인식 모델
│   ├── face_manager.py # 얼굴 데이터 저장/관리
│   └── utils.py        # 유틸리티 함수
├── data/               # 등록된 얼굴 데이터 (자동 생성)
├── requirements.txt    # 필요 라이브러리 목록
└── README.md           # 설명서
```
