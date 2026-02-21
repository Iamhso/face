import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
import cv2
import threading
import time
import numpy as np
from detector import FaceDetector
from face_manager import FaceManager

# Define RTC Configuration (STUN server is needed for cloud deployment)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = None
        self.face_manager = None
        
        # State for registration
        self.register_name = None
        self.should_register = False
        self.registration_result = None
        self.lock = threading.Lock()
        
        # Settings
        self.confidence_threshold = 0.9
        self.recognition_threshold = 0.8
        self.run_detection = True

    def initialize_resources(self, detector, face_manager):
        self.detector = detector
        self.face_manager = face_manager

    def update_settings(self, run_detection, confidence, recognition):
        self.run_detection = run_detection
        self.confidence_threshold = confidence
        self.recognition_threshold = recognition

    def trigger_registration(self, name):
        with self.lock:
            self.register_name = name
            self.should_register = True
            self.registration_result = None

    def get_registration_result(self):
        with self.lock:
            if self.registration_result:
                res = self.registration_result
                self.registration_result = None # Clear after read
                return res
            return None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # We need to initialize resources inside the process if not passed, 
        # but here we rely on passing them or loading them.
        # Ideally, load them once.
        # Since recv runs in a separate thread, we can't easily pass Streamlit cache objects directly 
        # if they are not thread-safe, but FaceDetector is mostly read-only after init.
        # Actually, creating detector inside __init__ of processor is safer for threading.
        
        if self.detector is None or self.face_manager is None:
            # If not initialized yet, skip processing to avoid race conditions or duplicate loads
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Detection Logic
        face_count = 0
        names = []
        boxes = None
        probs = None

        if self.run_detection:
            boxes, probs = self.detector.detect(img)
            
            if boxes is not None:
                # Filter
                valid_indices = [i for i, p in enumerate(probs) if p >= self.confidence_threshold]
                if len(valid_indices) > 0:
                    boxes = boxes[valid_indices]
                    probs = probs[valid_indices]
                    face_count = len(boxes)
                    
                    embeddings = self.detector.get_embeddings(img, boxes)
                    
                    if embeddings is not None:
                        for i, emb in enumerate(embeddings):
                            name, dist = self.face_manager.match_face(emb, threshold=self.recognition_threshold)
                            names.append(f"{name} ({dist:.2f})")
                        
                        # Registration Logic
                        with self.lock:
                            if self.should_register and self.register_name:
                                if len(embeddings) == 1:
                                    self.face_manager.add_face(self.register_name, embeddings[0])
                                    self.registration_result = f"SUCCESS:{self.register_name}"
                                    self.should_register = False
                                    self.register_name = None
                                elif len(embeddings) > 1:
                                    self.registration_result = "ERROR:Too many faces"
                                    self.should_register = False # One try per click
                                else:
                                    # No face found (unexpected here)
                                    pass

        # Draw
        img = self.detector.draw_boxes(img, boxes, probs, names)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="얼굴 인식 대시보드 (WebRTC)", layout="wide")
    st.title("얼굴 인식 대시보드 (WebRTC)")
    
    # ... Sidebar controls similar to before ...
    st.sidebar.title("제어판")
    
    run_detection = st.sidebar.checkbox("얼굴 인식 실행", value=True)
    confidence_threshold = st.sidebar.slider("탐지 정확도 임계값", 0.0, 1.0, 0.9)
    recognition_threshold = st.sidebar.slider("인식 거리 임계값", 0.0, 1.5, 0.8)

    st.sidebar.markdown("---")
    st.sidebar.subheader("새 얼굴 등록")
    with st.sidebar.form("register_form", clear_on_submit=True):
        new_name = st.text_input("이름 입력")
        register_button = st.form_submit_button("얼굴 등록")
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("등록된 얼굴 관리")
    
    fm = load_face_manager()
    registered_names = list(fm.faces.keys())
    
    # ... Deletion Logic same as before ...
    if registered_names:
        options = ["(선택 없음)"] + registered_names
        selected_name = st.sidebar.selectbox("삭제할 이름 선택", options)
        if selected_name != "(선택 없음)":
             if st.sidebar.button("선택한 얼굴 삭제", key="delete_confirm_btn"):
                 if fm.delete_face(selected_name):
                     st.sidebar.success(f"{selected_name} 삭제 완료!")
                     time.sleep(0.5)
                     st.rerun()

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("실시간 카메라")
        
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="face-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=True,
        )
        
        # Communicate with Processor
        if ctx.video_processor:
            # Pass shared resources to processor
            ctx.video_processor.initialize_resources(load_detector_v2(), load_face_manager())
            # Update settings
            ctx.video_processor.update_settings(run_detection, confidence_threshold, recognition_threshold)
            
            # Retrieve FaceManager/Detector if not initialized? 
            # Actually we can't easily pass local objects to the processor if it runs in a thread 
            # and we want to rely on the processor's internal state.
            # But registered faces need to be synced. Use a shared loading mechanism.
            
            # Handle Registration
            if register_button and new_name:
                ctx.video_processor.trigger_registration(new_name)
                with st.spinner("얼굴을 등록 중입니다. 카메라를 정면으로 잠시만 바라봐 주세요..."):
                    # 최대 5초간 결과를 기다림 (0.5초 간격으로 10번 확인)
                    for _ in range(10):
                        time.sleep(0.5)
                        res = ctx.video_processor.get_registration_result()
                        if res:
                            if res.startswith("SUCCESS"):
                                st.success(f"{res.split(':')[1]} 등록 성공!")
                                time.sleep(1)
                                st.rerun()
                            elif res.startswith("ERROR"):
                                st.error(f"등록 실패: {res.split(':')[1]}")
                                break
                    else:
                        st.warning("등록 시간이 초과되었습니다. 다시 시도해 주세요.")

    with col2:
        st.subheader("안내")
        st.info("WebRTC 모드는 서버가 아닌 사용자의 브라우저를 통해 카메라를 연결합니다. 따라서 배포 환경에서도 작동합니다.")
        st.warning("⚠️ 주의: `localhost` 또는 `127.0.0.1`이 아닌 주소(예: 내부 IP)로 접속 시, HTTPS 보안 연결이 없으면 브라우저가 카메라 접근을 차단할 수 있습니다.")

@st.cache_resource
def load_detector_v2():
    return FaceDetector()

@st.cache_resource
def load_face_manager():
    return FaceManager()

if __name__ == "__main__":
    main()
