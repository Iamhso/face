import streamlit as st
import cv2
import time
from camera import Camera
from detector import FaceDetector
from face_manager import FaceManager
import numpy as np
import os

def main():
    st.set_page_config(page_title="얼굴 인식 대시보드", layout="wide")
    
    st.title("얼굴 인식 대시보드")
    st.sidebar.title("제어판")

    # Initialize Camera in Session State FIRST
    if "camera" not in st.session_state:
        st.session_state.camera = Camera(source=0)
    
    # Ensure camera starts if checkbox was previously checked or default
    # But we control it via checkbox below.
    
    # Control variables
    run_detection = st.sidebar.checkbox("얼굴 인식 실행", value=True)
    confidence_threshold = st.sidebar.slider("탐지 정확도 임계값", 0.0, 1.0, 0.9)
    recognition_threshold = st.sidebar.slider("인식 거리 임계값", 0.0, 1.5, 0.8)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("새 얼굴 등록")
    # Use a form to prevent rerun on every keystroke
    with st.sidebar.form("register_form", clear_on_submit=True):
        new_name = st.text_input("이름 입력")
        register_button = st.form_submit_button("얼굴 등록")

    st.sidebar.markdown("---")
    st.sidebar.subheader("등록된 얼굴 관리")
    
    fm = load_face_manager()
    registered_names = list(fm.faces.keys())
    
    # Session state for delete selection to handle updates properly
    if "delete_selected" not in st.session_state:
        st.session_state.delete_selected = "(선택 없음)"

    if registered_names:
        options = ["(선택 없음)"] + registered_names
        
        # Ensure selected option is valid
        if st.session_state.delete_selected not in options:
             st.session_state.delete_selected = "(선택 없음)"
             
        selected_name = st.sidebar.selectbox(
            "삭제할 이름 선택", 
            options, 
            index=options.index(st.session_state.delete_selected)
        )
        st.session_state.delete_selected = selected_name
        
        if selected_name != "(선택 없음)":
            if "delete_confirm" not in st.session_state:
                st.session_state.delete_confirm = None

            if st.sidebar.button("삭제", key="init_delete"):
                st.session_state.delete_confirm = selected_name
            
            if st.session_state.delete_confirm == selected_name:
                st.sidebar.error(f"정말 '{selected_name}'을(를) 삭제하시겠습니까?")
                d_col1, d_col2 = st.sidebar.columns(2)
                with d_col1:
                    if st.button("✔️ 예", key="confirm_delete"):
                        if fm.delete_face(selected_name):
                            st.toast(f"{selected_name} 삭제 완료!", icon="🗑️")
                            st.session_state.delete_confirm = None
                            st.session_state.delete_selected = "(선택 없음)" # Reset selection
                            time.sleep(0.5)
                            st.rerun()
                with d_col2:
                    if st.button("❌ 아니오", key="cancel_delete"):
                        st.session_state.delete_confirm = None
                        st.rerun()
    else:
        st.sidebar.info("등록된 얼굴이 없습니다.")

    st.sidebar.markdown("---")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("실시간 카메라화면")
        placeholder = st.empty()
        run_camera = st.checkbox("카메라 시작", value=False, key="run_camera_check")

    with col2:
        st.subheader("탐지 상태")
        stats_placeholder = st.empty()
        
    # Load resources
    detector = load_detector_v2()
    face_manager = load_face_manager() # This is cached, so it might return old object if we don't clear cache?
    # Actually, load_face_manager returns a new instance if not cached, but it is cached.
    # FaceManager handles file I/O on init. If we delete, we update the object status.
    # If we add, we update object status.
    # So the object in cache IS updated. The issue is likely just UI refresh.
    
    camera = st.session_state.camera

    if run_camera:
        if not camera.running:
            camera.start()
        
        registered_in_this_run = False
        
        # Main Loop
        while run_camera:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Application Logic
            face_count = 0
            names = []
            
            boxes = None
            probs = None
            
            if run_detection:
                boxes, probs = detector.detect(frame)
                
                if boxes is not None:
                    valid_indices = [i for i, p in enumerate(probs) if p >= confidence_threshold]
                    if len(valid_indices) > 0:
                        boxes = boxes[valid_indices]
                        probs = probs[valid_indices]
                        
                        face_count = len(boxes)
                        
                        embeddings = detector.get_embeddings(frame, boxes)
                        
                        if embeddings is not None:
                            for i, emb in enumerate(embeddings):
                                name, dist = face_manager.match_face(emb, threshold=recognition_threshold)
                                names.append(f"{name} ({dist:.2f})")
                            
                            if register_button and new_name and not registered_in_this_run:
                                if len(embeddings) == 1:
                                    face_manager.add_face(new_name, embeddings[0])
                                    st.toast(f"{new_name} 등록 완료!", icon="✅")
                                    registered_in_this_run = True
                                    # Force UI update to show new name in list
                                    time.sleep(1)
                                    st.rerun()
                                elif len(embeddings) > 1:
                                    st.toast("얼굴이 너무 많습니다! 한 명만 나오게 해주세요.", icon="⚠️")
                                    registered_in_this_run = True 
                                else:
                                    pass

            # Draw
            frame = detector.draw_boxes(frame, boxes, probs, names)

            # Display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", width="stretch")
            
            stats_placeholder.markdown(f"**탐지된 얼굴 수:** {face_count}\n\n**식별됨:** {', '.join(names)}")
            
            time.sleep(0.01) 
    else:
        if camera.running:
            camera.stop()
        placeholder.info("카메라가 꺼져 있습니다.")

@st.cache_resource
def load_detector_v2():
    return FaceDetector()

@st.cache_resource
def load_face_manager():
    return FaceManager()

if __name__ == "__main__":
    main()
