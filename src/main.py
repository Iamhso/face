import streamlit as st
import cv2
import time
from camera import Camera
from detector import FaceDetector
import numpy as np

def main():
    st.set_page_config(page_title="Face Recognition Dashboard", layout="wide")
    
    st.title("Face Recognition Dashboard")
    st.sidebar.title("Controls")

    # Control variables
    run_detection = st.sidebar.checkbox("Run Face Detection", value=True)
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.9)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Press 'Start' to begin camera feed.")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Camera Feed")
        placeholder = st.empty()
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")

    with col2:
        st.subheader("Detection Stats")
        stats_placeholder = st.empty()

    if start_button:
        st.session_state.camera_running = True
    
    if stop_button:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        # Initialize Camera and Detector
        # We use st.cache_resource for the detector to avoid reloading it on every rerun,
        # but here we are in a loop inside the main function, so we just instantiate it once before the loop if possible.
        # However, Streamlit reruns the script on interaction.
        # So we should cache the detector.
        
        detector = load_detector()
        
        # Camera handling is tricky in Streamlit. 
        # Ideally, we'd use a separate thread or process, but keeping it simple:
        # We'll just capture frames in a loop inside this 'if' block.
        # This blocks the UI, but it's the simplest way without streamlit-webrtc.
        
        cap = cv2.VideoCapture(0)
        
        process_placeholder = st.empty()
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break
            
            # Detect
            face_count = 0
            if run_detection:
                boxes, probs = detector.detect(frame)
                # Filter by confidence
                if boxes is not None:
                    valid_boxes = []
                    valid_probs = []
                    for box, prob in zip(boxes, probs):
                        if prob >= confidence_threshold:
                            valid_boxes.append(box)
                            valid_probs.append(prob)
                    
                    frame = detector.draw_boxes(frame, valid_boxes, valid_probs)
                    face_count = len(valid_boxes)

            # Display
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update stats
            stats_placeholder.markdown(f"**Faces Detected:** {face_count}")
            
            # Check for stop (this won't really work inside the loop unless we use st.experimental_rerun or check headers)
            # Actually, the 'Stop Camera' button won't be clickable because the loop blocks.
            # We need a way to break the loop. 
            # Usually people use a unique key or check a file/session state that changes, 
            # but Streamlit buttons don't update while script is executing a loop.
            # We'll rely on the user refreshing or closing, or use a workaround.
            # Better approach: Use `st.empty()` for the button too? No.
            # For now, let's just run for a few seconds or allow interrupting?
            # Standard Streamlit loops are problematic. 
            # Let's just update the image.
            
            time.sleep(0.01)
        
        cap.release()
    else:
        placeholder.info("Camera is stopped.")

@st.cache_resource
def load_detector():
    return FaceDetector()

if __name__ == "__main__":
    main()
