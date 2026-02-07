import cv2
from src.camera import Camera
from src.detector import FaceDetector
import time

def main():
    print("Initializing Camera...")
    cam = Camera(source=0)
    cam.start()

    print("Initializing Face Detector...")
    detector = FaceDetector()

    print("Starting Main Loop. Press 'q' to quit.")
    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                # Detect faces
                boxes, probs = detector.detect(frame)
                
                # Draw boxes
                frame = detector.draw_boxes(frame, boxes, probs)

                cv2.imshow("Face Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up...")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
