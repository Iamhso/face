import cv2
import threading
import time

class Camera:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.frame = None
        self.last_access = time.time()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        self.cap = cv2.VideoCapture(self.source)
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.last_access = time.time()
            else:
                # Connection lost, try to reconnect
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
        
        # Cleanup
        if self.cap:
            self.cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
