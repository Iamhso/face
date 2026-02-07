from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2

class FaceDetector:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Loading Face Detector on {self.device}...")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        print("Face Detector loaded.")

    def detect(self, frame):
        """
        Detects faces in a given frame.
        :param frame: numpy array (B, G, R) from OpenCV
        :return: boxes (list of bounding boxes), probabilities (list of probabilities)
        """
        try:
            # MTCNN expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = self.mtcnn.detect(frame_rgb)
            return boxes, probs
        except Exception as e:
            print(f"Error during detection: {e}")
            return None, None

    def draw_boxes(self, frame, boxes, probs=None):
        if boxes is None:
            return frame
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if probs is not None:
                cv2.putText(frame, f"{probs[i]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
