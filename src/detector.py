from facenet_pytorch import MTCNN, InceptionResnetV1
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
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=False, device=self.device).eval()
        print("Face Detector and Recognizer loaded.")

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

    def draw_boxes(self, frame, boxes, probs=None, names=None):
        if boxes is None:
            return frame
        
        # Convert to PIL Image for drawing text
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            # Use Malgun Gothic for Korean support on Windows
            font_path = "malgun.ttf"
            font = ImageFont.truetype(font_path, 20)
        except IOError:
            try:
                 # Try absolute path for Windows
                font_path = "C:/Windows/Fonts/malgun.ttf"
                font = ImageFont.truetype(font_path, 20)
            except IOError:
                # Fallback to default if font not found
                print("Warning: Malgun Gothic font not found. Falling back to default.")
                font = ImageFont.load_default()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Draw rectangle using OpenCV (faster/easier for simple shapes) -> moved below to keep logic simple
            # Actually, doing everything in PIL or mixing? 
            # Let's mix: Draw rect on CV2 frame, then draw text on PIL, then convert back.
            # Efficient implementation: Draw all rectangles on 'frame' first.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = ""
            if names is not None and i < len(names):
                label += f"{names[i]} "
            if probs is not None:
                label += f"({probs[i]:.2f})"
                
            if label:
                # Draw text using PIL
                # Coordinate adjustment for text background if needed, but keeping it simple
                draw.text((x1, y1 - 25), label, font=font, fill=(0, 255, 0))

        # Convert back to BGR numpy array
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return frame
        
    def get_embeddings(self, frame, boxes):
        if boxes is None or len(boxes) == 0:
            return None
        
        faces = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, _ = frame_rgb.shape

        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            face = frame_rgb[y1:y2, x1:x2]
            
            if face.size == 0:
                continue

            try:
                face = cv2.resize(face, (160, 160))
            except Exception as e:
                continue
            
            # Standardize
            face = np.float32(face)
            mean, std = face.mean(), face.std()
            if std == 0: std = 1
            face = (face - mean) / std
            
            face = face.transpose(2, 0, 1)
            faces.append(face)
            
        if not faces:
            return None
            
        faces_array = np.array(faces)
        faces_tensor = torch.tensor(faces_array).to(self.device)
        
        try:
            with torch.no_grad():
                embeddings = self.resnet(faces_tensor)
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None

    def compute_distance(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)
