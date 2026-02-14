import os
import pickle
import numpy as np

class FaceManager:
    def __init__(self, storage_file="data/faces.pkl"):
        self.storage_file = storage_file
        self.faces = {} # Name -> [Embedding1, Embedding2, ...]
        self._load_faces()

    def _load_faces(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'rb') as f:
                    self.faces = pickle.load(f)
                print(f"Loaded {len(self.faces)} identities.")
            except Exception as e:
                print(f"Error loading faces: {e}")
                self.faces = {}
        else:
            print("No existing face database found.")
            # Create directory if needed
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)

    def save_faces(self):
        try:
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            with open(self.storage_file, 'wb') as f:
                pickle.dump(self.faces, f)
            print("Faces saved successfully.")
        except Exception as e:
            print(f"Error saving faces: {e}")

    def add_face(self, name, embedding):
        if name not in self.faces:
            self.faces[name] = []
        self.faces[name].append(embedding)
        self.save_faces()

    def delete_face(self, name):
        if name in self.faces:
            del self.faces[name]
            self.save_faces()
            return True
        return False

    def match_face(self, embedding, threshold=0.8):
        """
        Finds the closest matching face.
        :param embedding: The embedding to match.
        :param threshold: Distance threshold (smaller is closer).
        :return: Name of the match or "Unknown".
        """
        best_name = "Unknown"
        best_dist = float('inf')

        for name, embeddings in self.faces.items():
            for known_emb in embeddings:
                dist = np.linalg.norm(embedding - known_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        
        if best_dist < threshold:
            return best_name, best_dist
        else:
            return "Unknown", best_dist
