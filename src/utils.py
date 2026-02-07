import cv2
import datetime
import os

def resize_frame(frame, width=None, height=None):
    """
    Resizes a frame while maintaining aspect ratio.
    """
    if width is None and height is None:
        return frame

    (h, w) = frame.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def save_snapshot(frame, output_dir="snapshots"):
    """
    Saves a frame to disk with a timestamp.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename
