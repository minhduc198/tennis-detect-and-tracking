import cv2
from typing import Generator, Tuple
import numpy as np

class VideoReader:
    """
    Reads video frame-by-frame using OpenCV VideoCapture.
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator yielding (frame_idx, frame) for each frame of the video.
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    
    def get_properties(self) -> dict:
        """Returns video properties."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }
    
    def release(self):
        """Releases the VideoCapture resource."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()