import cv2
import numpy as np

class VideoWriter:
    
    def __init__(self, output_path: str, fps: float, width: int, height: int, 
                 codec: str = "mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for: {output_path}")
    
    def write_frame(self, frame: np.ndarray):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()