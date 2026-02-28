from ultralytics import YOLO
import cv2
import numpy as np



class PlayerDetector:
    def __init__(self, model_path, conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.prev_gray = None

    def detect(self, frame):
        H, W = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_mask = None

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        self.prev_gray = gray

        results = self.model(frame, imgsz=512, verbose=False)

        candidates = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id != 0 or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                h = y2 - y1

                if h < 0.05 * H:
                    continue
                if x1 < 0.1 * W or x2 > 0.9 * W:
                    continue                    
                if motion_mask is not None:
                    roi = motion_mask[y1:y2, x1:x2]
                    motion_score = np.sum(roi) / 255

                    if motion_score < 50:
                        continue

                candidates.append([x1, y1, x2, y2, conf, h])

        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda x: x[5], reverse=True)
        selected = candidates[:2]

        detections = []
        for x1, y1, x2, y2, conf, _ in selected:
            detections.append([x1, y1, x2, y2, conf])

        return detections
