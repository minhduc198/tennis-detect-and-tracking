import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):

        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)


        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        """Dự đoán vị trí tiếp theo"""
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])

    def update(self, x, y):
        """Cập nhật trạng thái dựa trên tọa độ thực tế từ Detection"""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)