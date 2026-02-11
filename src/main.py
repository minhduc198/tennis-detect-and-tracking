# src/main.py

import cv2
from detection.detect_player import PlayerDetector


VIDEO_PATH = "data/raw_video/tennis_input.mp4"
MODEL_PATH = "models/detection/yolo_player.pt"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    detector = PlayerDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.4
    )

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        # DEBUG visualization (tạm thời – tracking mnguoi sẽ xử lý sau)
        for det in detections:
            x1, y1, x2, y2, conf = det
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Player {conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.imshow("Detection - Player", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
