
import cv2
from detection.detect_player import PlayerDetector
from tracking.sort_tracker import SortTracker

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

    tracker = SortTracker(iou_threshold=0.3)
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_players = tracker.update(detections)

        # DEBUG visualization (tạm thời – tracking mnguoi sẽ xử lý sau)
        for player in tracked_players:
            x1, y1, x2, y2, track_id = player
            
            # Tạo màu khác nhau cho từng ID
            color = (0, 255, 0) if track_id == 1 else (255, 0, 0)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                frame, 
                f"ID: {int(track_id)}", 
                (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Tennis Tracking - Core Module", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
