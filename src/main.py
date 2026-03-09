import cv2
import os
import json

from detection.detect_player import PlayerDetector
from tracking.sort_tracker import SortTracker
from visualization.draw_bbox_trackid import draw_bbox_with_trackid


VIDEO_PATH = "data/raw_video/tennis_input6.mp4"
MODEL_PATH = "models/detection/yolo_player.pt"
OUTPUT_VIDEO = "data/processed_video/tennis_output_tracked.mp4"

PRED_BOXES_PATH = "pred_boxes.json"
PRED_TRACKS_PATH = "pred_tracks.json"


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("data/processed_video", exist_ok=True)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    detector = PlayerDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.4
    )

    tracker = SortTracker(iou_threshold=0.3)

    frame_count = 0
    saved_sample = False

    pred_boxes = {}
    pred_tracks = {}

    print("Processing video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        detections = detector.detect(frame)

        pred_boxes[frame_count] = detections

        tracked_players = tracker.update(detections)

        pred_tracks[frame_count] = tracked_players

        vis_frame = draw_bbox_with_trackid(frame, tracked_players)

        if not saved_sample and len(tracked_players) > 0:

            os.makedirs("evaluation_results", exist_ok=True)

            cv2.imwrite(
                "evaluation_results/sample_tracking_frame.png",
                vis_frame
            )

            saved_sample = True

        cv2.imshow("Tennis Player Tracking", vis_frame)

        writer.write(vis_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Saving prediction results...")

    with open(PRED_BOXES_PATH, "w") as f:
        json.dump(pred_boxes, f)

    with open(PRED_TRACKS_PATH, "w") as f:
        json.dump(pred_tracks, f)

    print("Prediction files saved:")
    print(" - pred_boxes.json")
    print(" - pred_tracks.json")

    print("\nOutput video saved:", OUTPUT_VIDEO)
    print("Sample frame saved in /evaluation_results/")


if __name__ == "__main__":
    main()