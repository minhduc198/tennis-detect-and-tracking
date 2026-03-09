import cv2
import os

from detection.detect_player import PlayerDetector
from tracking.sort_tracker import SortTracker

from visualization.draw_bbox_trackid import draw_bbox_with_trackid

from evaluation.metrics import DetectionMetrics, TrackingMetrics
from evaluation.analysis import plot_detection_stats, plot_tracking_stats


VIDEO_PATH = "data/raw_video/tennis_input6.mp4"
MODEL_PATH = "models/detection/yolo_player.pt"
OUTPUT_VIDEO = "data/processed_video/tennis_output_tracked.mp4"


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
    det_metrics = DetectionMetrics()
    track_metrics = TrackingMetrics()

    frame_count = 0
    saved_sample = False

    print("Processing video...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        detections = detector.detect(frame)

        det_metrics.update(detections)


        tracked_players = tracker.update(detections)

        track_metrics.update(tracked_players)


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



    det_summary = det_metrics.summary()
    track_summary = track_metrics.summary()

    print("\n===== Detection Metrics =====")
    print(det_summary)

    print("\n===== Tracking Metrics =====")
    print(track_summary)

  

    plot_detection_stats(det_summary)
    plot_tracking_stats(track_summary)

    print("\nEvaluation images saved in /evaluation_results/")
    print("Output video saved:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()