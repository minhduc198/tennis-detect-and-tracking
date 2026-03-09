import numpy as np


class DetectionMetrics:
    def __init__(self):
        self.total_detections = 0
        self.frames = 0

    def update(self, detections):
     
        self.total_detections += len(detections)
        self.frames += 1

    def summary(self):
        avg_det = 0
        if self.frames > 0:
            avg_det = self.total_detections / self.frames

        return {
            "frames_processed": self.frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": round(avg_det, 2),
        }


class TrackingMetrics:
    def __init__(self):
        self.track_ids = set()
        self.id_switches = 0
        self.prev_ids = set()

    def update(self, tracked_objects):
       

        current_ids = set()

        for obj in tracked_objects:
            track_id = int(obj[4])
            current_ids.add(track_id)
            self.track_ids.add(track_id)

        if self.prev_ids and current_ids != self.prev_ids:
            self.id_switches += 1

        self.prev_ids = current_ids

    def summary(self):
        return {
            "total_unique_players": len(self.track_ids),
            "id_switches": self.id_switches
        }