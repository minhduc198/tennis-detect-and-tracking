import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import KalmanFilter

class SortTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.trackers = {}  
        self.next_id = 1

    def update(self, detections):
        """
        Input: Detections [[x1, y1, x2, y2, conf], ...]
        Output: Kết quả đã gán ID [[x1, y1, x2, y2, track_id], ...]
        """
           
        predicted_tracks = {}
        for tid, data in self.trackers.items():
            pred_x, pred_y = data['kf'].predict()
  
            w = data['bbox'][2] - data['bbox'][0]
            h = data['bbox'][3] - data['bbox'][1]
            predicted_tracks[tid] = [pred_x, pred_y, pred_x + w, pred_y + h]

        final_results = []
        if len(detections) > 0 and len(predicted_tracks) > 0:
            track_ids = list(predicted_tracks.keys())
            pred_boxes = list(predicted_tracks.values())

            iou_matrix = np.zeros((len(detections), len(pred_boxes)))
            for i, det in enumerate(detections):
                for j, pred in enumerate(pred_boxes):
                    iou_matrix[i, j] = self._get_iou(det[:4], pred)

            row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)

            matched_det_indices = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    tid = track_ids[c]
                    bbox = detections[r][:4]
        
                    cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                    self.trackers[tid]['kf'].update(cx, cy)
                    self.trackers[tid]['bbox'] = bbox
                    
                    final_results.append(list(bbox) + [tid])
                    matched_det_indices.add(r)

            for i, det in enumerate(detections):
                if i not in matched_det_indices:
                    self._add_new_track(det[:4], final_results)
        else:

            for det in detections:
                self._add_new_track(det[:4], final_results)

        return final_results

    def _add_new_track(self, bbox, results_list):
        kf = KalmanFilter()
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        kf.update(cx, cy)
        self.trackers[self.next_id] = {'kf': kf, 'bbox': bbox}
        results_list.append(list(bbox) + [self.next_id])
        self.next_id += 1

    def _get_iou(self, box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter / float(area1 + area2 - inter + 1e-6)