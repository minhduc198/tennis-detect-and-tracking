import math
from .kalman import KalmanFilter

class SortTracker:
    def __init__(self, iou_threshold=0.3):
        self.max_age = 60 
        self.trackers = {
            1: {'kf': KalmanFilter(), 'bbox': None, 'active': False, 'missed': 0}, 
            2: {'kf': KalmanFilter(), 'bbox': None, 'active': False, 'missed': 0}  
        }

    def update(self, detections):
        valid_dets = [d for d in detections if len(d) >= 4]
        unique_dets = []
        for det in valid_dets:
            cx, cy = (det[0]+det[2])/2, (det[1]+det[3])/2
            is_dup = False
            for u in unique_dets:
                ux, uy = (u[0]+u[2])/2, (u[1]+u[3])/2
                if math.hypot(cx - ux, cy - uy) < 100:
                    is_dup = True
                    break
            if not is_dup:
                unique_dets.append(det)

        unique_dets.sort(key=lambda x: x[3], reverse=True)
        matched_pids = set()

        # 2. GÁN ID CỐ ĐỊNH (LUẬT THÉP)
        if len(unique_dets) >= 2:
            self._update_tracker(1, unique_dets[0][:4])
            self._update_tracker(2, unique_dets[-1][:4])
            matched_pids.update([1, 2])
            
        elif len(unique_dets) == 1:
            det = unique_dets[0]
            cx, cy = (det[0]+det[2])/2, (det[1]+det[3])/2
            best_pid = 1
            
            if self.trackers[1]['active'] and self.trackers[2]['active']:
                b1 = self.trackers[1]['bbox']
                d1 = math.hypot(cx - (b1[0]+b1[2])/2, cy - (b1[1]+b1[3])/2)
                
                b2 = self.trackers[2]['bbox']
                d2 = math.hypot(cx - (b2[0]+b2[2])/2, cy - (b2[1]+b2[3])/2)
                
                best_pid = 1 if d1 < d2 else 2
            elif self.trackers[2]['active']:
                best_pid = 2
                
            self._update_tracker(best_pid, det[:4])
            matched_pids.add(best_pid)

        final_results = []
        for pid in [1, 2]:
            if pid not in matched_pids and self.trackers[pid]['active']:
                self.trackers[pid]['missed'] += 1
                
                if self.trackers[pid]['missed'] < self.max_age:
                    pass 
                else:
                    self.trackers[pid]['active'] = False

            if self.trackers[pid]['active']:
                final_results.append(list(self.trackers[pid]['bbox']) + [pid])

        return final_results

    def _update_tracker(self, pid, bbox):
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2

        if not self.trackers[pid]['active'] or self.trackers[pid]['missed'] > 0:
            self.trackers[pid]['kf'] = KalmanFilter()
            
        self.trackers[pid]['kf'].predict()
        self.trackers[pid]['kf'].update(cx, cy)
        
        self.trackers[pid]['bbox'] = bbox
        self.trackers[pid]['active'] = True
        self.trackers[pid]['missed'] = 0