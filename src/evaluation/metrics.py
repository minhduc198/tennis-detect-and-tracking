import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    if boxAArea + boxBArea - interArea == 0:
        return 0.0

    return interArea / float(boxAArea + boxBArea - interArea)


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = iou(gt, pred[:4])

    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_gt = []
    unmatched_pred = []

    for i in range(len(gt_boxes)):
        if i not in gt_idx:
            unmatched_gt.append(i)

    for j in range(len(pred_boxes)):
        if j not in pred_idx:
            unmatched_pred.append(j)

    for g, p in zip(gt_idx, pred_idx):
        if iou_matrix[g, p] >= iou_threshold:
            matches.append((g, p))
        else:
            unmatched_gt.append(g)
            unmatched_pred.append(p)

    return matches, unmatched_gt, unmatched_pred


def detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):

    matches, unmatched_gt, unmatched_pred = match_boxes(
        gt_boxes, pred_boxes, iou_threshold
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }



def normalize_track(obj):

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, list):

        if len(obj) >= 5:
            return {
                "bbox": obj[:4],
                "id": int(obj[4])
            }

        if len(obj) == 4:
            return {
                "bbox": obj,
                "id": -1
            }

    return None


def tracking_metrics(gt_tracks, pred_tracks, iou_threshold=0.5):

    total_gt = 0
    misses = 0
    false_positives = 0
    id_switches = 0

    iou_sum = 0.0
    iou_count = 0

    prev_assignment = {}

    for frame, gt_list in gt_tracks.items():

        pred_list = pred_tracks.get(frame, [])

        gt_list = [normalize_track(g) for g in gt_list if normalize_track(g)]
        pred_list = [normalize_track(p) for p in pred_list if normalize_track(p)]

        total_gt += len(gt_list)

        gt_boxes = [g["bbox"] for g in gt_list]
        pred_boxes = [p["bbox"] for p in pred_list]

        matches, unmatched_gt, unmatched_pred = match_boxes(
            gt_boxes, pred_boxes, iou_threshold
        )

        misses += len(unmatched_gt)
        false_positives += len(unmatched_pred)

        for g_idx, p_idx in matches:

            iou_sum += iou(gt_boxes[g_idx], pred_boxes[p_idx])
            iou_count += 1

            gt_id = gt_list[g_idx]["id"]
            pred_id = pred_list[p_idx]["id"]

            if gt_id in prev_assignment and prev_assignment[gt_id] != pred_id:
                id_switches += 1

            prev_assignment[gt_id] = pred_id

    mota = (
        1.0 - (misses + false_positives + id_switches) / total_gt
        if total_gt > 0
        else 0.0
    )

    motp = iou_sum / iou_count if iou_count > 0 else 0.0

    return {
        "MOTA": mota,
        "MOTP": motp,
        "id_switches": id_switches,
        "false_positives": false_positives,
        "misses": misses,
        "total_gt": total_gt,
    }