import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.evaluation.metrics import iou, detection_metrics, tracking_metrics
from src.evaluation.analysis import (plot_boxes, compute_blur_score,find_blurry_frames, detect_occlusion)


def load_coco_gt(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    gt_data = {}

    for ann in data['annotations']:
        frame_idx = ann['image_id'] - 1          

        x, y, w, h = ann['bbox']                 
        x1, y1, x2, y2 = x, y, x + w, y + h     

        track_id = ann['attributes']['track_id'] + 1

        if frame_idx not in gt_data:
            gt_data[frame_idx] = {'boxes': [], 'tracks': []}

        gt_data[frame_idx]['boxes'].append([x1, y1, x2, y2])
        gt_data[frame_idx]['tracks'].append({
            'bbox': [x1, y1, x2, y2],
            'id':   track_id
        })

    return gt_data


def load_predictions(pred_boxes_path: str, pred_tracks_path: str):

    with open(pred_boxes_path) as f:
        raw_boxes = json.load(f)

    with open(pred_tracks_path) as f:
        raw_tracks = json.load(f)

    pred_boxes = {int(k): v for k, v in raw_boxes.items()}

    pred_tracks = {}

    for frame, tracks in raw_tracks.items():

        frame = int(frame)
        converted = []

        for t in tracks:

            if isinstance(t, list):

                if len(t) >= 5:
                    converted.append({
                        "bbox": t[:4],
                        "id": int(t[4])
                    })

                elif len(t) == 4:
                    converted.append({
                        "bbox": t,
                        "id": -1
                    })

            elif isinstance(t, dict):
                converted.append(t)

        pred_tracks[frame] = converted

    return pred_boxes, pred_tracks



def run_detection_evaluation(gt_data: dict, pred_boxes: dict) -> dict:
  
    print("=" * 60)
    print("DETECTION METRICS")
    print("=" * 60)

    results = []

    for frame_idx in sorted(gt_data.keys()):
        gt_boxes = gt_data[frame_idx]['boxes']

        preds = pred_boxes.get(frame_idx, [])

        dm = detection_metrics(gt_boxes, preds, iou_threshold=0.5)
        results.append(dm)

    if not results:
        print("  Không có frame nào để đánh giá!")
        return {}

    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall    = np.mean([r['recall']    for r in results])
    avg_f1        = np.mean([r['f1']        for r in results])
    total_tp = sum(r['tp'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fn = sum(r['fn'] for r in results)

    print(f"\n  Số frame đánh giá : {len(results)}")
    print(f"  Precision trung bình : {avg_precision:.4f}  "
          f"(YOLO detect bao nhiêu % là đúng)")
    print(f"  Recall    trung bình : {avg_recall:.4f}  "
          f"(tìm được bao nhiêu % người thật)")
    print(f"  F1 Score  trung bình : {avg_f1:.4f}")
    print(f"\n  Tổng TP : {total_tp}  (detect đúng)")
    print(f"  Tổng FP : {total_fp}  (detect thừa / sai vị trí)")
    print(f"  Tổng FN : {total_fn}  (bỏ sót người chơi)")

    return {
        'precision': avg_precision,
        'recall':    avg_recall,
        'f1':        avg_f1,
        'tp':        total_tp,
        'fp':        total_fp,
        'fn':        total_fn,
    }


def run_tracking_evaluation(gt_data: dict, pred_tracks: dict) -> dict:
  
    print("\n" + "=" * 60)
    print("TRACKING METRICS")
    print("=" * 60)

    gt_tracks = {
        fi: gt_data[fi]['tracks']
        for fi in gt_data
        if 'tracks' in gt_data[fi]
    }

    tm = tracking_metrics(gt_tracks, pred_tracks, iou_threshold=0.5)

    print(f"\n  MOTA (Tracking Accuracy)  : {tm['MOTA']:.4f}  "
          f"(tốt nhất = 1.0)")
    print(f"  MOTP (Tracking Precision) : {tm['MOTP']:.4f}  "
          f"(IoU trung bình của các match)")
    print(f"\n  ID Switches   : {tm['id_switches']}  "
          f"← số lần nhầm ID (càng nhỏ càng tốt)")
    print(f"  Misses        : {tm['misses']}  "
          f"(số lần bỏ sót người chơi)")
    print(f"  False Positives: {tm['false_positives']}  "
          f"(tracker báo có người nhưng thực tế không có)")
    print(f"  Tổng GT objects: {tm['total_gt']}")

    return tm


def run_blur_evaluation(video_path: str) -> dict:
    print("\n" + "=" * 60)
    print("CHẤT LƯỢNG VIDEO (BLUR ANALYSIS)")
    print("=" * 60)

    blurry_frames = find_blurry_frames(video_path, threshold=100.0)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    blur_ratio = len(blurry_frames) / total_frames * 100 if total_frames > 0 else 0

    print(f"\n  Tổng số frame   : {total_frames}")
    print(f"  Frame bị mờ     : {len(blurry_frames)} ({blur_ratio:.1f}%)")
    if blurry_frames:
        worst = sorted(blurry_frames, key=lambda x: x[1])[:5]
        print(f"  5 frame mờ nhất : {worst}")

    return {'total': total_frames, 'blurry': len(blurry_frames)}


def run_occlusion_analysis(gt_data: dict) -> dict:
    print("\n" + "=" * 60)
    print("PHÂN TÍCH OCCLUSION")
    print("=" * 60)

    occlusion_frames = []

    for frame_idx, data in gt_data.items():
        boxes = data['boxes']
        if len(boxes) >= 2:
            pairs = detect_occlusion(boxes, iou_threshold=0.1)
            if pairs:
                occlusion_frames.append(frame_idx)

    ratio = len(occlusion_frames) / len(gt_data) * 100 if gt_data else 0

    print(f"\n  Tổng frame có GT     : {len(gt_data)}")
    print(f"  Frame có occlusion   : {len(occlusion_frames)} ({ratio:.1f}%)")
    if occlusion_frames[:10]:
        print(f"  Ví dụ các frame      : {occlusion_frames[:10]}")

    return {'occlusion_frames': occlusion_frames}


def visualize_sample_frames(gt_data: dict, pred_boxes: dict,
                             video_path: str, num_samples: int = 6):

    print("\n" + "=" * 60)
    print("VISUALIZE — So sánh GT vs Prediction")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_frames = sorted(gt_data.keys())[::max(1, len(gt_data) // num_samples)][:num_samples]

    images, titles = [], []

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gt_boxes   = gt_data[frame_idx]['boxes']
        preds      = pred_boxes.get(frame_idx, [])
        pred_b     = [[p[0], p[1], p[2], p[3]] for p in preds]

        vis = plot_boxes(frame, gt_boxes,  color=(0, 255, 0), labels=[f"GT{i+1}" for i in range(len(gt_boxes))])
        vis = plot_boxes(vis,   pred_b,    color=(0, 0, 255), labels=[f"P{i+1}"  for i in range(len(pred_b))])

        images.append(vis)
        titles.append(f"Frame {frame_idx}")

    cap.release()

    if not images:
        print("  Không có frame nào để visualize.")
        return

    cols = 3
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')


    from matplotlib.patches import Patch
    legend = [Patch(color='green', label='Ground Truth (GT)'),
              Patch(color='red',   label='Prediction (YOLO)')]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=12)

    plt.suptitle("So sánh Ground Truth vs Prediction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("evaluation_visual.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("  Đã lưu ảnh → evaluation_visual.png")
    print("  Chú thích: Xanh lá = GT, Đỏ = Prediction YOLO")


if __name__ == "__main__":
    print("\n TENNIS TRACKING — ĐÁNH GIÁ HỆ THỐNG")
    print("=" * 60)

    print("\n Đang load dữ liệu...")

    gt_data = load_coco_gt("annotations/instances_default.json")
    pred_boxes, pred_tracks = load_predictions("pred_boxes.json", "pred_tracks.json")

    print(f"  GT frames loaded    : {len(gt_data)}")
    print(f"  Pred frames loaded  : {len(pred_boxes)}")

    det_result = run_detection_evaluation(gt_data, pred_boxes)
    trk_result = run_tracking_evaluation(gt_data, pred_tracks)
    occ_result = run_occlusion_analysis(gt_data)

    VIDEO_PATH = "data/raw_video/tennis_input6.mp4"
    visualize_sample_frames(gt_data, pred_boxes, VIDEO_PATH) 
    run_blur_evaluation(VIDEO_PATH)                           

    print("\n" + "=" * 60)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 60)
    if det_result:
        print(f"  Detection  — Precision: {det_result['precision']:.4f} | "
              f"Recall: {det_result['recall']:.4f} | "
              f"F1: {det_result['f1']:.4f}")
    if trk_result:
        print(f"  Tracking   — MOTA: {trk_result['MOTA']:.4f} | "
              f"MOTP: {trk_result['MOTP']:.4f} | "
              f"ID Switches: {trk_result['id_switches']}")
    if occ_result:
        print(f"  Occlusion  — {len(occ_result['occlusion_frames'])} frames có 2 người che nhau")
    print()