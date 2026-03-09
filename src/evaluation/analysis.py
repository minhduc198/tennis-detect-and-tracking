import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_boxes(frame, boxes, ids=None, color=(0, 255, 0), thickness=2, labels=None):
    img = frame.copy()
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = None
        if ids is not None and idx < len(ids):
            text = str(ids[idx])
        elif labels is not None and idx < len(labels):
            text = labels[idx]
        if text is not None:
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return img


def compute_blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def find_blurry_frames(video_path, threshold=100.0):
    cap = cv2.VideoCapture(video_path)
    bad_frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        score = compute_blur_score(frame)
        if score < threshold:
            bad_frames.append((idx, score))
        idx += 1
    cap.release()
    return bad_frames


def detect_occlusion(gt_boxes, iou_threshold=0.3):
    occluded_pairs = []
    for i in range(len(gt_boxes)):
        for j in range(i + 1, len(gt_boxes)):
            if iou(gt_boxes[i], gt_boxes[j]) > iou_threshold:
                occluded_pairs.append((i, j))
    return occluded_pairs


# reuse IoU from metrics to avoid duplication
from .metrics import iou


def gallery(images, cols=3, figsize=(12, 8), titles=None):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    return fig


def manual_label_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    labels = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, f"Frame {idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow("label", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            labels.setdefault(idx, []).append('occlusion')
        elif key == ord('b'):
            labels.setdefault(idx, []).append('blur')
        idx += 1
    cap.release()
    cv2.destroyAllWindows()
    return labels


if __name__ == "__main__":
    # quick demonstration
    import numpy as _np
    dummy = _np.zeros((200, 300, 3), dtype=_np.uint8)
    boxes = [[10, 10, 100, 150]]
    img = plot_boxes(dummy, boxes, labels=['player'])
    cv2.imwrite('test.png', img)
    print('wrote test.png with a box')
