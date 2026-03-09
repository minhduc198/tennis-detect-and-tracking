import matplotlib.pyplot as plt
import os


def plot_detection_stats(metrics, save_dir="evaluation_results"):
    os.makedirs(save_dir, exist_ok=True)

    labels = ["Frames", "Total Detections"]
    values = [
        metrics["frames_processed"],
        metrics["total_detections"]
    ]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)

    plt.title("Detection Statistics")
    plt.ylabel("Count")

    path = os.path.join(save_dir, "detection_stats.png")
    plt.savefig(path)

    print("Saved:", path)


def plot_tracking_stats(metrics, save_dir="evaluation_results"):
    os.makedirs(save_dir, exist_ok=True)

    labels = ["Unique Players", "ID Switches"]
    values = [
        metrics["total_unique_players"],
        metrics["id_switches"]
    ]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)

    plt.title("Tracking Performance")
    plt.ylabel("Count")

    path = os.path.join(save_dir, "tracking_stats.png")
    plt.savefig(path)

    print("Saved:", path)