import cv2
import numpy as np
from typing import List, Tuple

def generate_color_by_id(track_id: int) -> Tuple[int, int, int]:
    """
    Generates a unique BGR color for each track ID using HSV color space.
    
    Logic: 
    - Hue is distributed across [0, 180) based on the track_id.
    - Saturation and Value are fixed at maximum for high visibility.
    
    Reference: Image Formation - HSV color model
    """
    # Use golden ratio conjugate to distribute colors more evenly
    # Golden ratio conjugate is approximately 0.618033988749895
    phi = 0.618033988749895
    hue = int((track_id * phi * 180) % 180)  # OpenCV HSV: H is in [0, 179]
    
    # Create HSV color and convert to BGR
    hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    
    return tuple(map(int, bgr_color[0, 0]))


def draw_bbox_with_trackid(
    frame: np.ndarray,
    tracked_objects: List[Tuple[float, float, float, float, int]],
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = False
) -> np.ndarray:
    """
    Draws bounding boxes and Track IDs on the frame.
    
    Args:
        frame: Image frame (BGR format)
        tracked_objects: List of tuples (x1, y1, x2, y2, track_id)
        thickness: Bounding box line thickness
        font_scale: Font size for the text
        show_confidence: Whether to display the detection confidence score
    """
    output = frame.copy()
    
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        # Generate color based on track ID (HSV -> BGR)
        color = generate_color_by_id(track_id)
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label = f"ID: {track_id}"
        if show_confidence and len(obj) > 5:
            conf = obj[5]
            label += f" ({conf:.2f})"
        
        # Calculate text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw background for text (for better readability)
        cv2.rectangle(
            output, 
            (x1, y1 - text_h - baseline - 5), 
            (x1 + text_w + 5, y1), 
            color, 
            cv2.FILLED
        )
        
        # Draw Track ID text (white text on colored background)
        cv2.putText(
            output, 
            label, 
            (x1 + 2, y1 - baseline - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255),  
            thickness
        )
    
    return output


def draw_player_labels(
    frame: np.ndarray,
    tracked_players: List[Tuple[float, float, float, float, int]],
    player_names: dict = None
) -> np.ndarray:
    """
    Draws bounding boxes with player names (if mapping is provided).
    
    Args:
        frame: Image frame
        tracked_players: List of tuples (x1, y1, x2, y2, track_id)
        player_names: Dict mapping {track_id: "Player Name"}
    """
    output = frame.copy()
    
    if player_names is None:
        player_names = {}
    
    for player in tracked_players:
        x1, y1, x2, y2, track_id = player[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        color = generate_color_by_id(track_id)
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Label: Player Name or Track ID
        label = player_names.get(track_id, f"Player {track_id}")
        
        cv2.putText(
            output, 
            label, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )
    
    return output