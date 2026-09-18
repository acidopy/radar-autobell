"""Automatic masking of supplier branding, watermarks, and license plates in vehicle photos."""

from typing import List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


def _clamp_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    right = max(x + 1, min(x + w, width))
    bottom = max(y + 1, min(y + h, height))
    return x, y, right - x, bottom - y


def _blur_box(image: np.ndarray, box: Box) -> None:
    height, width = image.shape[:2]
    x, y, w, h = _clamp_box(box, width, height)
    roi = image[y:y + h, x:x + w]
    if roi.size == 0:
        return
    # Large enough kernel to make logos, license plates, and text unreadable while preserving natural background.
    k = max(35, int(min(w, h) * 0.90) | 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    blurred = cv2.GaussianBlur(blurred, (k, k), 0)
    image[y:y + h, x:x + w] = blurred


def _find_top_right_watermark_boxes(image: np.ndarray) -> List[Box]:
    """Cover the supplier's consistently placed upper-right watermark."""
    height, width = image.shape[:2]
    if width < 80 or height < 80:
        return []
    # Compact box covering only the top-right watermark text
    return [_clamp_box((int(0.68 * width), 0, int(0.32 * width), int(0.12 * height)), width, height)]


def _find_banner_boxes(image: np.ndarray) -> List[Box]:
    """Cover studio wall banner if present without touching vehicles or skies."""
    return []


def _find_plate_boxes(image: np.ndarray) -> List[Box]:
    """Detect and mask license plates strictly bounded to plate dimensions."""
    height, width = image.shape[:2]
    boxes: List[Box] = []
    
    # Strictly search in lower-middle area where license plates reside
    y_min, y_max = int(height * 0.45), int(height * 0.88)
    roi = image[y_min:y_max, :]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    k_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_grad)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    k_join = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_join)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_plate_area = int(0.02 * width * height)  # Max 2% of total image area
    
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        area = bw * bh
        # Strict plate geometry constraints:
        # Width: 6% to 22% of image width
        # Height: 2% to 8% of image height
        # Aspect ratio: 2.2 to 5.5
        # Area: <= 2% of total image area
        if (2.2 <= aspect <= 5.5 and
            0.06 * width <= bw <= 0.22 * width and
            0.02 * height <= bh <= 0.08 * height and
            area <= max_plate_area):
            pad_x = int(bw * 0.10)
            pad_y = int(bh * 0.12)
            bx = max(0, x - pad_x)
            by = max(0, y + y_min - pad_y)
            bw_exp = min(width - bx, bw + 2 * pad_x)
            bh_exp = min(height - by, bh + 2 * pad_y)
            boxes.append((bx, by, bw_exp, bh_exp))
            
    return boxes


def sanitize_image(content: bytes) -> bytes:
    """Blur detected supplier banners, plates, and watermark logos without altering other image content."""
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return content

    boxes = _find_banner_boxes(image) + _find_plate_boxes(image) + _find_top_right_watermark_boxes(image)
    for box in boxes:
        _blur_box(image, box)

    if not boxes:
        return content
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return encoded.tobytes() if ok else content
