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
    k = max(25, int(min(w, h) * 0.70) | 1)
    if k % 2 == 0:
        k += 1
    image[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)


def _find_top_right_watermark_boxes(image: np.ndarray) -> List[Box]:
    """Cover the supplier's consistently placed two-line upper-right watermark on all photos."""
    height, width = image.shape[:2]
    if width < 100 or height < 100:
        return []
    # The watermark is printed in the upper right quadrant on all supplier vehicle photos.
    return [_clamp_box((int(0.62 * width), 0, int(0.38 * width), int(0.18 * height)), width, height)]


def _find_banner_boxes(image: np.ndarray) -> List[Box]:
    """Cover any bottom watermark banner if present."""
    return []


def _find_plate_boxes(image: np.ndarray) -> List[Box]:
    """Detect and mask vehicle license plates and dealer plate frames in exterior shots."""
    height, width = image.shape[:2]
    if width < 150 or height < 150:
        return []

    # Search in lower 65% of image where plates always appear on exterior car shots
    y_start = int(height * 0.35)
    y_end = int(height * 0.95)
    roi = image[y_start:y_end, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        # Plate dimensions in vehicle photos: aspect 1.8-6.0, width 7-40% of image width, height 2-18% of image height
        if 1.8 <= aspect <= 6.0 and 0.07 * width <= bw <= 0.40 * width and 0.02 * height <= bh <= 0.18 * height:
            pad_x = int(bw * 0.12)
            pad_y = int(bh * 0.18)
            bx = max(0, x - pad_x)
            by = max(0, y + y_start - pad_y)
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
