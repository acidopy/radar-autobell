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
    """Cover the supplier's consistently placed two-line upper-right watermark on all photos."""
    height, width = image.shape[:2]
    if width < 80 or height < 80:
        return []
    # The watermark is printed in the upper right quadrant on all supplier vehicle photos.
    return [_clamp_box((int(0.55 * width), 0, int(0.45 * width), int(0.22 * height)), width, height)]


def _find_banner_boxes(image: np.ndarray) -> List[Box]:
    """Cover the curved studio wall banner (Hyundai Glovis / Autobell Global) behind vehicles."""
    height, width = image.shape[:2]
    boxes: List[Box] = []

    # Detect Cyan / Blue background studio banner (Hyundai Glovis / Autobell)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([80, 25, 90])
    upper_cyan = np.array([140, 255, 255])
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    cyan_mask[:int(height * 0.08), :] = 0
    cyan_mask[int(height * 0.50):, :] = 0

    k_banner = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 9))
    closed_banner = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, k_banner)
    contours_b, _ = cv2.findContours(closed_banner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_b:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw >= int(0.10 * width) and bh >= 8:
            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.25)
            boxes.append(_clamp_box((x - pad_x, y - pad_y, bw + 2 * pad_x, bh + 2 * pad_y), width, height))

    return boxes


def _find_plate_boxes(image: np.ndarray) -> List[Box]:
    """Detect and mask vehicle license plates, Autobell dealer tags, and plate frames."""
    height, width = image.shape[:2]
    boxes: List[Box] = []
    y_min, y_max = int(height * 0.28), int(height * 0.96)

    # 1. White / Light plate tags & dealer logos (Autobell logo tag, Korean plates)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 80, 255]))
    white_mask[:y_min, :] = 0
    white_mask[y_max:, :] = 0
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9))
    closed_white = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k_close)
    contours_w, _ = cv2.findContours(closed_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_w:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        if 1.5 <= aspect <= 5.8 and 0.04 * width <= bw <= 0.40 * width and 0.02 * height <= bh <= 0.18 * height:
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.20)
            boxes.append(_clamp_box((x - pad_x, y - pad_y, bw + 2 * pad_x, bh + 2 * pad_y), width, height))

    # 2. Gradient / contrast plates (yellow, dark-border, or blue EV plates)
    gray = cv2.cvtColor(image[y_min:y_max, :], cv2.COLOR_BGR2GRAY)
    k_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_grad)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_join = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7))
    closed_grad = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_join)
    contours_g, _ = cv2.findContours(closed_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_g:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        if 1.6 <= aspect <= 5.5 and 0.05 * width <= bw <= 0.35 * width and 0.02 * height <= bh <= 0.16 * height:
            pad_x = int(bw * 0.12)
            pad_y = int(bh * 0.15)
            boxes.append(_clamp_box((x - pad_x, y + y_min - pad_y, bw + 2 * pad_x, bh + 2 * pad_y), width, height))

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
