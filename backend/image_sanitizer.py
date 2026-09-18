"""Automatic masking of supplier branding in client-facing vehicle photos."""

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
    # Large enough kernel to make logos and text unreadable while preserving natural background.
    k = max(25, int(min(w, h) * 0.70) | 1)
    if k % 2 == 0:
        k += 1
    image[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)


def _find_top_right_watermark_boxes(image: np.ndarray) -> List[Box]:
    """Cover the supplier's consistently placed two-line upper-right watermark."""
    height, width = image.shape[:2]
    if width < 100 or height < 100:
        return []
    # The watermark is printed in the upper right quadrant on all supplier vehicle photos.
    return [_clamp_box((int(0.66 * width), 0, int(0.34 * width), int(0.18 * height)), width, height)]


def sanitize_image(content: bytes) -> bytes:
    """Blur detected supplier banners and plate logos without changing other image content."""
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
