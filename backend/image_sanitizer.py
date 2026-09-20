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
    # A blur can still leave letter silhouettes visible in thumbnails. Replace
    # the whole protected region with its median color so no brand shape or
    # plate text survives compression, resizing, or sharpening downstream.
    fill = np.median(roi.reshape(-1, roi.shape[2]), axis=0).astype(np.uint8)
    image[y:y + h, x:x + w] = fill


def _find_studio_banner_box(image: np.ndarray) -> List[Box]:
    """Find the long blue studio banner without masking outdoor backgrounds."""
    height, width = image.shape[:2]
    y0, y1 = int(0.04 * height), int(0.35 * height)
    x0, x1 = int(0.04 * width), int(0.96 * width)
    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([80, 55, 70], dtype=np.uint8), np.array([115, 255, 255], dtype=np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        global_y = y0 + y
        if (w >= int(0.35 * width) and h >= max(8, int(0.02 * height))
                and h <= int(0.12 * height) and global_y >= int(0.08 * height)
                and aspect >= 4.0):
            candidates.append((w * h, x, y, w, h))
    if not candidates:
        return []
    _, x, y, w, h = max(candidates)
    return [_clamp_box((x0 + x - int(0.015 * width), y0 + y - int(0.02 * height),
                        w + int(0.03 * width), h + int(0.04 * height)), width, height)]


def _find_supplier_brand_boxes(image: np.ndarray) -> List[Box]:
    """Return the fixed supplier-brand regions used by Autobell studio photos.

    The entire upper sign is masked (including its centered slogan), while
    the front plate is lower and left-of-center on the front three-quarter
    photos.  Keeping the regions explicit prevents the upper mask from
    spreading into the vehicle.
    """
    height, width = image.shape[:2]
    if width < 80 or height < 80:
        return []
    boxes = _find_studio_banner_box(image)
    boxes.extend([
        # A few outdoor photos have an isolated top-right watermark instead.
        _clamp_box((int(0.82 * width), int(0.02 * height), int(0.16 * width), int(0.08 * height)), width, height),
        # Front plates vary between left-of-center three-quarter views and
        # centered studio views; keep both fallback regions narrow.
        _clamp_box((int(0.04 * width), int(0.70 * height), int(0.16 * width), int(0.13 * height)), width, height),
        _clamp_box((int(0.18 * width), int(0.64 * height), int(0.24 * width), int(0.16 * height)), width, height),
    ])
    return boxes


def sanitize_image(content: bytes) -> bytes:
    """Blur supplier branding while leaving vehicle and pricing logic untouched."""
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return content

    boxes = _find_supplier_brand_boxes(image)
    for box in boxes:
        _blur_box(image, box)

    if not boxes:
        return content
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return encoded.tobytes() if ok else content
