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
    return [
        # Full overhead sign: left logo, centered slogan, and right logo.
        _clamp_box((int(0.08 * width), int(0.05 * height), int(0.88 * width), int(0.16 * height)), width, height),
        # Front plate on the front three-quarter views.
        _clamp_box((int(0.08 * width), int(0.68 * height), int(0.38 * width), int(0.24 * height)), width, height),
    ]


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
