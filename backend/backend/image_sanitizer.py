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
    # Large enough to make both logos and text unreadable while preserving context.
    kernel = max(15, int(min(w, h) * 0.65) | 1)
    if kernel % 2 == 0:
        kernel += 1
    image[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (kernel, kernel), 0)


def _find_banner_boxes(image: np.ndarray) -> List[Box]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Blue/cyan is characteristic of the supplier banner, regardless of its position.
    mask = cv2.inRange(hsv, np.array([80, 45, 55]), np.array([125, 255, 255]))
    mask[int(height * 0.50):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))

    boxes: List[Box] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            w < width * 0.22
            or w > width * 0.98
            or y > height * 0.36
            or h < height * 0.018
            or h > height * 0.15
            or w / max(h, 1) < 3.5
        ):
            continue
        roi = hsv[y:y + h, x:x + w]
        blue_density = float(np.mean((roi[:, :, 0] >= 80) & (roi[:, :, 0] <= 125) & (roi[:, :, 1] >= 45)))
        if float(np.mean(roi[:, :, 2])) < 160 or blue_density < 0.12:
            continue
        # Add a small border so logos at either edge are covered too.
        boxes.append(_clamp_box((x - 0.03 * w, y - 0.35 * h, 1.06 * w, 1.7 * h), width, height))
    return boxes


def _find_plate_boxes(image: np.ndarray) -> List[Box]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Supplier logos on plates are blue/cyan. Find them in the lower part of the car photo.
    mask = cv2.inRange(hsv, np.array([90, 55, 35]), np.array([135, 255, 255]))
    mask[:int(height * 0.55)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    boxes: List[Box] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            w < width * 0.025
            or w > width * 0.08
            or h < height * 0.010
            or h > height * 0.08
            or w / max(h, 1) < 1.7
        ):
            continue
        # A logo on a light plate has a bright, low-saturation neighborhood.
        pad_x = max(int(w * 0.55), int(width * 0.018))
        pad_y = max(int(h * 0.75), int(height * 0.018))
        rx0, ry0 = max(0, x - pad_x), max(0, y - pad_y)
        rx1, ry1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        neighborhood = hsv[ry0:ry1, rx0:rx1]
        bright_plate = float(np.mean((neighborhood[:, :, 2] > 150) & (neighborhood[:, :, 1] < 100)))
        logo_blue_density = float(np.mean((hsv[y:y + h, x:x + w, 0] >= 90) & (hsv[y:y + h, x:x + w, 0] <= 125) & (hsv[y:y + h, x:x + w, 1] > 100)))
        if bright_plate < 0.50 or logo_blue_density < 0.18:
            continue
        # A plate surrounds the logo; expand generously for different camera angles.
        boxes.append(_clamp_box((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), width, height))
    # Overlapping blue components often belong to one plate logo; merge them.
    merged: List[Box] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        x, y, w, h = box
        if any(x < ox + ow and ox < x + w and y < oy + oh and oy < y + h for ox, oy, ow, oh in merged):
            continue
        merged.append(box)
    return merged


def _find_top_right_watermark_boxes(image: np.ndarray) -> List[Box]:
    """Cover the supplier's consistently placed two-line upper-right watermark."""
    height, width = image.shape[:2]
    if width < 400 or height < 250:
        return []
    # The watermark is printed in the same normalized corner on all source photos.
    return [_clamp_box((0.76 * width, 0.018 * height, 0.24 * width, 0.155 * height), width, height)]


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
