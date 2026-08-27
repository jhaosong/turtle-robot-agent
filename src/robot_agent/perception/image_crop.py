"""Compact image artifacts derived from semantic detection boxes."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from robot_agent.state import Detection, ImagePosition


@dataclass(frozen=True)
class DetectionCrop:
    """An independent image crop and its bounds in the source frame."""

    image: np.ndarray
    left: int
    top: int
    right: int
    bottom: int
    source_width: int
    source_height: int

    def metadata(self) -> dict[str, Any]:
        return {
            "bounds_px": {
                "left": self.left,
                "top": self.top,
                "right": self.right,
                "bottom": self.bottom,
            },
            "crop_size_px": {
                "width": self.right - self.left,
                "height": self.bottom - self.top,
            },
            "source_size_px": {
                "width": self.source_width,
                "height": self.source_height,
            },
        }


def crop_detection_image(
    image: np.ndarray,
    detection: Detection | ImagePosition,
    *,
    padding_ratio: float = 0.05,
) -> DetectionCrop | None:
    """Crop one detection from an image using normalized bbox coordinates.

    The returned array is copied so retaining it does not retain the full source
    frame in memory. ``None`` means the image or detection has no usable box.
    """
    if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
        return None
    if not math.isfinite(padding_ratio) or padding_ratio < 0.0:
        raise ValueError("padding_ratio must be a finite non-negative value")

    position = detection.image_position if isinstance(detection, Detection) else detection
    if position is None:
        return None

    values = (
        position.x_normalized,
        position.y_normalized,
        position.width_normalized,
        position.height_normalized,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if position.width_normalized <= 0.0 or position.height_normalized <= 0.0:
        return None

    source_height, source_width = int(image.shape[0]), int(image.shape[1])
    if source_width <= 0 or source_height <= 0:
        return None

    box_width = position.width_normalized * source_width
    box_height = position.height_normalized * source_height
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    center_x = position.x_normalized * source_width
    center_y = position.y_normalized * source_height

    left = max(0, math.floor(center_x - box_width / 2.0 - pad_x))
    top = max(0, math.floor(center_y - box_height / 2.0 - pad_y))
    right = min(source_width, math.ceil(center_x + box_width / 2.0 + pad_x))
    bottom = min(source_height, math.ceil(center_y + box_height / 2.0 + pad_y))
    if right <= left or bottom <= top:
        return None

    return DetectionCrop(
        image=image[top:bottom, left:right].copy(),
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        source_width=source_width,
        source_height=source_height,
    )


def crop_to_jpeg_data_url(
    crop: DetectionCrop,
    *,
    quality: int = 85,
) -> str:
    """Encode a crop for a multimodal LLM image content block."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - supplied by the ROS image
        raise RuntimeError("OpenCV is required to encode detection crops") from error

    encoded, buffer = cv2.imencode(
        ".jpg",
        crop.image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not encoded:
        raise RuntimeError("Failed to encode detection crop as JPEG")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
