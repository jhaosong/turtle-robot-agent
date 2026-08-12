"""Perception adapters that convert sensor data into semantic detections."""

from .color_detection import detect_colored_blobs

__all__ = ["detect_colored_blobs"]
