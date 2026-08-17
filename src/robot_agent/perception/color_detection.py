"""Headless HSV blob detection adapted from TurtleBot's LookForObject behavior."""

from __future__ import annotations

from typing import Any

from robot_agent.state import Detection, ImagePosition


HSV_THRESHOLDS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red": ((160, 220, 0), (180, 255, 255)),
    "green": ((40, 220, 0), (90, 255, 255)),
    "blue": ((100, 100, 40), (140, 255, 255)),
}


def detect_colored_blobs(image: Any, color: str) -> list[Detection]:
    """Return image-plane color detections without claiming world coordinates."""
    if color not in HSV_THRESHOLDS:
        raise ValueError(f"Unsupported detection color: {color}. Allowed: {sorted(HSV_THRESHOLDS)}")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on ROS image environment
        raise RuntimeError("OpenCV is required for camera color detection") from exc

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = HSV_THRESHOLDS[color]
    mask = cv2.inRange(hsv, lower, upper)
    parameters = cv2.SimpleBlobDetector_Params()
    parameters.minArea = 100
    parameters.maxArea = 100000
    parameters.filterByArea = True
    parameters.filterByColor = False
    parameters.filterByInertia = False
    parameters.filterByConvexity = False
    parameters.thresholdStep = 50
    keypoints = cv2.SimpleBlobDetector_create(parameters).detect(mask)
    height, width = image.shape[:2]
    return [
        Detection(
            label="colored_object",
            color=color,
            confidence=min(1.0, point.size / 100.0),
            image_position=ImagePosition(
                x_px=float(point.pt[0]),
                y_px=float(point.pt[1]),
                x_normalized=float(point.pt[0]) / float(width),
                y_normalized=float(point.pt[1]) / float(height),
                width_normalized=float(point.size) / float(width),
                height_normalized=float(point.size) / float(height),
            ),
        )
        for point in keypoints
    ]
