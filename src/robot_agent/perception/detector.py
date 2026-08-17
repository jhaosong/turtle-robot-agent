"""Pluggable image detectors used by active search tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from robot_agent.state import Detection, ImagePosition

from .color_detection import HSV_THRESHOLDS, detect_colored_blobs


class Detector(ABC):
    """Convert one BGR image into bounded semantic detections."""

    @abstractmethod
    def validate_query(self, *, color: str | None, label: str | None) -> None:
        """Fail before motion when this detector cannot satisfy the query."""

    @abstractmethod
    def detect(
        self,
        image: Any,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> list[Detection]:
        raise NotImplementedError


class ColorBlobDetector(Detector):
    def validate_query(self, *, color: str | None, label: str | None) -> None:
        if color not in HSV_THRESHOLDS:
            raise ValueError(
                "Color blob detection requires one of: "
                f"{sorted(HSV_THRESHOLDS)}"
            )
        if label not in {None, "colored_object"}:
            raise ValueError(
                "Color blob detection only supports label='colored_object'"
            )

    def detect(
        self,
        image: Any,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> list[Detection]:
        self.validate_query(color=color, label=label)
        return detect_colored_blobs(image, color or "")


class YoloDetector(Detector):
    """Ultralytics YOLO detector with bounded input size for continuous use."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        input_size: int = 640,
        confidence_threshold: float = 0.1,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError(
                "ROBOT_AGENT_DETECTOR_BACKEND=yolo requires the ultralytics package"
            ) from exc
        self.model_name = model_name
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self._model = YOLO(model_name)

    def validate_query(self, *, color: str | None, label: str | None) -> None:
        if not label:
            raise ValueError("YOLO detection requires an object label")
        if color is not None:
            raise ValueError("YOLO class detection does not filter by color")

    def detect(
        self,
        image: Any,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> list[Detection]:
        self.validate_query(color=color, label=label)
        results = self._model.predict(
            source=image,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_label = str(names[class_id])
                if detected_label != label:
                    continue
                x_min, y_min, x_max, y_max = (
                    float(value) for value in box.xyxy[0]
                )
                height, width = result.orig_shape[:2]
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                detections.append(
                    Detection(
                        label=detected_label,
                        confidence=float(box.conf[0]),
                        image_position=ImagePosition(
                            x_px=center_x,
                            y_px=center_y,
                            x_normalized=center_x / float(width),
                            y_normalized=center_y / float(height),
                            width_normalized=(x_max - x_min) / float(width),
                            height_normalized=(y_max - y_min) / float(height),
                        ),
                    )
                )
        return detections


class YoloeDetector(Detector):
    """Ultralytics YOLOE-26 open-vocabulary text-prompt detection."""

    def __init__(
        self,
        model_name: str = "yoloe-26s-seg.pt",
        input_size: int = 640,
        confidence_threshold: float = 0.1,
    ) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "ROBOT_AGENT_DETECTOR_BACKEND=yoloe requires "
                "ultralytics==8.4.67 with YOLOE support"
            ) from exc
        self.model_name = model_name
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self._model = YOLOE(model_name)
        self._active_label: str | None = None

    def validate_query(self, *, color: str | None, label: str | None) -> None:
        if not label:
            raise ValueError("YOLOE detection requires a text-prompt label")

    def detect(
        self,
        image: Any,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> list[Detection]:
        self.validate_query(color=color, label=label)
        assert label is not None
        # Reuse the text embedding across frames in one continuous search.
        if label != self._active_label:
            self._model.set_classes([label])
            self._active_label = label
        results = self._model.predict(
            source=image,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            height, width = result.orig_shape[:2]
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_label = str(result.names[class_id])
                if detected_label != label:
                    continue
                x_min, y_min, x_max, y_max = (
                    float(value) for value in box.xyxy[0]
                )
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                detections.append(
                    Detection(
                        label=detected_label,
                        confidence=float(box.conf[0]),
                        # Text prompting verifies the label, not the requested color.
                        color=None,
                        image_position=ImagePosition(
                            x_px=center_x,
                            y_px=center_y,
                            x_normalized=center_x / float(width),
                            y_normalized=center_y / float(height),
                            width_normalized=(x_max - x_min) / float(width),
                            height_normalized=(y_max - y_min) / float(height),
                        ),
                    )
                )
        return detections


class VlmDetector(Detector):
    """Extension point for a future bounded VLM implementation."""

    def validate_query(self, *, color: str | None, label: str | None) -> None:
        raise NotImplementedError("No VLM detector is configured")

    def detect(
        self,
        image: Any,
        *,
        color: str | None = None,
        label: str | None = None,
    ) -> list[Detection]:
        raise NotImplementedError("No VLM detector is configured")


def build_detector(
    backend: str,
    *,
    yolo_model: str = "yolov8n.pt",
    yoloe_model: str = "yoloe-26s-seg.pt",
    yolo_input_size: int = 640,
    confidence_threshold: float = 0.1,
) -> Detector:
    if backend == "color_blob":
        return ColorBlobDetector()
    if backend == "yolo":
        return YoloDetector(yolo_model, yolo_input_size, confidence_threshold)
    if backend == "yoloe":
        return YoloeDetector(yoloe_model, yolo_input_size, confidence_threshold)
    if backend == "vlm":
        raise NotImplementedError("ROBOT_AGENT_DETECTOR_BACKEND=vlm is not implemented")
    raise ValueError(f"Unsupported detector backend: {backend}")
