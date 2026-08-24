"""Pluggable image detectors used by active search tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any
import warnings

from robot_agent.state import Detection, ImagePosition


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


class YoloeDetector(Detector):
    """Ultralytics YOLOE-26 text detection with optional visual prompting."""

    def __init__(
        self,
        model_name: str = "yoloe-26s-seg.pt",
        input_size: int = 640,
        confidence_threshold: float = 0.1,
        prompt_catalog_path: Path | None = None,
    ) -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "YOLOE perception requires ultralytics==8.4.67 with YOLOE support"
            ) from exc
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self._model = YOLOE(model_name)
        self._active_label: str | None = None
        self._accepted_labels: set[str] = set()
        self._prompt_catalog_path = prompt_catalog_path
        self._prompt_catalog = self._load_prompt_catalog(prompt_catalog_path)
        self.prompt_mode = "text"

    @staticmethod
    def _load_prompt_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
        if path is None:
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load YOLOE prompt catalog {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("YOLOE prompt catalog must contain a JSON object")
        return payload

    def _activate_prompt(self, label: str) -> None:
        prompt = self._prompt_catalog.get(label, {})
        reference_name = prompt.get("reference_image")
        reference_bbox = prompt.get("reference_bbox")
        if reference_name and reference_bbox and self._prompt_catalog_path is not None:
            reference_path = self._prompt_catalog_path.parent / str(reference_name)
            try:
                import numpy as np
                from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

                if not reference_path.is_file():
                    raise FileNotFoundError(reference_path)
                self._model.predict(
                    source=str(reference_path),
                    refer_image=str(reference_path),
                    visual_prompts={
                        "bboxes": np.asarray([reference_bbox], dtype=float),
                        "cls": np.asarray([0], dtype=int),
                    },
                    predictor=YOLOEVPSegPredictor,
                    imgsz=self.input_size,
                    conf=self.confidence_threshold,
                    verbose=False,
                )
                self._accepted_labels = {"object0", label}
                self._active_label = label
                self.prompt_mode = "visual"
                return
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"YOLOE visual prompt unavailable for {label!r}; "
                    f"using text descriptions instead: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        descriptions = prompt.get("descriptions") or [label]
        if not isinstance(descriptions, list) or not all(
            isinstance(item, str) and item.strip() for item in descriptions
        ):
            raise ValueError(f"Invalid YOLOE text descriptions for {label!r}")
        self._model.set_classes(descriptions)
        self._accepted_labels = set(descriptions)
        self._active_label = label
        self.prompt_mode = "text"

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
        # Reuse text or visual embeddings across frames in one search.
        if label != self._active_label:
            self._activate_prompt(label)
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
                if detected_label not in self._accepted_labels:
                    continue
                x_min, y_min, x_max, y_max = (
                    float(value) for value in box.xyxy[0]
                )
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                detections.append(
                    Detection(
                        label=label,
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
