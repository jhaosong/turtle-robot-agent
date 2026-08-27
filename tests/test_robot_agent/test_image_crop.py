from __future__ import annotations

import unittest

import numpy as np

from robot_agent.perception import crop_detection_image
from robot_agent.state import Detection, ImagePosition


class DetectionCropTest(unittest.TestCase):
    def test_crops_normalized_bbox_and_does_not_retain_source_frame(self):
        image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        detection = Detection(
            label="fire extinguisher",
            confidence=0.9,
            image_position=ImagePosition(
                x_px=100.0,
                y_px=50.0,
                x_normalized=0.5,
                y_normalized=0.5,
                width_normalized=0.4,
                height_normalized=0.2,
            ),
        )

        crop = crop_detection_image(image, detection, padding_ratio=0.0)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual((crop.left, crop.top, crop.right, crop.bottom), (60, 40, 140, 60))
        self.assertEqual(crop.image.shape, (20, 80, 3))
        self.assertFalse(np.shares_memory(image, crop.image))

    def test_clips_padded_bbox_to_image_edges(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        position = ImagePosition(
            x_px=0.0,
            y_px=0.0,
            x_normalized=0.02,
            y_normalized=0.03,
            width_normalized=0.2,
            height_normalized=0.2,
        )

        crop = crop_detection_image(image, position, padding_ratio=0.1)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.left, 0)
        self.assertEqual(crop.top, 0)
        self.assertLessEqual(crop.right, 200)
        self.assertLessEqual(crop.bottom, 100)

    def test_rejects_detection_without_a_usable_bbox(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detection = Detection(label="object", confidence=0.5)

        self.assertIsNone(crop_detection_image(image, detection))


if __name__ == "__main__":
    unittest.main()
