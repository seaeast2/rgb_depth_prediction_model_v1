import unittest

import numpy as np
from PIL import Image

from rgb_depth_prediction import RGBDepthEstimator


class RGBDepthEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = RGBDepthEstimator(processing_size=32)

    def test_prediction_preserves_dimensions_and_range(self):
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        image[:, 10:] = 255
        depth = self.estimator.predict(image)

        self.assertEqual(depth.shape, (12, 20))
        self.assertEqual(depth.dtype, np.float32)
        self.assertGreaterEqual(depth.min(), 0)
        self.assertLessEqual(depth.max(), 1)
        self.assertGreater(depth.max(), depth.min())

    def test_predict_image_returns_grayscale_image(self):
        result = self.estimator.predict_image(Image.new("RGB", (20, 12), "blue"))

        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (20, 12))

    def test_rejects_non_rgb_array(self):
        with self.assertRaises(ValueError):
            self.estimator.predict(np.zeros((12, 20), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
