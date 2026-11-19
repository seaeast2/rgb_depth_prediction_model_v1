"""A deterministic baseline for estimating relative depth from one RGB image.

This is intentionally small enough to run without model weights.  It combines
image luminance, local image gradients, and a weak center prior into a relative
depth map.  The output is useful as a baseline and as a drop-in interface
before replacing the implementation with a learned model such as DUST3R.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image

ImageInput: TypeAlias = Image.Image | np.ndarray | str | Path


class RGBDepthEstimator:
    """Estimate normalized relative depth from an RGB image.

    A value of ``1`` represents a comparatively near pixel and ``0`` a
    comparatively distant pixel.  Absolute metric depth cannot be recovered
    from a single image without camera calibration or learned scale.
    """

    def __init__(self, processing_size: int = 256) -> None:
        if not isinstance(processing_size, int) or processing_size < 32:
            raise ValueError("processing_size must be an integer >= 32")
        self.processing_size = processing_size

    def predict(self, image: ImageInput) -> np.ndarray:
        """Return a float32 ``(height, width)`` relative depth map."""
        rgb = self._as_rgb(image)
        height, width = rgb.shape[:2]
        resized = Image.fromarray(rgb).resize(
            (self.processing_size, self.processing_size), Image.Resampling.BILINEAR
        )
        small = np.asarray(resized, dtype=np.float32) / 255.0

        luminance = (
            0.2126 * small[..., 0]
            + 0.7152 * small[..., 1]
            + 0.0722 * small[..., 2]
        )
        vertical, horizontal = np.gradient(luminance)
        gradient = np.sqrt(horizontal * horizontal + vertical * vertical)

        yy, xx = np.mgrid[0 : self.processing_size, 0 : self.processing_size]
        center_prior = 1.0 - np.sqrt(
            ((xx - (self.processing_size - 1) / 2) / self.processing_size) ** 2
            + ((yy - (self.processing_size - 1) / 2) / self.processing_size) ** 2
        )
        score = 0.55 * (1.0 - luminance) + 0.30 * gradient + 0.15 * center_prior
        depth = self._normalize(score)

        output = Image.fromarray((depth * 255).astype(np.uint8)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        return np.asarray(output, dtype=np.float32) / 255.0

    def predict_image(self, image: ImageInput) -> Image.Image:
        """Return the predicted relative depth as an 8-bit grayscale image."""
        depth = self.predict(image)
        return Image.fromarray(np.round(depth * 255).astype(np.uint8), mode="L")

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        low, high = np.percentile(values, (2, 98))
        if high <= low:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _as_rgb(image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(image) as loaded:
                image = loaded.convert("RGB")
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a path, PIL image, or RGB numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3)")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image dimensions must be greater than zero")
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number):
                raise TypeError("RGB array must contain numeric values")
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate relative depth from an RGB image")
    parser.add_argument("input", type=Path, help="input RGB image")
    parser.add_argument("output", type=Path, help="output grayscale depth image")
    parser.add_argument("--size", type=int, default=256, help="internal processing size")
    args = parser.parse_args()
    RGBDepthEstimator(args.size).predict_image(args.input).save(args.output)


if __name__ == "__main__":
    main()
