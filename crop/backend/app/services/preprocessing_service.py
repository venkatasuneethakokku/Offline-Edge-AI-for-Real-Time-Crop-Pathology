import io
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


class PreprocessingService:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size

    def preprocess_image_bytes(self, content: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"Invalid image file: {exc}") from exc

        image = image.resize(self.target_size)
        # Model pipeline already includes MobileNetV2 preprocess_input.
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        return array
