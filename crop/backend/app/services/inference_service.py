import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, tflite_model_path: Path, class_names_path: Path, disease_info_path: Path) -> None:
        self.tflite_model_path = Path(tflite_model_path)
        self.class_names_path = Path(class_names_path)
        self.disease_info_path = Path(disease_info_path)

        self.interpreter: tf.lite.Interpreter | None = None
        self.input_details: dict[str, Any] | None = None
        self.output_details: dict[str, Any] | None = None

        self.class_names: list[str] = []
        self.disease_info: Dict[str, Dict[str, Any]] = {}
        self.disease_info_normalized: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return (
            self.interpreter is not None
            and self.input_details is not None
            and self.output_details is not None
        )

    @staticmethod
    def _normalize_key(value: str) -> str:
        key = value.strip().lower()
        key = key.replace("&", "and")
        key = re.sub(r"[\s\-]+", "_", key)
        key = re.sub(r"[^a-z0-9_]+", "_", key)
        key = re.sub(r"_+", "_", key).strip("_")
        return key

    def _lookup_disease_info(self, class_name: str) -> Dict[str, Any] | None:
        if class_name in self.disease_info:
            return self.disease_info[class_name]
        return self.disease_info_normalized.get(self._normalize_key(class_name))

    def load(self) -> None:
        with self._lock:
            if self.is_loaded:
                return

            if not self.tflite_model_path.exists():
                raise FileNotFoundError(f"TFLite model not found: {self.tflite_model_path}")
            if not self.class_names_path.exists():
                raise FileNotFoundError(f"class_names.json not found: {self.class_names_path}")
            if not self.disease_info_path.exists():
                raise FileNotFoundError(f"disease_info.json not found: {self.disease_info_path}")

            # Load class names
            with self.class_names_path.open("r", encoding="utf-8") as f:
                class_names = json.load(f)
            if not isinstance(class_names, list) or not class_names:
                raise ValueError("class_names.json must contain a non-empty list of class names")
            self.class_names = [str(item) for item in class_names]

            # Load disease metadata
            with self.disease_info_path.open("r", encoding="utf-8") as f:
                disease_info = json.load(f)
            if not isinstance(disease_info, dict):
                raise ValueError("disease_info.json must contain an object keyed by class name")
            self.disease_info = disease_info

            # Create normalized lookup map
            self.disease_info_normalized = {}
            for key, value in self.disease_info.items():
                norm = self._normalize_key(str(key))
                if norm not in self.disease_info_normalized:
                    self.disease_info_normalized[norm] = value

            # Load TFLite model using file bytes (fixes Unicode path issues on Windows)
            with open(self.tflite_model_path, "rb") as f:
                model_content = f.read()

            self.interpreter = tf.lite.Interpreter(model_content=model_content)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]

            missing = [
                name for name in self.class_names
                if self._lookup_disease_info(name) is None
            ]
            if missing:
                LOGGER.warning(
                    "Disease metadata missing for %s classes: %s",
                    len(missing),
                    missing,
                )

            LOGGER.info(
                "Inference service loaded successfully: classes=%s, model=%s",
                len(self.class_names),
                self.tflite_model_path,
            )

    def _prepare_input_tensor(self, image_batch: np.ndarray) -> np.ndarray:
        if self.input_details is None:
            raise RuntimeError("Input tensor details are not initialized.")

        model_dtype = self.input_details["dtype"]
        image_batch = np.asarray(image_batch)

        if model_dtype in (np.int8, np.uint8):
            scale, zero_point = self.input_details.get("quantization", (0.0, 0))
            if not scale:
                raise RuntimeError("Invalid quantization parameters for TFLite input.")
            quantized = np.round(image_batch / scale + zero_point)
            qinfo = np.iinfo(model_dtype)
            quantized = np.clip(quantized, qinfo.min, qinfo.max).astype(model_dtype)
            return quantized

        return image_batch.astype(model_dtype)

    def _extract_probabilities(self, output_tensor: np.ndarray) -> np.ndarray:
        if self.output_details is None:
            raise RuntimeError("Output tensor details are not initialized.")

        output = np.asarray(output_tensor[0])
        model_dtype = self.output_details["dtype"]

        if model_dtype in (np.int8, np.uint8):
            scale, zero_point = self.output_details.get("quantization", (0.0, 0))
            if scale:
                output = (output.astype(np.float32) - zero_point) * scale
            else:
                output = output.astype(np.float32)
        else:
            output = output.astype(np.float32)

        if output.ndim != 1:
            raise RuntimeError(f"Unexpected output tensor shape: {output.shape}")

        total = float(np.sum(output))
        if total <= 0.0 or np.any(output < 0.0):
            stabilized = output - np.max(output)
            exp_output = np.exp(stabilized)
            probs = exp_output / np.sum(exp_output)
        else:
            probs = output / total

        return probs

    def predict(self, preprocessed_image: np.ndarray) -> Dict[str, Any]:
        if (
            not self.is_loaded
            or self.interpreter is None
            or self.input_details is None
            or self.output_details is None
        ):
            raise RuntimeError("Model not loaded. Call load() before predict().")

        if preprocessed_image.ndim != 4:
            raise ValueError(
                f"Expected image batch with 4 dimensions, got shape {preprocessed_image.shape}"
            )

        with self._lock:
            input_tensor = self._prepare_input_tensor(preprocessed_image)
            self.interpreter.set_tensor(self.input_details["index"], input_tensor)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(self.output_details["index"])

        probs = self._extract_probabilities(raw_output)
        predicted_index = int(np.argmax(probs))
        confidence = float(np.max(probs))

        if predicted_index >= len(self.class_names):
            raise RuntimeError(
                f"Predicted class index out of range: {predicted_index}"
            )

        class_name = self.class_names[predicted_index]
        info = self._lookup_disease_info(class_name) or {
            "disease_name": class_name,
            "description": "No description available for this class.",
            "symptoms": [],
            "treatment": [],
            "prevention": [],
        }

        return {
            "disease_name": info.get("disease_name", class_name),
            "confidence": round(confidence, 4),
            "description": info.get("description", ""),
            "symptoms": info.get("symptoms", []),
            "treatment": info.get("treatment", []),
            "prevention": info.get("prevention", []),
        }