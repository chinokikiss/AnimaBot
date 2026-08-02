import io
import json
import os
import threading

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download


# Resolve the model only when recognition is requested. Importing the agent
# must not perform a network request just to support requests without images.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "style_predictor_500.onnx")
_MODEL = None
_MODEL_LOCK = threading.Lock()


def _resolve_model_path() -> str:
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    return hf_hub_download(
        repo_id="AugustLabs/Author_ID",
        filename="style_predictor_500.onnx",
    )


class AuthorID:
    """
    Author_ID: Anime Artist Style Recognition
    Single ONNX file contains: model + centroids + author names
    """

    def __init__(self, onnx_path):
        model_onnx = onnx.load(onnx_path)
        self.names = []
        self.input_size = 384

        for prop in model_onnx.metadata_props:
            if prop.key == "author_names":
                self.names = json.loads(prop.value)
            elif prop.key == "input_size":
                self.input_size = int(prop.value)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32
        ).reshape(1, 3, 1, 1)
        self.std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        ).reshape(1, 3, 1, 1)

    def preprocess(self, image):
        if isinstance(image, (bytes, bytearray, memoryview)):
            image = io.BytesIO(bytes(image))

        img = Image.open(image)

        # Handle transparency.
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]
        return (img_np - self.mean) / self.std

    def predict(self, image, top_k=5):
        """Return a list of ``(author_name, similarity_score)`` tuples."""
        img_np = self.preprocess(image)
        top_indices, top_scores = self.session.run(None, {"image": img_np})

        results = []
        for idx, score in zip(top_indices[0][:top_k], top_scores[0][:top_k]):
            results.append((self.names[idx], float(score)))
        return results


def _get_model() -> AuthorID:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = AuthorID(_resolve_model_path())
    return _MODEL


def artist_recognition(image, top_k: int = 5) -> dict[str, float]:
    """Recognize artists and return confidence tags keyed by ``@artist``."""
    results = _get_model().predict(image, top_k=top_k)
    return {
        f"@{str(name).lstrip('@')}": round(float(score), 2)
        for name, score in results
        if str(name).strip()
    }

