"""
Direct Gemini image object detection helpers.

This module keeps object detection separate from `ObjectTracker` for simple
one-off image analysis flows.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROMPT = (
    "Detect all prominent items in the image. "
    'Return a JSON array where each item has keys "label" and "box_2d". '
    'The "box_2d" value must be [ymin, xmin, ymax, xmax] normalized to 0-1000. '
    "Use short descriptive labels."
)


@dataclass
class GeminiDetection:
    """
    Bounding-box detection returned by Gemini.

    Args:
        label: Short object label from the model.
        box_2d: Bounding box as [ymin, xmin, ymax, xmax] on a 0-1000 scale.

    Example:
        >>> det = GeminiDetection(label="red mug", box_2d=[120, 220, 640, 540])
        >>> det.label
        'red mug'
    """

    label: str
    box_2d: List[int]


def _parse_json_payload(text: str) -> object:
    """Parse Gemini JSON output, tolerating markdown fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _coerce_detection(item: object) -> Optional[GeminiDetection]:
    """Validate one model-produced detection item."""
    if not isinstance(item, dict):
        return None

    label = item.get("label")
    box_2d = item.get("box_2d")
    if not isinstance(label, str) or not isinstance(box_2d, Sequence) or len(box_2d) != 4:
        return None

    try:
        coords = [int(float(value)) for value in box_2d]
    except (TypeError, ValueError):
        return None

    y1, x1, y2, x2 = coords
    if y1 >= y2 or x1 >= x2:
        return None

    clipped = [
        max(0, min(1000, y1)),
        max(0, min(1000, x1)),
        max(0, min(1000, y2)),
        max(0, min(1000, x2)),
    ]
    return GeminiDetection(label=label.strip(), box_2d=clipped)


def detect_objects_in_pil_image(
    image: Image.Image,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    max_objects: int = 6,
    prompt: str = DEFAULT_PROMPT,
) -> List[GeminiDetection]:
    """
    Run direct Gemini object detection on a PIL image.

    Args:
        image: Input image.
        api_key: Gemini API key. Defaults to env lookup.
        model_name: Gemini model name.
        max_objects: Maximum detections to return after parsing.
        prompt: Detection prompt sent with the image.

    Returns:
        Parsed detections with normalized boxes.

    Example:
        >>> image = Image.open("scene.png")
        >>> detect_objects_in_pil_image(image, api_key="test-key")
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

    client = genai.Client(api_key=resolved_key)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt],
        config=config,
    )
    payload = _parse_json_payload(response.text or "")
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list from Gemini, got: {type(payload).__name__}")

    detections: List[GeminiDetection] = []
    for item in payload:
        detection = _coerce_detection(item)
        if detection is not None:
            detections.append(detection)
        if len(detections) >= max_objects:
            break
    return detections


def detect_objects_in_image_file(
    image_path: Union[str, Path],
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    max_objects: int = 6,
    prompt: str = DEFAULT_PROMPT,
) -> List[GeminiDetection]:
    """
    Run direct Gemini object detection on an image file.

    Args:
        image_path: Path to the color image file.
        api_key: Gemini API key. Defaults to env lookup.
        model_name: Gemini model name.
        max_objects: Maximum detections to return after parsing.
        prompt: Detection prompt sent with the image.

    Returns:
        Parsed detections with normalized boxes.

    Example:
        >>> detect_objects_in_image_file("scene.png")
    """
    with Image.open(image_path) as image:
        return detect_objects_in_pil_image(
            image=image.convert("RGB"),
            api_key=api_key,
            model_name=model_name,
            max_objects=max_objects,
            prompt=prompt,
        )


def detect_objects_in_rgb_array(
    image_rgb: np.ndarray,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    max_objects: int = 6,
    prompt: str = DEFAULT_PROMPT,
) -> List[GeminiDetection]:
    """
    Run direct Gemini object detection on an RGB numpy image.

    Args:
        image_rgb: RGB image array with shape [H, W, 3].
        api_key: Gemini API key. Defaults to env lookup.
        model_name: Gemini model name.
        max_objects: Maximum detections to return after parsing.
        prompt: Detection prompt sent with the image.

    Returns:
        Parsed detections with normalized boxes.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image_rgb.shape}")
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    return detect_objects_in_pil_image(
        image=image,
        api_key=api_key,
        model_name=model_name,
        max_objects=max_objects,
        prompt=prompt,
    )
