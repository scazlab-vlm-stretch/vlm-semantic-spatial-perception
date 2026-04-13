## Rosbridge Object Detection Example

- `examples/test_rosbridge_object_detection.py` is intentionally minimal: color-only input, no depth sync, no camera info, no continuous tracker state.
- Keep the visualization simple and aligned with current `ObjectTracker` outputs: draw up to 6 normalized 2D boxes, and fall back to center points if a box is unavailable.
- `examples/test_rosbridge_object_detection_bare.py` bypasses `ObjectTracker` and uses a direct Gemini image-understanding helper in `src/perception/gemini_object_detection.py` for plain box detection from a color image.
