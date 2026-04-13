"""
Minimal rosbridge color-only object detection viewer.

Subscribes to a compressed color image topic over rosbridge, runs VLM object
detection on the latest frame, and draws up to 6 detected object boxes.

Usage:
    uv run examples/test_rosbridge_object_detection.py

Requirements:
    uv add roslibpy python-dotenv
    GEMINI_API_KEY or GOOGLE_API_KEY set in the environment
"""

import asyncio
import base64
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import roslibpy
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception import ObjectTracker

load_dotenv()


ROSBRIDGE_HOST = "192.168.1.23"
ROSBRIDGE_PORT = 9090
COLOR_TOPIC = "/camera/color/image_raw/compressed"

THROTTLE_MS = 100
QUEUE_LENGTH = 1
MAX_OBJECTS = 6


_lock = threading.Lock()
_latest_frame = None
_latest_stamp = None


def _stamp_to_sec(header_stamp):
    return float(header_stamp["secs"]) + float(header_stamp["nsecs"]) * 1e-9


def _decode_compressed_color(msg):
    jpg = base64.b64decode(msg["data"])
    arr = np.frombuffer(jpg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _color_cb(msg):
    global _latest_frame, _latest_stamp
    try:
        frame = _decode_compressed_color(msg)
        stamp = _stamp_to_sec(msg["header"]["stamp"])
        with _lock:
            _latest_frame = frame
            _latest_stamp = stamp
    except Exception as exc:
        print(f"[color_cb] {exc}")


def _get_latest_frame():
    with _lock:
        if _latest_frame is None:
            return None, None
        return _latest_stamp, _latest_frame.copy()


def _draw_detections(image, objects):
    vis = image.copy()
    height, width = vis.shape[:2]

    for obj in objects[:MAX_OBJECTS]:
        if obj.bounding_box_2d:
            y1, x1, y2, x2 = obj.bounding_box_2d
            pt1 = (int((x1 / 1000.0) * width), int((y1 / 1000.0) * height))
            pt2 = (int((x2 / 1000.0) * width), int((y2 / 1000.0) * height))
            cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(
                vis,
                obj.object_id,
                (pt1[0], max(20, pt1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        elif obj.position_2d:
            y_norm, x_norm = obj.position_2d
            center = (
                int((x_norm / 1000.0) * width),
                int((y_norm / 1000.0) * height),
            )
            cv2.circle(vis, center, 8, (0, 255, 0), -1)
            cv2.putText(
                vis,
                obj.object_id,
                (center[0] + 10, max(20, center[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.putText(
        vis,
        f"showing up to {MAX_OBJECTS} objects",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return vis


async def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        return

    ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
    ros.run()

    if not ros.is_connected:
        print(f"Failed to connect to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}")
        return

    color_topic = roslibpy.Topic(
        ros,
        COLOR_TOPIC,
        "sensor_msgs/CompressedImage",
        queue_length=QUEUE_LENGTH,
        throttle_rate=THROTTLE_MS,
    )
    color_topic.subscribe(_color_cb)

    tracker = ObjectTracker(
        api_key=api_key,
        model_name="auto",
        thinking_budget=0,
        max_parallel_requests=5,
        fast_mode=True,
    )

    print("Subscribed.")
    print(f"Color topic: {COLOR_TOPIC}")
    print("Waiting for frames. Press ESC to quit.")

    last_processed_stamp = None
    last_detection_time = 0.0
    detection_interval_sec = 1.0
    display_frame = None

    try:
        while ros.is_connected:
            stamp, frame = _get_latest_frame()
            now = time.time()

            should_detect = (
                frame is not None
                and stamp is not None
                and stamp != last_processed_stamp
                and now - last_detection_time >= detection_interval_sec
            )

            if should_detect:
                raw_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    raw_bgr,
                    "detecting...",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("rosbridge_object_detection", raw_bgr)
                cv2.waitKey(1)

                try:
                    objects = await tracker.detect_objects(frame)
                    objects = objects[:MAX_OBJECTS]
                    print(
                        f"[t={stamp:.2f}] detected {len(objects)} object(s): "
                        f"{[obj.object_id for obj in objects]}"
                    )
                    display_frame = cv2.cvtColor(
                        _draw_detections(frame, objects), cv2.COLOR_RGB2BGR
                    )
                except Exception as exc:
                    print(f"Detection error: {type(exc).__name__}: {exc}")
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                last_processed_stamp = stamp
                last_detection_time = now

            elif display_frame is None and frame is not None:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if display_frame is not None:
                cv2.imshow("rosbridge_object_detection", display_frame)

            if cv2.waitKey(1) == 27:
                break

            await asyncio.sleep(0.01)

    finally:
        color_topic.unsubscribe()
        ros.terminate()
        await tracker.aclose()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
