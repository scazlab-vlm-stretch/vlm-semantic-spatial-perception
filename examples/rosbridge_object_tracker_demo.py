"""
ROS Bridge Object Tracker Demo

Receives color and depth frames from a RealSense camera over a rosbridge
WebSocket connection and runs VLM-based object detection using ObjectTracker.

Usage:
    python examples/rosbridge_object_tracker_demo.py

Requirements:
    pip install roslibpy
    GEMINI_API_KEY or GOOGLE_API_KEY set in .env or environment
"""

import os
import sys
import base64
import threading
import asyncio
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from dotenv import load_dotenv
import roslibpy

load_dotenv()

# -----------------------------
# ROS Bridge Config
# -----------------------------
ROSBRIDGE_HOST = '192.168.1.23'
ROSBRIDGE_PORT = 9090

COLOR_TOPIC = '/camera/color/image_raw/compressed'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'

THROTTLE_MS = 50       # ms between messages (lower = more CPU)
QUEUE_LENGTH = 1
SLOP_SEC = 0.1          # max time diff (s) for color/depth to be considered synced

# Fallback intrinsics if camera_info topic is unavailable
# These approximate a RealSense D435i at 640x480
FALLBACK_INTRINSICS = {
    'fx': 615.0, 'fy': 615.0,
    'cx': 320.0, 'cy': 240.0,
    'width': 640, 'height': 480,
}

# -----------------------------
# Shared State
# -----------------------------
_lock = threading.Lock()

_color_buffer = deque(maxlen=10)
_depth_buffer = deque(maxlen=10)
_latest_pair = None       # (stamp_sec, color_rgb, depth_meters, dt)
_camera_intrinsics = None
_frame_ready = threading.Event()


# -----------------------------
# Decode helpers
# -----------------------------

def _stamp_to_sec(header_stamp):
    return float(header_stamp['secs']) + float(header_stamp['nsecs']) * 1e-9


def _decode_compressed_color(msg) -> np.ndarray:
    """Decode a CompressedImage msg into an RGB numpy array."""
    jpg = base64.b64decode(msg['data'])
    arr = np.frombuffer(jpg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('cv2.imdecode returned None')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_depth_image(msg) -> np.ndarray:
    """Decode a raw sensor_msgs/Image depth msg into float32 meters."""
    height = msg['height']
    width = msg['width']
    encoding = msg['encoding']
    data = msg['data']

    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)

    if encoding == '16UC1':
        depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape((height, width))
        return depth_mm.astype(np.float32) / 1000.0
    elif encoding == '32FC1':
        return np.frombuffer(raw, dtype=np.float32).reshape((height, width))
    else:
        raise ValueError(f'Unsupported depth encoding: {encoding}')


def _parse_camera_info(msg):
    """Parse a sensor_msgs/CameraInfo msg into CameraIntrinsics."""
    from src.camera.base_camera import CameraIntrinsics
    K = msg['K']  # row-major 3x3, flattened
    return CameraIntrinsics(
        fx=float(K[0]),
        fy=float(K[4]),
        cx=float(K[2]),
        cy=float(K[5]),
        width=int(msg['width']),
        height=int(msg['height']),
    )


# -----------------------------
# Frame sync
# -----------------------------

def _sync_frames():
    """Match closest color/depth pair within SLOP_SEC. Must be called under _lock."""
    global _latest_pair

    if not _color_buffer or not _depth_buffer:
        return

    best = None
    best_diff = None

    for i, (tc, color) in enumerate(_color_buffer):
        for j, (td, depth) in enumerate(_depth_buffer):
            diff = abs(tc - td)
            if diff <= SLOP_SEC and (best_diff is None or diff < best_diff):
                best = (i, j, tc, td, color, depth)
                best_diff = diff

    if best is None:
        return

    i, j, tc, td, color, depth = best
    _latest_pair = (max(tc, td), color.copy(), depth.copy(), abs(tc - td))
    _frame_ready.set()

    for _ in range(i + 1):
        _color_buffer.popleft()
    for _ in range(j + 1):
        _depth_buffer.popleft()


# -----------------------------
# ROS callbacks
# -----------------------------

def _color_cb(msg):
    try:
        frame = _decode_compressed_color(msg)
        stamp = _stamp_to_sec(msg['header']['stamp'])
        with _lock:
            _color_buffer.append((stamp, frame))
            _sync_frames()
    except Exception as e:
        print(f'[color_cb] {e}')


def _depth_cb(msg):
    try:
        depth = _decode_depth_image(msg)
        stamp = _stamp_to_sec(msg['header']['stamp'])
        with _lock:
            _depth_buffer.append((stamp, depth))
            _sync_frames()
    except Exception as e:
        print(f'[depth_cb] {e}')


def _camera_info_cb(msg):
    global _camera_intrinsics
    with _lock:
        if _camera_intrinsics is not None:
            return
    try:
        intrinsics = _parse_camera_info(msg)
        with _lock:
            _camera_intrinsics = intrinsics
        print(f'Camera intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} '
              f'cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f} '
              f'({intrinsics.width}x{intrinsics.height})')
    except Exception as e:
        print(f'[camera_info_cb] {e}')


# -----------------------------
# Snapshot helpers
# -----------------------------

def get_latest_snapshot():
    """Return a stable copy of the latest synced pair, or None."""
    with _lock:
        if _latest_pair is None:
            return None
        stamp, color, depth, dt = _latest_pair
        return stamp, color.copy(), depth.copy(), dt


def get_intrinsics():
    """Return current CameraIntrinsics, falling back to hardcoded defaults."""
    with _lock:
        if _camera_intrinsics is not None:
            return _camera_intrinsics
    from src.camera.base_camera import CameraIntrinsics
    print('Using fallback camera intrinsics (subscribe to camera_info for accurate values)')
    return CameraIntrinsics(**FALLBACK_INTRINSICS)


# -----------------------------
# Visualization (same as object_tracker_demo.py)
# -----------------------------

def visualize_detections(image: np.ndarray, objects: list) -> np.ndarray:
    vis = image.copy()
    height, width = vis.shape[:2]

    affordance_colors = {
        'graspable':   (0, 255, 0),
        'pourable':    (255, 0, 0),
        'containable': (0, 0, 255),
        'pushable':    (255, 255, 0),
        'pullable':    (255, 0, 255),
        'openable':    (0, 255, 255),
        'supportable': (128, 128, 128),
    }

    for obj in objects:
        if obj.position_2d:
            y_norm, x_norm = obj.position_2d
            cx = int((x_norm / 1000.0) * width)
            cy = int((y_norm / 1000.0) * height)

            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)

            label = obj.object_id
            if obj.position_3d is not None:
                label += f' ({obj.position_3d[2]:.2f}m)'
            cv2.putText(vis, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if obj.bounding_box_2d:
            y1, x1, y2, x2 = obj.bounding_box_2d
            pt1 = (int((x1 / 1000.0) * width), int((y1 / 1000.0) * height))
            pt2 = (int((x2 / 1000.0) * width), int((y2 / 1000.0) * height))
            cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)

        if obj.interaction_points:
            for affordance, point in obj.interaction_points.items():
                y_norm, x_norm = point.position_2d
                px = int((x_norm / 1000.0) * width)
                py = int((y_norm / 1000.0) * height)
                color = affordance_colors.get(affordance, (255, 255, 255))
                cv2.drawMarker(vis, (px, py), color,
                               markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cv2.putText(vis, affordance[:4], (px + 12, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis


def print_detection_results(objects: list):
    print('\n' + '=' * 60)
    print('DETECTION RESULTS')
    print('=' * 60)

    if not objects:
        print('No objects detected')
        return

    for i, obj in enumerate(objects, 1):
        print(f'\n{i}. {obj.object_id} ({obj.object_type})')

        if obj.position_2d:
            print(f'   Position 2D: {obj.position_2d}')
        if obj.position_3d is not None:
            x, y, z = obj.position_3d
            print(f'   Position 3D: [{x:.3f}, {y:.3f}, {z:.3f}] m')

        print(f'   Affordances ({len(obj.affordances)}):')
        for affordance in sorted(obj.affordances):
            print(f'      • {affordance}', end='')
            if affordance in obj.interaction_points:
                print(f' → {obj.interaction_points[affordance].position_2d}')
            else:
                print()


# -----------------------------
# Main
# -----------------------------

async def main():
    print('=' * 60)
    print('ROSBRIDGE OBJECT TRACKER DEMO')
    print('=' * 60)

    # --- API key ---
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print('GEMINI_API_KEY or GOOGLE_API_KEY not set')
        return

    # --- Connect to ROS bridge ---
    print(f'\nConnecting to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}...')
    ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
    ros.run()

    if not ros.is_connected:
        print('Failed to connect to rosbridge')
        return
    print('Connected.')

    color_topic = roslibpy.Topic(
        ros, COLOR_TOPIC, 'sensor_msgs/CompressedImage',
        queue_length=QUEUE_LENGTH, throttle_rate=THROTTLE_MS,
    )
    depth_topic = roslibpy.Topic(
        ros, DEPTH_TOPIC, 'sensor_msgs/Image',
        queue_length=QUEUE_LENGTH, throttle_rate=THROTTLE_MS,
    )
    camera_info_topic = roslibpy.Topic(
        ros, CAMERA_INFO_TOPIC, 'sensor_msgs/CameraInfo',
        queue_length=1, throttle_rate=1000,  # camera_info is static; 1Hz is fine
    )

    color_topic.subscribe(_color_cb)
    depth_topic.subscribe(_depth_cb)
    camera_info_topic.subscribe(_camera_info_cb)

    print(f'Subscribed to:')
    print(f'  color: {COLOR_TOPIC}')
    print(f'  depth: {DEPTH_TOPIC}')
    print(f'  info:  {CAMERA_INFO_TOPIC}')

    # --- Wait for first synced frame pair ---
    print('\nWaiting for first synced frame pair...')
    while not _frame_ready.wait(timeout=1.0):
        if not ros.is_connected:
            print('Lost connection while waiting for frames')
            ros.terminate()
            return
        print('  still waiting...')
    print('Got first frame pair.')

    # --- Initialize ObjectTracker ---
    print('\nInitializing ObjectTracker...')
    from src.perception import ObjectTracker

    try:
        tracker = ObjectTracker(
            api_key=api_key,
            model_name='auto',
            thinking_budget=0,
            max_parallel_requests=5,
            fast_mode=True
        )
        print(f'ObjectTracker ready (model: {tracker.model_name})')
    except Exception as e:
        print(f'Failed to initialize ObjectTracker: {type(e).__name__}: {e}')
        ros.terminate()
        return

    intrinsics = get_intrinsics()

    print('\nStarting detection loop. Press ESC to quit.\n')

    try:
        while ros.is_connected:
            # --- Grab latest synced pair ---
            snapshot = get_latest_snapshot()
            if snapshot is None:
                await asyncio.sleep(0.05)
                continue

            stamp, color_frame, depth_frame, dt = snapshot
            _frame_ready.clear()

            # --- Display input ---
            depth_vis = (
                cv2.convertScaleAbs(
                    (depth_frame * 1000).astype(np.uint16), alpha=0.03
                )
                if depth_frame.dtype != np.float32
                else cv2.normalize(
                    np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0),
                    None, 0, 255, cv2.NORM_MINMAX,
                ).astype(np.uint8)
            )
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            dt_text = f'dt={dt:.3f}s'
            color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(color_bgr, dt_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(depth_vis, dt_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow('color', color_bgr)
            cv2.imshow('depth', depth_vis)

            if cv2.waitKey(1) == 27:
                break

            # --- Run detection ---
            print(f'[t={stamp:.2f}] Running detection (sync_dt={dt:.3f}s)...')
            try:
                objects = await tracker.detect_objects(
                    color_frame, depth_frame, intrinsics
                )
            except Exception as e:
                print(f'Detection error: {type(e).__name__}: {e}')
                await asyncio.sleep(1.0)
                continue

            print_detection_results(objects)

            # --- Visualize detections ---
            if objects:
                vis = visualize_detections(color_frame, objects)
                cv2.imshow('detections', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # Wait for a new frame before detecting again
            _frame_ready.wait(timeout=5.0)

    finally:
        color_topic.unsubscribe()
        depth_topic.unsubscribe()
        camera_info_topic.unsubscribe()
        ros.terminate()
        await tracker.aclose()
        cv2.destroyAllWindows()
        print('Done.')


if __name__ == '__main__':
    asyncio.run(main())
