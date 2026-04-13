import base64
import cv2
import numpy as np
import roslibpy
import threading
from collections import deque

# -----------------------------
# Config
# -----------------------------
ROSBRIDGE_HOST = '192.168.1.23'
ROSBRIDGE_PORT = 9090

COLOR_TOPIC = '/camera/color/image_raw/compressed'

THROTTLE_MS = 100
QUEUE_LENGTH = 1
SLOP_SEC = 0.1

# -----------------------------
# Shared state
# -----------------------------
lock = threading.Lock()

color_buffer = deque(maxlen=10)   # stores (stamp_sec, frame); deque drops old items automatically once it hits max length

latest_pair = None                # stores (stamp_sec, color_frame, depth_frame, dt), dt is the time difference between when color & depth frames were recieved.


def stamp_to_sec(header_stamp):
    # ROS header stamp is usually {'secs': ..., 'nsecs': ...}
    return float(header_stamp['secs']) + float(header_stamp['nsecs']) * 1e-9


def decode_compressed_color(msg):
    jpg = base64.b64decode(msg['data'])
    arr = np.frombuffer(jpg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR) #decompresses jpeg/png into an image arr. IMREAD_COLOR tells to decode image as / covert to 3-channel coloor img
    return frame


def color_cb(msg):
    global latest_pair
    try:
        frame = decode_compressed_color(msg)
        if frame is None:
            return

        stamp = stamp_to_sec(msg['header']['stamp'])

        with lock:
            latest_pair = (0, frame, 0, 0)
    except Exception as e:
        print(f'Color callback error: {e}')


ros = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
ros.run()

color_topic = roslibpy.Topic(
    ros,
    COLOR_TOPIC,
    'sensor_msgs/CompressedImage',
    queue_length=QUEUE_LENGTH,
    throttle_rate=THROTTLE_MS,
)


color_topic.subscribe(color_cb)

print('Subscribed.')
print(f'Color topic: {COLOR_TOPIC}')
print(f'Slop: {SLOP_SEC} sec')

while ros.is_connected:
    pair = None
    with lock:
        if latest_pair is not None:
            # stamp_sec, color_frame, depth_frame, dt = latest_pair
            # pair = (stamp_sec, color_frame.copy(), depth_frame.copy(), dt) #copy gives you a stable snapshot of the frame / avoiding sharing the same mutable NumPy array outside the lock
            pair = latest_pair

    if pair is not None:
        stamp_sec, color_frame, depth_frame, dt = pair

        # Show sync error on the images
        text = f'dt={dt:.3f}s'
        cv2.putText(color_frame, text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('color_aligned', color_frame)

    if cv2.waitKey(1) == 27:
        break

color_topic.unsubscribe()
ros.terminate()
cv2.destroyAllWindows()