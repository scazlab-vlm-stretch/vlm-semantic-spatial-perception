"""
Grounded SAM2 live demo using Intel RealSense RGB stream.

Uses GroundingDINO for text-prompted detection and SAM2 for segmentation/tracking.
Press 'q' to quit, 'p' to change the text prompt at runtime.
"""

import argparse
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# Reuse the tracker class from the existing camera demo
from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker

# ── GPU settings ──────────────────────────────────────────────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded SAM2 + RealSense live demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="person.",
        help='Text prompt for detection (e.g. "person. cup. bottle.")',
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=5,
        help="Run GroundingDINO every N frames; track in between (default: 20)",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="RealSense color stream width"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="RealSense color stream height"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="RealSense color stream FPS"
    )
    parser.add_argument(
        "--sam2-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config path",
    )
    parser.add_argument(
        "--sam2-ckpt",
        type=str,
        default="./checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path",
    )
    parser.add_argument(
        "--grounding-model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID for GroundingDINO",
    )
    return parser.parse_args()


def init_realsense(width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    print(f"[RealSense] Stream started: {width}x{height} @ {fps}fps")
    return pipeline


def get_new_prompt() -> str:
    prompt = input("\nEnter new prompt (e.g. 'cup. bottle.'): ").strip()
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def main():
    args = parse_args()

    # ── Load tracker ──────────────────────────────────────────────────────────
    print(f"[Init] Loading models... prompt='{args.prompt}'")
    tracker = IncrementalObjectTracker(
        grounding_model_id=args.grounding_model,
        sam2_model_cfg=args.sam2_cfg,
        sam2_ckpt_path=args.sam2_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        prompt_text=args.prompt,
        detection_interval=args.detection_interval,
    )
    print("[Init] Models loaded.")

    # ── Start RealSense ───────────────────────────────────────────────────────
    pipeline = init_realsense(args.width, args.height, args.fps)

    print("[Info] Running. Press 'q' to quit, 'p' to change prompt.")
    frame_idx = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # BGR from RealSense → RGB for the tracker
            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            annotated = tracker.add_image(rgb)

            if annotated is not None and isinstance(annotated, np.ndarray):
                display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            else:
                # No detections this frame — show raw feed
                display = bgr

            # Overlay prompt text
            cv2.putText(
                display,
                f"Prompt: {tracker.prompt_text}  |  frame {frame_idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Grounded SAM2 - RealSense", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Info] Quit.")
                break
            elif key == ord("p"):
                new_prompt = get_new_prompt()
                tracker.set_prompt(new_prompt)

            frame_idx += 1

    except KeyboardInterrupt:
        print("[Info] Interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    main()
