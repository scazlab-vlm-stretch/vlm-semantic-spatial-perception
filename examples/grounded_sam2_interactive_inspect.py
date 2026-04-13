"""
Interactive Grounded SAM2 + auto-labelling on RealSense.

Modes:
  LIVE    - streaming inference, auto-retags every N frames
  PAUSED  - frozen on last frame; inspect results
  OBJECT  - cycle through detected objects one at a time
              n / right-arrow  next object
              p / left-arrow   prev object
              s                sub-segmentation (enter label prompt in terminal)
              b                back to PAUSED

Sub-segmentation crops to the object bbox, masks out everything outside the
object's segmentation mask, then runs GroundingDINO + SAM2 on the result.

Run from any directory:
  python examples/grounded_sam2_interactive_inspect.py --tagger florence2
"""

import argparse
import os
import time

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image

from grounded_sam2_realsense_ram_demo import (
    Florence2Tagger,
    RAMTagger,
    VLMTagger,
    init_realsense,
    tags_to_prompt,
)
from perception.gsam2 import IncrementalObjectTracker

# ── GPU ───────────────────────────────────────────────────────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Defaults ──────────────────────────────────────────────────────────────────
_GSAM2_ROOT = "/home/liam/installs/Grounded-SAM-2"
_SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
_SAM2_CKPT  = os.path.join(_GSAM2_ROOT, "checkpoints/sam2.1_hiera_large.pt")
_RAM_CKPT   = os.path.join(_GSAM2_ROOT, "ram_checkpoints/ram_plus_swin_large_14m.pth")

_PALETTE = [
    (255, 102,  99), ( 99, 204, 255), (153, 255, 153), (255, 204,  99),
    (204, 153, 255), (255, 153, 204), ( 99, 255, 204), (255, 178, 102),
]


def _color(obj_id: int):
    return _PALETTE[(obj_id - 1) % len(_PALETTE)]


def _to_bool_mask(mask) -> np.ndarray:
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    return np.asarray(mask, dtype=bool)


def isolate_object(rgb: np.ndarray, mask_dict, obj_id: int) -> np.ndarray:
    """Darken everything except obj_id and add a colour tint to its mask."""
    out = rgb.astype(np.float32).copy()
    obj_mask = _to_bool_mask(mask_dict.labels[obj_id].mask)
    out[~obj_mask] *= 0.2
    tint = np.zeros_like(out)
    tint[obj_mask] = _color(obj_id)
    out = np.clip(out + tint * 0.35, 0, 255).astype(np.uint8)

    info = mask_dict.labels[obj_id]
    x1, y1, x2, y2 = int(info.x1), int(info.y1), int(info.x2), int(info.y2)
    color = _color(obj_id)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, f"[{obj_id}] {info.class_name}",
                (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)
    return out


def run_subseg(rgb: np.ndarray, tracker: IncrementalObjectTracker,
               obj_id: int, prompt: str, pad: int = 40):
    """
    Crop to the object bbox, zero out pixels outside the object mask,
    run GroundingDINO + SAM2 with `prompt` on that masked crop.
    Returns (annotated_full_rgb, list[tuple[label, score]]).
    """
    info = tracker.last_mask_dict.labels[obj_id]
    H, W = rgb.shape[:2]
    x1 = max(0, int(info.x1) - pad)
    y1 = max(0, int(info.y1) - pad)
    x2 = min(W, int(info.x2) + pad)
    y2 = min(H, int(info.y2) + pad)

    crop     = rgb[y1:y2, x1:x2].copy()
    obj_mask = _to_bool_mask(info.mask)[y1:y2, x1:x2]
    crop[~obj_mask] = 0

    crop_pil = Image.fromarray(crop)
    boxes, labels = tracker.grounding_predictor.predict(crop_pil, prompt)
    if boxes.shape[0] == 0:
        print("[SubSeg] No components detected.")
        return None, []

    tracker.sam2_segmentor.set_image(crop)
    masks, scores, _ = tracker.sam2_segmentor.predict_masks_from_boxes(boxes)

    flat_scores = scores[:, 0] if scores.ndim > 1 else scores
    detections = sv.Detections(
        xyxy=boxes.cpu().numpy(),
        mask=masks.astype(bool),
        class_id=np.arange(len(labels), dtype=np.int32),
    )
    label_strs = [f"{l} ({s:.2f})" for l, s in zip(labels, flat_scores)]
    annotated_crop = sv.MaskAnnotator().annotate(crop.copy(), detections)
    annotated_crop = sv.BoxAnnotator().annotate(annotated_crop, detections)
    annotated_crop = sv.LabelAnnotator().annotate(annotated_crop, detections, label_strs)

    base = isolate_object(rgb, tracker.last_mask_dict, obj_id)
    base[y1:y2, x1:x2] = annotated_crop
    return base, list(zip(labels, flat_scores.tolist()))


def draw_hud(img_bgr: np.ndarray, mode: str, **kw) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    bar_h = 50
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, H - bar_h), (W, H), (0, 0, 0), -1)
    img_bgr = cv2.addWeighted(overlay, 0.65, img_bgr, 0.35, 0)

    y = H - 16
    if mode == "LIVE":
        txt = "LIVE  |  Space=pause  r=re-detect  q=quit"
        col = (0, 255, 100)
    elif mode == "PAUSED":
        txt = "PAUSED  |  Space=resume  o=object-mode  r=re-detect  q=quit"
        col = (255, 200, 0)
    else:
        idx   = kw.get("obj_idx", 0) + 1
        total = kw.get("total_objs", 1)
        oid   = kw.get("obj_id", "?")
        lbl   = kw.get("obj_label", "")
        txt   = (f"OBJECT [{idx}/{total}]  id={oid}  '{lbl}'  |  "
                 "n=next  p=prev  s=sub-seg  b=back  q=quit")
        col = (99, 200, 255)

    cv2.putText(img_bgr, txt, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
    return img_bgr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tagger", default="florence2", choices=["ram", "vlm", "florence2"])
    p.add_argument("--florence2-model", default="microsoft/Florence-2-large")
    p.add_argument("--vlm-model",       default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--vlm-max-new-tokens", type=int, default=128)
    p.add_argument("--ram-ckpt",        default=_RAM_CKPT)
    p.add_argument("--ram-image-size",  type=int, default=384)
    p.add_argument("--detection-interval", type=int, default=20)
    p.add_argument("--tag-interval",    type=int, default=10)
    p.add_argument("--sam2-cfg",        default=_SAM2_CFG)
    p.add_argument("--sam2-ckpt",       default=_SAM2_CKPT)
    p.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=30)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.tagger == "ram":
        tagger = RAMTagger(args.ram_ckpt, args.ram_image_size, device)
        tagger_label = "RAM+"
    elif args.tagger == "vlm":
        tagger = VLMTagger(args.vlm_model, args.vlm_max_new_tokens, device)
        tagger_label = args.vlm_model.split("/")[-1]
    else:
        tagger = Florence2Tagger(args.florence2_model, device)
        tagger_label = args.florence2_model.split("/")[-1]

    print("[Init] Loading Grounded SAM2 tracker ...")
    tracker = IncrementalObjectTracker(
        grounding_model_id=args.grounding_model,
        sam2_model_cfg=args.sam2_cfg,
        sam2_ckpt_path=args.sam2_ckpt,
        device=device,
        prompt_text="object.",
        detection_interval=args.detection_interval,
    )
    print("[Init] Tracker loaded.")

    pipeline = init_realsense(args.width, args.height, args.fps)

    print(f"\n[Info] Tagger: {tagger_label}  tag-interval: {args.tag_interval}")
    print("[Controls]")
    print("  Space  - pause / resume live stream")
    print("  r      - re-detect on current frame now")
    print("  o      - enter per-object inspection mode")
    print("  q      - quit")
    print("  (object mode)  n/→=next  p/←=prev  s=sub-segment  b=back\n")

    WIN = f"Grounded SAM2 Interactive | {tagger_label}"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    mode           = "LIVE"
    last_rgb       = None
    last_annotated = None
    last_prompt    = "object."
    frame_idx      = 0
    obj_ids        = []
    obj_cursor     = 0
    subseg_result  = None

    try:
        while True:
            if mode == "LIVE":
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                bgr      = np.asanyarray(color_frame.get_data())
                last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if frame_idx % args.tag_interval == 0:
                    t0 = time.perf_counter()
                    new_prompt, raw = tagger.tag(last_rgb)
                    dt = (time.perf_counter() - t0) * 1000
                    print(f"[{tagger_label}] {dt:.0f}ms  raw: {raw}")
                    if new_prompt and new_prompt != last_prompt:
                        last_prompt = new_prompt
                        tracker.set_prompt(last_prompt)

                t0 = time.perf_counter()
                annotated = tracker.add_image(last_rgb)
                dt = (time.perf_counter() - t0) * 1000
                if annotated is not None:
                    last_annotated = annotated
                    print(f"[GSAM] {dt:.0f}ms  frame={frame_idx}")
                frame_idx += 1

            if mode == "LIVE":
                base = last_annotated if last_annotated is not None else last_rgb
                if base is None:
                    cv2.waitKey(1)
                    continue
                display = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
                display = draw_hud(display, "LIVE")

            elif mode == "PAUSED":
                base = last_annotated if last_annotated is not None else last_rgb
                if base is None:
                    cv2.waitKey(50)
                    continue
                display = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
                display = draw_hud(display, "PAUSED")

            else:  # OBJECT
                if not obj_ids or last_rgb is None:
                    mode = "PAUSED"
                    continue
                obj_id = obj_ids[obj_cursor]
                info   = tracker.last_mask_dict.labels.get(obj_id)
                if info is None:
                    mode = "PAUSED"
                    continue

                frame_rgb = subseg_result if subseg_result is not None \
                            else isolate_object(last_rgb, tracker.last_mask_dict, obj_id)
                display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                display = draw_hud(display, "OBJECT",
                                   obj_id=obj_id, obj_label=info.class_name,
                                   total_objs=len(obj_ids), obj_idx=obj_cursor)

            cv2.imshow(WIN, display)

            wait_ms = 1 if mode == "LIVE" else 50
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                if mode == "LIVE":
                    mode = "PAUSED"
                    print("[Paused]")
                elif mode == "PAUSED":
                    mode = "LIVE"
                    print("[Live]")

            elif key == ord("r") and mode in ("LIVE", "PAUSED"):
                if last_rgb is not None:
                    t0 = time.perf_counter()
                    new_prompt, raw = tagger.tag(last_rgb)
                    dt = (time.perf_counter() - t0) * 1000
                    print(f"[{tagger_label}] re-detect {dt:.0f}ms  raw: {raw}")
                    last_prompt = new_prompt
                    tracker.set_prompt(last_prompt)
                    annotated = tracker.add_image(last_rgb)
                    if annotated is not None:
                        last_annotated = annotated
                    mode = "PAUSED"

            elif key == ord("o") and mode == "PAUSED":
                obj_ids = sorted(tracker.last_mask_dict.labels.keys())
                if not obj_ids:
                    print("[Object mode] No objects detected.")
                else:
                    obj_cursor    = 0
                    subseg_result = None
                    mode          = "OBJECT"
                    names = [tracker.last_mask_dict.labels[i].class_name for i in obj_ids]
                    print(f"[Object mode] {len(obj_ids)} objects: {names}")

            elif mode == "OBJECT":
                if key in (ord("n"), 83):
                    obj_cursor    = (obj_cursor + 1) % len(obj_ids)
                    subseg_result = None

                elif key in (ord("p"), 81):
                    obj_cursor    = (obj_cursor - 1) % len(obj_ids)
                    subseg_result = None

                elif key == ord("b"):
                    mode          = "PAUSED"
                    subseg_result = None
                    print("[Back to PAUSED]")

                elif key == ord("s"):
                    obj_id = obj_ids[obj_cursor]
                    lbl    = tracker.last_mask_dict.labels[obj_id].class_name
                    print(f"\n[Sub-seg] Object {obj_id}: '{lbl}'")
                    prompt = input("  Enter component labels (e.g. 'handle. blade. button.'): ").strip()
                    if prompt:
                        if not prompt.endswith("."):
                            prompt += "."
                        print(f"[Sub-seg] Running with prompt: '{prompt}' ...")
                        t0 = time.perf_counter()
                        result, found = run_subseg(last_rgb, tracker, obj_id, prompt)
                        dt = (time.perf_counter() - t0) * 1000
                        if result is not None:
                            subseg_result = result
                            print(f"[Sub-seg] {dt:.0f}ms  found: {[f[0] for f in found]}")
                        else:
                            subseg_result = None
                            print(f"[Sub-seg] {dt:.0f}ms  nothing found.")

    except KeyboardInterrupt:
        print("[Interrupted]")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    main()
