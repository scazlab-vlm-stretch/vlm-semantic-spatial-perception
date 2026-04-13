"""
Perception Predicate Demo
=========================
Live RealSense demo that:
  1. Detects and tracks objects with GSAM2 (GroundingDINO + SAM2).
  2. Evaluates a fixed set of predicates per object using Gemini (VLM).
  3. Registers those predicate schemas in the DKB.
  4. Visualizes each object with its mask, bounding box, and a predicate
     status panel overlaid on the frame.

Predicates evaluated per object
--------------------------------
  - isBroken      : object appears damaged, cracked, or non-functional
  - shouldDiscard : object should be thrown away / removed
  - shouldRestock : object is low / empty and needs replenishing

Usage
-----
    cd /home/liam/dev/vlm-semantic-spatial-perception
    uv run python examples/perception_predicate_demo.py [options]

Press 'q' to quit, 'r' to force a re-evaluation of all predicates.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
import torch
from PIL import Image

# ── Adjust path so src/ and examples/ utils are importable ───────────────────
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))
# grounded_sam2_tracking_camera_with_continuous_id.py imports `utils.common_utils` etc.
# Those modules live in src/perception/gsam2/. Register that directory as the `utils` package.
_gsam2_dir = ROOT / "src" / "perception" / "gsam2"
import types as _types
_utils_pkg = _types.ModuleType("utils")
_utils_pkg.__path__ = [str(_gsam2_dir)]
_utils_pkg.__package__ = "utils"
sys.modules.setdefault("utils", _utils_pkg)

from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker  # noqa: E402

from src.llm_interface.google_genai import GoogleGenAIClient  # noqa: E402
from src.llm_interface.base import GenerateConfig, ImagePart  # noqa: E402
from src.planning.domain_knowledge_base import DomainKnowledgeBase  # noqa: E402

# ── GPU settings ──────────────────────────────────────────────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Predicates evaluated per object ───────────────────────────────────────────
PREDICATES: Dict[str, str] = {
    "isBroken": "The object appears damaged, cracked, broken, or non-functional.",
    "shouldDiscard": "The object should be thrown away, removed, or discarded (e.g. trash, spoiled, broken beyond repair).",
    "shouldRestock": "The object is not broken, looks to be in good condition, can be resold.",
}

# Human-readable labels shown in the predicate panel
PREDICATE_LABELS: Dict[str, str] = {
    "isBroken":     "Broken",
    "shouldDiscard": "Discard",
    "shouldRestock": "Restock",
}

# Colour coding for predicate overlay: True → green, False → red
_TRUE_COLOUR = (50, 205, 50)    # lime green

# How often to re-run predicate evaluation (frames)
DEFAULT_EVAL_INTERVAL = 60


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GSAM2 + Predicate Evaluation Demo")
    p.add_argument("--prompt", type=str, default="object.",
                   help='GroundingDINO text prompt (e.g. "bottle. cup. tool.")')
    p.add_argument("--detection-interval", type=int, default=10,
                   help="Run GroundingDINO every N frames (default: 10)")
    p.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL,
                   help="Re-evaluate predicates every N frames (default: 60)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--sam2-cfg", type=str,
                   default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="SAM2 config (relative to Grounded-SAM-2 install, or absolute path)")
    p.add_argument("--sam2-ckpt", type=str,
                   default="/home/liam/installs/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--grounding-model", type=str,
                   default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--gemini-model", type=str,
                   default="gemini-2.0-flash",
                   help="Gemini model for predicate evaluation")
    p.add_argument("--dkb-dir", type=str, default="outputs/predicate_demo_dkb",
                   help="Directory for DKB persistence")
    return p.parse_args()


# ── RealSense init ─────────────────────────────────────────────────────────────
def init_realsense(width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    print(f"[RealSense] Stream started: {width}x{height} @ {fps}fps")
    return pipeline


# ── DKB predicate schema registration ─────────────────────────────────────────
def register_predicate_schemas(dkb: DomainKnowledgeBase) -> None:
    """Seed the DKB predicate library with the demo predicates."""
    for name, description in PREDICATES.items():
        if name not in dkb._predicates:
            dkb._predicates[name] = {
                "signature": f"({name} ?obj)",
                "description": description,
                "usage_count": 0,
                "arity": 1,
            }
    dkb.save()
    print(f"[DKB] Registered {len(PREDICATES)} predicate schemas: {list(PREDICATES.keys())}")


# ── VLM predicate evaluator ────────────────────────────────────────────────────
class PredicateEvaluator:
    """
    Evaluates a fixed set of boolean predicates for a detected object crop
    using a Gemini VLM.
    """

    _SYSTEM_PROMPT = """\
You are a visual inspection assistant for a robot.
Given an image crop of a single object and its label, evaluate each predicate.
Return ONLY valid JSON with no markdown, no explanation.
"""

    def __init__(self, llm_client: GoogleGenAIClient) -> None:
        self._client = llm_client

    def evaluate(
        self,
        crop_rgb: np.ndarray,
        object_label: str,
        predicates: Dict[str, str],
    ) -> Dict[str, bool]:
        """
        Returns {predicate_name: bool} for each predicate.
        Falls back to all-False on error.
        """
        pil_img = Image.fromarray(crop_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        pred_lines = "\n".join(
            f'  "{name}": {desc}' for name, desc in predicates.items()
        )
        prompt = (
            f"{self._SYSTEM_PROMPT}\n"
            f"Object label: {object_label}\n\n"
            f"Evaluate each predicate as true or false:\n{pred_lines}\n\n"
            f"Return JSON exactly like:\n"
            + json.dumps({name: False for name in predicates}, indent=2)
        )

        cfg = GenerateConfig(
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
        )
        try:
            response = self._client.generate(
                contents=[ImagePart(data=img_bytes, mime_type="image/png"), prompt],
                config=cfg,
            )
            result = json.loads(response.text)
            return {name: bool(result.get(name, False)) for name in predicates}
        except Exception as exc:
            print(f"  [Evaluator] Error for '{object_label}': {exc}")
            return {name: False for name in predicates}


# ── Per-object predicate state ─────────────────────────────────────────────────
class ObjectPredicateState:
    """Tracks evaluated predicate values and staleness per object."""

    def __init__(self) -> None:
        # object_id → {predicate_name: bool}
        self._state: Dict[str, Dict[str, bool]] = {}
        # object_id → last evaluation frame index
        self._last_eval: Dict[str, int] = {}

    def set(self, object_id: str, predicates: Dict[str, bool], frame_idx: int) -> None:
        self._state[object_id] = predicates
        self._last_eval[object_id] = frame_idx

    def get(self, object_id: str) -> Optional[Dict[str, bool]]:
        return self._state.get(object_id)

    def needs_eval(self, object_id: str, frame_idx: int, interval: int) -> bool:
        last = self._last_eval.get(object_id, -interval)
        return (frame_idx - last) >= interval


# ── Visualization helpers ──────────────────────────────────────────────────────
_PALETTE = [
    (255, 56, 56), (255, 157, 51), (255, 255, 51), (51, 255, 51),
    (51, 255, 255), (51, 51, 255), (255, 51, 255), (180, 51, 255),
]


def _colour_for_id(obj_id: int) -> Tuple[int, int, int]:
    return _PALETTE[obj_id % len(_PALETTE)]


def draw_predicate_panel(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    predicates: Optional[Dict[str, bool]],
    colour: Tuple[int, int, int],
    panel_pos: Optional[Tuple[int, int, int]] = None,
) -> None:
    """
    Draw bounding box + label + predicate status panel.
    panel_pos: (px, py, pw) — pre-computed, collision-resolved position.
    Modifies `image` in place.
    """
    # Bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(image, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(image, label, (x1 + 2, y1 - baseline - 2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    if not predicates:
        return

    panel_line_h = 22
    panel_h = len(predicates) * panel_line_h + 8

    if panel_pos is not None:
        px, py, panel_w = panel_pos
    else:
        H, W = image.shape[:2]
        panel_w = min(max(x2 - x1, 110), W - x1)
        px = x1
        py = y2 + 4 if y2 + 4 + panel_h <= H else max(0, y1 - panel_h - 4)

    # Connector line from box edge to panel if the panel has been nudged away
    attach_x = x1 + (x2 - x1) // 2  # centre-bottom of box
    default_py = y2 + 4
    if py != default_py:
        panel_mid_x = px + panel_w // 2
        cv2.line(image, (attach_x, y2), (panel_mid_x, py), colour, 1, cv2.LINE_AA)

    # Panel background
    overlay = image.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for i, (pred_name, value) in enumerate(predicates.items()):
        row_y = py + 8 + i * panel_line_h + panel_line_h // 2
        display_label = PREDICATE_LABELS.get(pred_name, pred_name)
        if value:
            symbol = "YES"
            symbol_colour = _TRUE_COLOUR
            text_colour = (220, 220, 220)
        else:
            symbol = "NO"
            symbol_colour = (100, 100, 100)
            text_colour = (120, 120, 120)
        line = f"{display_label}:"
        cv2.putText(image, line, (px + 8, row_y + 5),
                    font, 0.48, text_colour, 1, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(line, font, 0.48, 1)
        cv2.putText(image, symbol, (px + 10 + lw, row_y + 5),
                    font, 0.48, symbol_colour, 1, cv2.LINE_AA)


def _resolve_panel_positions(
    panels: List[Tuple[int, int, int, int]],
    H: int,
    W: int,
    max_passes: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Nudge panel rects (px, py, pw, ph) apart so they don't overlap.
    Iteratively pushes overlapping panels away from each other vertically,
    clamping to frame bounds.
    """
    rects = list(panels)  # [(px, py, pw, ph), ...]
    for _ in range(max_passes):
        moved = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                ax, ay, aw, ah = rects[i]
                bx, by, bw, bh = rects[j]
                # Check horizontal overlap
                if ax >= bx + bw or bx >= ax + aw:
                    continue
                # Check vertical overlap
                overlap = min(ay + ah, by + bh) - max(ay, by)
                if overlap <= 0:
                    continue
                # Push the lower panel down (or upper panel up) by half the overlap
                shift = overlap // 2 + 1
                if ay <= by:
                    new_by = min(by + shift, H - bh)
                    if new_by != by:
                        rects[j] = (bx, new_by, bw, bh)
                        moved = True
                else:
                    new_ay = min(ay + shift, H - ah)
                    if new_ay != ay:
                        rects[i] = (ax, new_ay, aw, ah)
                        moved = True
        if not moved:
            break
    return rects


def annotate_frame(
    rgb: np.ndarray,
    mask_dict,
    pred_state: ObjectPredicateState,
) -> np.ndarray:
    """
    Draw coloured masks + per-object predicate panels on `rgb`.
    Returns annotated RGB image.
    """
    if mask_dict is None or not mask_dict.labels:
        return rgb

    annotated = rgb.copy()
    H, W = annotated.shape[:2]

    # Build supervision Detections for mask overlay
    boxes, masks_list, ids, class_names = [], [], [], []

    for obj_id_int, obj_info in mask_dict.labels.items():
        mask = obj_info.mask
        if mask is None:
            continue
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue

        x1 = int(obj_info.x1) if obj_info.x1 is not None else 0
        y1 = int(obj_info.y1) if obj_info.y1 is not None else 0
        x2 = int(obj_info.x2) if obj_info.x2 is not None else W
        y2 = int(obj_info.y2) if obj_info.y2 is not None else H

        boxes.append([x1, y1, x2, y2])
        masks_list.append(mask_bool[None])  # (1, H, W)
        ids.append(obj_id_int)
        class_names.append(obj_info.class_name or "object")

    if not boxes:
        return annotated

    all_masks = np.concatenate(masks_list, axis=0)
    detections = sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        mask=all_masks,
        class_id=np.array(ids, dtype=np.int32),
    )

    # Draw masks with supervision
    mask_annotator = sv.MaskAnnotator(opacity=0.35)
    annotated = mask_annotator.annotate(annotated, detections)

    # Compute default panel positions
    panel_line_h = 22
    panel_meta = []
    for idx, obj_id_int in enumerate(ids):
        x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
        pw = max(x2 - x1, 110)
        class_name = class_names[idx]
        safe_name = class_name.strip().lower().replace(" ", "_")
        object_id = f"{safe_name}_{obj_id_int}"
        predicates = pred_state.get(object_id)
        n_rows = len(predicates) if predicates else 0
        ph = n_rows * panel_line_h + 8
        px = x1
        py = y2 + 4 if y2 + 4 + ph <= H else max(0, y1 - ph - 4)
        pw = min(pw, W - px)
        panel_meta.append((obj_id_int, class_name, object_id, x1, y1, x2, y2, px, py, pw, ph))

    # Resolve overlaps among panels
    raw_rects = [(m[7], m[8], m[9], m[10]) for m in panel_meta]
    resolved = _resolve_panel_positions(raw_rects, H, W)

    # Draw bounding boxes + predicate panels
    for idx, obj_id_int in enumerate(ids):
        obj_id_int, class_name, object_id, x1, y1, x2, y2, _, _, _, _ = panel_meta[idx]
        px, py, pw, ph = resolved[idx]
        colour = _colour_for_id(obj_id_int)
        label = f"[{obj_id_int}] {class_name}"
        predicates = pred_state.get(object_id)
        draw_predicate_panel(annotated, x1, y1, x2, y2, label, predicates, colour,
                             panel_pos=(px, py, pw))

    return annotated


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    # ── DKB ──────────────────────────────────────────────────────────────────
    dkb = DomainKnowledgeBase(Path(args.dkb_dir))
    dkb.load()
    register_predicate_schemas(dkb)

    # ── LLM + Evaluator ───────────────────────────────────────────────────────
    print(f"[Init] Loading Gemini model '{args.gemini_model}' …")
    llm_client = GoogleGenAIClient(model=args.gemini_model, api_key=api_key)
    evaluator = PredicateEvaluator(llm_client)
    pred_state = ObjectPredicateState()

    # ── GSAM2 tracker ─────────────────────────────────────────────────────────
    print(f"[Init] Loading GSAM2 tracker (prompt='{args.prompt}') …")
    tracker = IncrementalObjectTracker(
        grounding_model_id=args.grounding_model,
        sam2_model_cfg=args.sam2_cfg,
        sam2_ckpt_path=args.sam2_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        prompt_text=args.prompt,
        detection_interval=args.detection_interval,
    )
    print("[Init] Models loaded.")

    # ── RealSense ─────────────────────────────────────────────────────────────
    pipeline = init_realsense(args.width, args.height, args.fps)

    print("[Info] Running. Press 'q' to quit, 'r' to force predicate re-evaluation.")
    frame_idx = 0
    force_eval = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # ── GSAM2 tracking step ───────────────────────────────────────────
            tracker.add_image(rgb)
            mask_dict = tracker.last_mask_dict

            # ── Predicate evaluation ──────────────────────────────────────────
            if mask_dict is not None and mask_dict.labels:
                H, W = rgb.shape[:2]
                for obj_id_int, obj_info in mask_dict.labels.items():
                    class_name = (obj_info.class_name or "object").strip().lower().replace(" ", "_")
                    object_id = f"{class_name}_{obj_id_int}"

                    if not (force_eval or pred_state.needs_eval(object_id, frame_idx, args.eval_interval)):
                        continue

                    # Crop around the bounding box (with padding)
                    x1 = max(0, int(obj_info.x1 or 0) - 10)
                    y1 = max(0, int(obj_info.y1 or 0) - 10)
                    x2 = min(W, int(obj_info.x2 or W) + 10)
                    y2 = min(H, int(obj_info.y2 or H) + 10)
                    crop = rgb[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    label = f"{class_name} (id={obj_id_int})"
                    print(f"  [Eval] Evaluating predicates for '{label}' …")
                    t0 = time.time()
                    result = evaluator.evaluate(crop, label, PREDICATES)
                    elapsed = time.time() - t0
                    pred_state.set(object_id, result, frame_idx)
                    print(f"  [Eval] {label}: {result}  ({elapsed:.1f}s)")

                    # Update DKB usage count for triggered predicates
                    for pred_name, value in result.items():
                        if value and pred_name in dkb._predicates:
                            dkb._predicates[pred_name]["usage_count"] = (
                                dkb._predicates[pred_name].get("usage_count", 0) + 1
                            )
                    dkb.save()

            force_eval = False

            # ── Visualization ─────────────────────────────────────────────────
            annotated_rgb = annotate_frame(rgb, mask_dict, pred_state)
            display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Perception Predicate Demo", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Info] Quit.")
                break
            elif key == ord("r"):
                print("[Info] Forcing predicate re-evaluation on next frame.")
                force_eval = True

            frame_idx += 1

    except KeyboardInterrupt:
        print("[Info] Interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[Done] DKB saved to {args.dkb_dir}")


if __name__ == "__main__":
    main()
