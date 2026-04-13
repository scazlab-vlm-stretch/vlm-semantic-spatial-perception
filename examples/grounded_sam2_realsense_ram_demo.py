"""
Grounded SAM2 + auto-labelling on Intel RealSense RGB stream.

Three tagger backends selectable via --tagger:
  ram       – RAM+ recognition model (default)
  vlm       – Small VLM via transformers (e.g. SmolVLM, Qwen2-VL, PaliGemma)
  florence2 – Microsoft Florence-2 object detection (fast, ~30-50ms on GPU)

The tagger runs every --tag-interval frames to refresh the GroundingDINO prompt.

Controls:
  q  – quit
  r  – force re-tag immediately
  p  – manually enter a prompt override
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image

# ram was written against old transformers where these lived in modeling_utils;
# they moved to pytorch_utils — patch them back before ram imports.
import transformers.modeling_utils as _tmu
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
_tmu.apply_chunking_to_forward = apply_chunking_to_forward
_tmu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
_tmu.prune_linear_layer = prune_linear_layer

from src.perception.gsam2 import IncrementalObjectTracker

# ── GPU settings ──────────────────────────────────────────────────────────────
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounded SAM2 + auto-labelling (RAM+ or VLM) on RealSense"
    )
    # Tagger selection
    parser.add_argument(
        "--tagger", type=str, default="ram", choices=["ram", "vlm", "florence2"],
        help="Tagging backend: 'ram', 'vlm', or 'florence2' (default: ram)",
    )
    parser.add_argument(
        "--tag-interval", type=int, default=1,
        help="Run tagger every N frames to refresh the prompt (default: 60)",
    )

    # RAM+ options
    parser.add_argument(
        "--ram-ckpt", type=str,
        default="/home/liam/installs/Grounded-SAM-2/ram_checkpoints/ram_plus_swin_large_14m.pth",
    )
    parser.add_argument("--ram-image-size", type=int, default=384)

    # VLM options
    parser.add_argument(
        "--vlm-model", type=str,
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="HuggingFace model ID for the VLM tagger (default: SmolVLM-Instruct)",
    )
    parser.add_argument(
        "--vlm-max-new-tokens", type=int, default=128,
        help="Max new tokens for VLM generation (default: 128)",
    )

    # Florence-2 options
    parser.add_argument(
        "--florence2-model", type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model ID for Florence-2 tagger (default: Florence-2-base)",
    )

    # Grounded SAM2
    parser.add_argument("--detection-interval", type=int, default=20)
    parser.add_argument("--sam2-cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-ckpt", type=str,
                        default="/home/liam/installs/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--grounding-model", type=str,
                        default="IDEA-Research/grounding-dino-tiny")

    # RealSense
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    return parser.parse_args()


# ── Shared helper ─────────────────────────────────────────────────────────────

def tags_to_prompt(tags: list[str]) -> str:
    """Convert a list of tag strings to GroundingDINO format: 'person. cup. chair.'"""
    clean = [t.strip().lower() for t in tags if t.strip()]
    return " ".join(t + "." for t in clean)


# ── RAM+ tagger ───────────────────────────────────────────────────────────────

class RAMTagger:
    def __init__(self, ckpt_path: str, image_size: int, device: str):
        from ram.models import ram_plus
        from ram import get_transform

        print(f"[RAM] Loading ram_plus from {ckpt_path} ...")
        self.model = ram_plus(pretrained=ckpt_path, image_size=image_size, vit="swin_l")
        self.model.eval().to(device)
        self.transform = get_transform(image_size=image_size)
        self.device = device
        print("[RAM] Loaded.")

    def tag(self, rgb_image: np.ndarray) -> tuple[str, str]:
        """
        Returns (prompt, raw_response).
        prompt   – GroundingDINO-formatted string
        raw      – original pipe-separated tag string from RAM+
        """
        pil_img = Image.fromarray(rgb_image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            tags, _ = self.model.generate_tag(tensor)
        raw = tags[0] if isinstance(tags, (list, tuple)) else tags
        tag_list = [t.strip() for t in raw.split("|") if t.strip()]
        return tags_to_prompt(tag_list), raw


# ── VLM tagger ────────────────────────────────────────────────────────────────

VLM_PROMPT = (
    "List every distinct object visible in this image as a short comma-separated list "
    "of singular nouns only (e.g. person, cup, chair, laptop). "
    "Do not include descriptions, quantities, or sentences. "
    "Output only the comma-separated list."
)


class VLMTagger:
    def __init__(self, model_id: str, max_new_tokens: int, device: str):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"[VLM] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        print("[VLM] Loaded.")

    def tag(self, rgb_image: np.ndarray) -> tuple[str, str]:
        """
        Returns (prompt, raw_response).
        prompt   – GroundingDINO-formatted string parsed from VLM output
        raw      – raw text generated by the VLM
        """
        pil_img = Image.fromarray(rgb_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[pil_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.processor.decode(generated, skip_special_tokens=True).strip()

        # Parse comma/semicolon/newline separated nouns into a prompt
        tag_list = re.split(r"[,;\n]+", raw)
        return tags_to_prompt(tag_list), raw


# ── Florence-2 tagger ─────────────────────────────────────────────────────────

class Florence2Tagger:
    """
    Uses Microsoft Florence-2 <OD> task to detect objects and extract labels.
    Single forward pass replaces RAM+/VLM for label generation.
    """

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"[Florence2] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        self.model.eval()
        self.device = device
        print("[Florence2] Loaded.")

    def tag(self, rgb_image: np.ndarray) -> tuple[str, str]:
        """
        Returns (prompt, raw_response).
        prompt – GroundingDINO-formatted string of detected object labels
        raw    – comma-separated list of unique detected labels
        """
        pil_img = Image.fromarray(rgb_image)
        task = "<OD>"
        inputs = self.processor(text=task, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )

        decoded = self.processor.decode(output_ids[0], skip_special_tokens=False)
        result = self.processor.post_process_generation(
            decoded,
            task=task,
            image_size=(pil_img.width, pil_img.height),
        )

        # Deduplicate labels while preserving order
        seen = set()
        labels = []
        for lbl in result[task].get("labels", []):
            lbl = lbl.strip().lower()
            if lbl and lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)

        raw = ", ".join(labels)
        return tags_to_prompt(labels), raw


# ── RealSense ─────────────────────────────────────────────────────────────────

def init_realsense(width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    print(f"[RealSense] Stream started: {width}x{height} @ {fps}fps")
    return pipeline


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load tagger ───────────────────────────────────────────────────────────
    if args.tagger == "ram":
        tagger = RAMTagger(args.ram_ckpt, args.ram_image_size, device)
        tagger_label = "RAM+"
    elif args.tagger == "vlm":
        tagger = VLMTagger(args.vlm_model, args.vlm_max_new_tokens, device)
        tagger_label = args.vlm_model.split("/")[-1]
    else:
        tagger = Florence2Tagger(args.florence2_model, device)
        tagger_label = args.florence2_model.split("/")[-1]

    # ── Load Grounded SAM2 tracker ────────────────────────────────────────────
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
    print(f"[Info] Tagger: {tagger_label}  tag-interval: {args.tag_interval}")
    print("[Info] Controls: q=quit  r=re-tag now  p=manual prompt")

    frame_idx = 0
    current_prompt = "object."

    try:
        while True:
            t_frame_start = time.perf_counter()

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # ── Tagger refresh ────────────────────────────────────────────────
            t_tag = 0.0
            if frame_idx % args.tag_interval == 0:
                t0 = time.perf_counter()
                new_prompt, raw_response = tagger.tag(rgb)
                t_tag = time.perf_counter() - t0

                print(f"[{tagger_label}] raw ({t_tag*1000:.1f}ms): {raw_response}")
                print(f"[{tagger_label}] prompt: {new_prompt}")

                if new_prompt and new_prompt != current_prompt:
                    current_prompt = new_prompt
                    tracker.set_prompt(current_prompt)

            # ── Grounded SAM2 tracking ────────────────────────────────────────
            t0 = time.perf_counter()
            annotated = tracker.add_image(rgb)
            t_gsam = time.perf_counter() - t0

            t_total = time.perf_counter() - t_frame_start
            print(
                f"[Timing] frame={frame_idx:05d} | "
                f"tag={t_tag*1000:6.1f}ms | "
                f"GSAM={t_gsam*1000:6.1f}ms | "
                f"total={t_total*1000:6.1f}ms | "
                f"~{1/t_total:.1f}fps"
            )

            if annotated is not None and isinstance(annotated, np.ndarray):
                display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            else:
                display = bgr.copy()

            # ── Overlay HUD ───────────────────────────────────────────────────
            cv2.putText(
                display,
                f"[{tagger_label}] {current_prompt[:75]}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
            )
            next_tag = args.tag_interval - (frame_idx % args.tag_interval)
            cv2.putText(
                display,
                f"GSAM={t_gsam*1000:.0f}ms  total={t_total*1000:.0f}ms  retag in {next_tag}f",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(f"Grounded SAM2 + {tagger_label} | RealSense", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Info] Quit.")
                break
            elif key == ord("r"):
                frame_idx = (frame_idx // args.tag_interval) * args.tag_interval
                print("[Info] Forcing re-tag on next frame.")
                continue
            elif key == ord("p"):
                manual = input("\nEnter prompt override: ").strip()
                if manual:
                    if not manual.endswith("."):
                        manual += "."
                    current_prompt = manual
                    tracker.set_prompt(current_prompt)
                    print(f"[Manual] Prompt set to: {current_prompt}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("[Info] Interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[Done]")


if __name__ == "__main__":
    main()
