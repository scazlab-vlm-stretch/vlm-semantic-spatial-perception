"""
GSAM2-based object tracker that replaces LLM detection with RAM+ + GroundingDINO + SAM2.

Provides GSAM2ObjectTracker (async detect_objects compatible with ObjectTracker) and
GSAM2ContinuousObjectTracker (drop-in for ContinuousObjectTracker).
"""

import asyncio
import io
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
from PIL import Image

from pathlib import Path

from .gsam2 import IncrementalObjectTracker
from .object_registry import DetectedObject, DetectedObjectRegistry, InteractionPoint
from .object_tracker import ContinuousObjectTracker, ObjectTracker, TrackingStats, save_debug_frame
from .utils.coordinates import compute_3d_position, pixel_to_normalized
from ..utils.logging_utils import get_structured_logger

import re
from typing import Optional as _Optional

# Labels that GroundingDINO sometimes produces that don't represent real objects.
_JUNK_LABELS = frozenset({
    "image", "photo", "picture", "frame", "background", "scene", "view",
    "object", "thing", "item", "area", "region", "part", "surface",
    "unknown", "none", "null", "",
})

# Simple stopwords to skip when extracting object hints from task text
_STOPWORDS = frozenset({
    "a", "an", "the", "put", "place", "move", "pick", "up", "on", "onto",
    "in", "into", "to", "from", "and", "or", "of", "with", "get", "take",
    "stack", "grasp", "bring", "is", "are", "it", "its", "that", "this",
})


def _transform_cam_to_world(
    cam_pos: np.ndarray,
    robot_state: _Optional[dict],
) -> _Optional[np.ndarray]:
    """
    Transform a camera-frame 3-D point to world/robot-base frame.

    Reads the camera transform from robot_state["camera"] which must contain:
      - "position":        [x, y, z]  camera origin in world frame
      - "quaternion_xyzw": [qx, qy, qz, qw]  camera orientation in world frame

    Returns the world-frame position, or None if the transform is unavailable.
    """
    if robot_state is None:
        return None
    cam_tf = robot_state.get("camera")
    if cam_tf is None:
        return None
    try:
        from scipy.spatial.transform import Rotation
        cam_origin = np.array(cam_tf["position"], dtype=float)
        cam_rot    = Rotation.from_quat(cam_tf["quaternion_xyzw"])
        return cam_rot.apply(cam_pos) + cam_origin
    except Exception:
        return None


def _extract_noun_phrases(text: str) -> List[str]:
    """
    Extract candidate object noun phrases from a task description.

    Uses a simple adjective+noun pattern (e.g. "red block", "blue block")
    without requiring NLP dependencies.  Returns lowercased multi-word phrases
    and single content words that are not stopwords.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    phrases: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _STOPWORDS:
            i += 1
            continue
        # If next word is also a content word, emit a two-word phrase
        if i + 1 < len(words) and words[i + 1] not in _STOPWORDS:
            phrases.append(f"{w} {words[i + 1]}")
            i += 2
        else:
            phrases.append(w)
            i += 1
    return phrases


def _build_ram_tagger(ckpt_path: str, image_size: int, device: str):
    """Load RAM+ tagger and return a callable tag(rgb_np) -> (prompt_str, raw_str)."""
    import torch
    from ram.models import ram_plus
    from ram import get_transform

    model = ram_plus(pretrained=ckpt_path, image_size=image_size, vit="swin_l")
    model.eval().to(device)
    transform = get_transform(image_size=image_size)

    def _tags_to_prompt(tags: list) -> str:
        clean = [t.strip().lower() for t in tags if t.strip()]
        return " ".join(t + "." for t in clean)

    def tag(rgb_image: np.ndarray):
        import torch
        pil_img = Image.fromarray(rgb_image)
        tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            tags, _ = model.generate_tag(tensor)
        raw = tags[0] if isinstance(tags, (list, tuple)) else tags
        tag_list = [t.strip() for t in raw.split("|") if t.strip()]
        return _tags_to_prompt(tag_list), raw

    return tag


class GSAM2ObjectTracker:
    """
    Object tracker using RAM+ → GroundingDINO → SAM2 instead of an LLM.

    Compatible drop-in for the detect_objects() interface of ObjectTracker.
    Maintains a DetectedObjectRegistry populated from SAM2 mask outputs.

    Args:
        sam2_model_cfg: SAM2 config file path
        sam2_ckpt_path: SAM2 checkpoint path
        grounding_model_id: GroundingDINO HuggingFace model ID
        ram_ckpt_path: RAM+ checkpoint path (if None, tagger must be set manually)
        ram_image_size: Image size for RAM+ preprocessing
        detection_interval: Frames between GroundingDINO re-detections
        device: torch device string
        tag_interval: Frames between RAM+ re-tagging (1 = every frame)
        logger: Optional logger
    """

    # Schema for LLM affordance inference responses
    _AFFORDANCE_SCHEMA = {
        "type": "object",
        "properties": {
            "affordances": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of affordance labels for this object",
            }
        },
        "required": ["affordances"],
    }

    # Schema for LLM predicate evaluation responses
    _PREDICATE_SCHEMA = {
        "type": "object",
        "properties": {
            "predicates": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of ground predicate strings that hold TRUE for this object. "
                    "Unary: 'predicate_name object_id'. "
                    "Binary: 'predicate_name object_id other_object_id'."
                ),
            }
        },
        "required": ["predicates"],
    }

    def __init__(
        self,
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path: str = "./checkpoints/sam2.1_hiera_large.pt",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        ram_ckpt_path: Optional[str] = None,
        ram_image_size: int = 384,
        detection_interval: int = 20,
        device: str = "cuda",
        tag_interval: int = 1,
        llm_client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or get_structured_logger("GSAM2ObjectTracker")
        self.device = device
        self.tag_interval = tag_interval
        self.registry = DetectedObjectRegistry()
        self._llm_client = llm_client
        # Cache affordances by object type (safe_name) to avoid redundant LLM calls
        self._affordance_cache: Dict[str, Set[str]] = {}
        # PDDL predicates to evaluate per object (set via set_pddl_predicates)
        self._pddl_predicates: List[str] = []

        self.logger.info("Loading GroundingDINO (%s) + SAM2 (%s) on %s…",
                         grounding_model_id, sam2_ckpt_path, device)
        self._gsam2 = IncrementalObjectTracker(
            grounding_model_id=grounding_model_id,
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            device=device,
            prompt_text="object.",
            detection_interval=detection_interval,
        )
        self.logger.info("GroundingDINO + SAM2 loaded.")

        self._tagger = None
        if ram_ckpt_path is not None:
            self.logger.info("Loading RAM+ tagger from %s…", ram_ckpt_path)
            self._tagger = _build_ram_tagger(ram_ckpt_path, ram_image_size, device)
            self.logger.info("RAM+ tagger loaded.")

        self._current_prompt: str = "object."
        self._frame_count: int = 0
        self._extra_tags: List[str] = []

    def set_tagger(self, tagger_callable):
        """Override the RAM+ tagger with any callable tag(rgb_np) -> (prompt, raw)."""
        self._tagger = tagger_callable

    def set_pddl_predicates(self, predicates: List[str]) -> None:
        """Store PDDL predicate signatures to evaluate per detected object."""
        self._pddl_predicates = list(predicates or [])
        self.logger.debug("GSAM2ObjectTracker: %d PDDL predicates configured", len(self._pddl_predicates))

    def set_extra_tags(self, tags: List[str]) -> None:
        """Inject additional search terms (e.g. from task description) into the GroundingDINO prompt."""
        self._extra_tags = [t.strip().lower() for t in tags if t.strip()]
        # Immediately merge into current prompt so the next frame uses them
        self._current_prompt = self._merge_prompt(self._current_prompt, self._extra_tags)
        self._gsam2.set_prompt(self._current_prompt)
        self.logger.debug("Extra tags set: %s → prompt: %s", self._extra_tags, self._current_prompt)

    @staticmethod
    def _merge_prompt(ram_prompt: str, extra_tags: List[str]) -> str:
        """Append extra tags to RAM prompt, deduplicating existing terms."""
        existing = {t.rstrip(".").strip() for t in ram_prompt.split() if t.strip()}
        additions = [t for t in extra_tags if t not in existing]
        if not additions:
            return ram_prompt
        extra_str = " ".join(t + "." for t in additions)
        return (ram_prompt.rstrip() + " " + extra_str).strip()

    async def detect_objects(
        self,
        color_frame: Union[np.ndarray, Image.Image],
        depth_frame: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Any] = None,
        temperature: Optional[float] = None,
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> List[DetectedObject]:
        """
        Detect and segment objects using RAM+ → GroundingDINO → SAM2.

        Args:
            color_frame: RGB image (np.ndarray HxWx3 or PIL Image)
            depth_frame: Optional depth image in metres (same HxW as color_frame)
            camera_intrinsics: Camera intrinsics for 3D back-projection
            temperature: Unused (kept for interface compatibility)
            robot_state: Optional robot state dict containing camera transform for
                         converting camera-frame positions to world frame.

        Returns:
            List of DetectedObject instances added to self.registry
        """
        if isinstance(color_frame, Image.Image):
            rgb_np = np.array(color_frame)
        else:
            rgb_np = color_frame

        loop = asyncio.get_event_loop()

        # --- RAM+ tagging (CPU-bound, run in thread pool) ---
        if self._tagger is not None and self._frame_count % self.tag_interval == 0:
            new_prompt, raw = await loop.run_in_executor(
                None, self._tagger, rgb_np
            )
            if new_prompt:
                merged = self._merge_prompt(new_prompt, self._extra_tags)
                if merged != self._current_prompt:
                    self.logger.debug("RAM+ prompt update: %s (was: %s)", merged, self._current_prompt)
                    self._current_prompt = merged
                    self._gsam2.set_prompt(merged)

        # --- SAM2 tracking (GPU-bound, run in thread pool to avoid blocking event loop) ---
        def _run_gsam2():
            return self._gsam2.add_image(rgb_np)

        await loop.run_in_executor(None, _run_gsam2)

        self._frame_count += 1

        # --- Convert SAM2 results → DetectedObject ---
        detected: List[DetectedObject] = []
        mask_dict = self._gsam2.last_mask_dict
        # Clear stale predicate facts from the previous frame before rebuilding
        if self._pddl_predicates:
            self.registry.clear_predicates()

        if mask_dict is None or not mask_dict.labels:
            return detected

        h, w = rgb_np.shape[:2]

        for obj_id, obj_info in mask_dict.labels.items():
            class_name = obj_info.class_name or "object"
            # Normalise to valid PDDL identifier: spaces → underscores, lowercase.
            # Deduplicate repeated phrases — GroundingDINO echoes the prompt label
            # twice, e.g. "blue block blue block" or "red table red table".
            # Strategy: if the word list is an exact repeat of its first half,
            # keep only the first half. Also strip consecutive duplicate words.
            _words = class_name.strip().lower().split()
            _n = len(_words)
            if _n >= 2 and _n % 2 == 0 and _words[:_n // 2] == _words[_n // 2:]:
                _words = _words[:_n // 2]
            _deduped: list = []
            for _word in _words:
                if not _deduped or _word != _deduped[-1]:
                    _deduped.append(_word)
            safe_name = "_".join(_deduped)

            # Skip detections where GroundingDINO returned a non-object label
            if safe_name in _JUNK_LABELS:
                self.logger.debug("Skipping junk label '%s' (obj_id=%s)", safe_name, obj_id)
                continue

            object_id = f"{safe_name}_{obj_id}"

            # Pixel bounding box from ObjectInfo
            x1 = int(obj_info.x1) if obj_info.x1 is not None else 0
            y1 = int(obj_info.y1) if obj_info.y1 is not None else 0
            x2 = int(obj_info.x2) if obj_info.x2 is not None else w
            y2 = int(obj_info.y2) if obj_info.y2 is not None else h

            # Mask centroid for 2D position
            mask = obj_info.mask
            if mask is not None:
                import torch
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
            if mask is not None and mask.any():
                ys, xs = np.where(mask)
                centroid_y = int(ys.mean())
                centroid_x = int(xs.mean())
            else:
                centroid_y = (y1 + y2) // 2
                centroid_x = (x1 + x2) // 2

            position_2d = pixel_to_normalized((centroid_y, centroid_x), (h, w))
            # bounding_box_2d: [y1, x1, y2, x2] normalized to 0-1000 (same scale as position_2d)
            bbox_2d = [
                int(y1 / h * 1000),
                int(x1 / w * 1000),
                int(y2 / h * 1000),
                int(x2 / w * 1000),
            ]

            # 3D position: back-project to camera frame, then transform to world frame
            position_3d = None
            if depth_frame is not None and camera_intrinsics is not None:
                cam_pos = compute_3d_position(position_2d, depth_frame, camera_intrinsics)
                if cam_pos is not None:
                    position_3d = _transform_cam_to_world(cam_pos, robot_state)
                    if position_3d is None:
                        position_3d = cam_pos  # fallback: keep camera-frame if no transform

            interaction_point = InteractionPoint(
                position_2d=position_2d,
                position_3d=position_3d,
            )

            # Crop the object from the frame (used for both affordance and predicate inference)
            crop = rgb_np[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                crop = rgb_np

            # Infer affordances via LLM if available, else use cache or default
            if self._llm_client is not None:
                affordances = await self._infer_affordances(safe_name, crop)
            else:
                affordances = self._affordance_cache.get(safe_name, {"graspable"})

            obj = DetectedObject(
                object_type=safe_name,
                object_id=object_id,
                affordances=affordances,
                interaction_points={"grasp": interaction_point},
                position_2d=position_2d,
                position_3d=position_3d,
                bounding_box_2d=bbox_2d,
            )

            self.registry.add_object(obj)
            detected.append(obj)

        # Infer PDDL predicates for all detected objects now that the full
        # detected list is built (binary predicates like "on obj1 obj2" need
        # all object IDs to be known before querying).
        if self._llm_client is not None and self._pddl_predicates and detected:
            all_object_ids = [o.object_id for o in detected]
            for obj in detected:
                oy1, ox1, oy2, ox2 = (obj.bounding_box_2d or [0, 0, h, w])
                obj_crop = rgb_np[max(0, oy1 * h // 1000):min(h, oy2 * h // 1000),
                                  max(0, ox1 * w // 1000):min(w, ox2 * w // 1000)]
                if obj_crop.size == 0:
                    obj_crop = rgb_np
                pred_strings = await self._infer_predicates(
                    obj.object_id, obj.object_type, obj_crop, all_object_ids
                )
                if pred_strings:
                    self.registry.add_predicates(pred_strings)

        # Evict registry entries whose GSAM2 track ID is no longer active.
        # object_ids have the form "<name>_<track_id>"; split on the last "_"
        # to recover the numeric track_id and compare against the current frame.
        active_track_ids: Set[int] = set(mask_dict.labels.keys())
        stale_ids = []
        for reg_id in list(self.registry._objects.keys()):
            parts = reg_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                if int(parts[1]) not in active_track_ids:
                    stale_ids.append(reg_id)
        for stale_id in stale_ids:
            self.registry.remove_object(stale_id)
            self.logger.debug("Evicted stale registry entry '%s'", stale_id)

        self.logger.debug(
            "GSAM2 detected %d objects: %s",
            len(detected),
            [o.object_id for o in detected],
        )
        return detected

    async def _infer_affordances(
        self,
        object_type: str,
        crop_rgb: np.ndarray,
    ) -> Set[str]:
        """
        Query the LLM to infer affordances for an object given its cropped image.

        Results are cached by object_type so the LLM is only queried once per
        novel object class per tracker lifetime.  Falls back to {"graspable"}
        on any failure.
        """
        if object_type in self._affordance_cache:
            return self._affordance_cache[object_type]

        try:
            from ..llm_interface.base import GenerateConfig, ImagePart

            buf = io.BytesIO()
            Image.fromarray(crop_rgb).save(buf, format="PNG")
            crop_png = buf.getvalue()

            prompt = (
                f"You are a robotics perception system. Given an image of a '{object_type}', "
                "list the affordances that apply to this object for a robot manipulation task. "
                "Common affordances include: graspable, placeable_on, openable, closeable, "
                "pushable, containable, movable, fixed. "
                "Return only affordances that clearly apply based on the object's appearance. "
                "Always include 'graspable' if the object can be picked up."
            )

            config = GenerateConfig(
                temperature=0.2,
                max_output_tokens=1024,
                response_json_schema=self._AFFORDANCE_SCHEMA,
            )

            response = await self._llm_client.generate_async(
                [prompt, ImagePart(data=crop_png, mime_type="image/png")],
                config=config,
            )

            parsed = json.loads(response.text)
            affordances = set(parsed.get("affordances", ["graspable"]))
            if not affordances:
                affordances = {"graspable"}

            self._affordance_cache[object_type] = affordances
            self.logger.debug("Affordances for '%s': %s", object_type, affordances)
            return affordances

        except Exception as e:
            self.logger.warning(
                "Affordance inference failed for '%s': %s — falling back to {'graspable'}",
                object_type, e,
            )
            return {"graspable"}

    async def _infer_predicates(
        self,
        object_id: str,
        object_type: str,
        crop_rgb: np.ndarray,
        all_object_ids: List[str],
    ) -> List[str]:
        """
        Query the LLM to evaluate which PDDL predicates hold TRUE for this object.

        Uses the masked crop image and the full list of detected object IDs so
        the model can produce both unary predicates (e.g. "clear blue_block_2")
        and binary predicates (e.g. "on blue_block_2 table_5").

        Returns a list of grounded predicate strings ready for registry.add_predicates().
        Falls back to an empty list on any failure.
        """
        if not self._pddl_predicates:
            return []

        try:
            from ..llm_interface.base import GenerateConfig, ImagePart

            buf = io.BytesIO()
            Image.fromarray(crop_rgb).save(buf, format="PNG")
            crop_png = buf.getvalue()

            other_ids = [oid for oid in all_object_ids if oid != object_id]
            predicates_str = "\n".join(f"  - {p}" for p in self._pddl_predicates)
            others_str = ", ".join(other_ids) if other_ids else "(none)"

            prompt = (
                f"You are a robotics perception system evaluating PDDL state predicates.\n"
                f"The image shows object '{object_id}' (type: {object_type}).\n"
                f"Other objects in the scene: {others_str}\n\n"
                f"Available predicates:\n{predicates_str}\n\n"
                "For each predicate that holds TRUE for this object right now, "
                "output one grounded predicate string per entry:\n"
                "  - Unary:  'predicate_name object_id'\n"
                "  - Binary: 'predicate_name object_id other_object_id'\n\n"
                "Only assert predicates you can confidently determine from the image. "
                "Do not assert spatial relations you cannot see (e.g. do not guess 'on' "
                "unless the object is visibly resting on another object in this image)."
            )

            config = GenerateConfig(
                temperature=0.1,
                max_output_tokens=512,
                response_json_schema=self._PREDICATE_SCHEMA,
            )

            response = await self._llm_client.generate_async(
                [prompt, ImagePart(data=crop_png, mime_type="image/png")],
                config=config,
            )

            parsed = json.loads(response.text)
            pred_strings: List[str] = []
            for entry in parsed.get("predicates", []):
                s = str(entry).strip()
                if s:
                    pred_strings.append(s)

            self.logger.debug(
                "Predicates for '%s': %s", object_id, pred_strings
            )
            return pred_strings

        except Exception as e:
            self.logger.warning(
                "Predicate inference failed for '%s': %s — skipping",
                object_id, e,
            )
            return []

    def get_all_objects(self) -> List[DetectedObject]:
        return self.registry.get_all_objects()

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        return self.registry.get_object(object_id)


class GSAM2ContinuousObjectTracker:
    """
    Background continuous tracker using GSAM2ObjectTracker.

    Mirrors the ContinuousObjectTracker interface (start/stop/set_frame_provider/registry).
    """

    def __init__(
        self,
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path: str = "./checkpoints/sam2.1_hiera_large.pt",
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        ram_ckpt_path: Optional[str] = None,
        ram_image_size: int = 384,
        detection_interval: int = 20,
        device: str = "cuda",
        tag_interval: int = 1,
        update_interval: float = 0.0,
        on_detection_complete: Optional[Callable[[int], None]] = None,
        llm_client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        debug_save_dir: Optional[Union[str, Path]] = None,
    ):
        self._tracker = GSAM2ObjectTracker(
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            grounding_model_id=grounding_model_id,
            ram_ckpt_path=ram_ckpt_path,
            ram_image_size=ram_image_size,
            detection_interval=detection_interval,
            device=device,
            tag_interval=tag_interval,
            llm_client=llm_client,
            logger=logger,
        )
        self.registry = self._tracker.registry
        self.update_interval = update_interval
        self.on_detection_complete = on_detection_complete
        self.logger = logger or get_structured_logger("GSAM2ContinuousObjectTracker")
        self.stats = TrackingStats()

        self._frame_provider: Optional[Callable] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Last detection bundle (for save_snapshot compatibility)
        self._last_bundle_lock = threading.Lock()
        self._last_bundle: Optional[Dict[str, Any]] = None

        # Debug image streaming
        self._debug_save_dir: Optional[Path] = Path(debug_save_dir) if debug_save_dir else None
        self._debug_frame_index: int = 0
        self._debug_lock = threading.Lock()

    def set_tagger(self, tagger_callable):
        self._tracker.set_tagger(tagger_callable)

    # ------------------------------------------------------------------ #
    # Stubs for ObjectTracker interface compatibility                      #
    # ------------------------------------------------------------------ #

    def set_pddl_predicates(self, predicates) -> None:
        """Forward PDDL predicates to the inner tracker for per-object LLM evaluation."""
        self._tracker.set_pddl_predicates(predicates)

    def set_pddl_actions(self, _actions) -> None:
        """No-op: action schemas are not needed for visual predicate evaluation."""

    def set_task_context(self, task_description=None, available_actions=None, goal_objects=None) -> None:
        """Extract object noun phrases from task description and inject into GroundingDINO prompt."""
        hints: List[str] = []
        if goal_objects:
            hints.extend(goal_objects)
        if task_description:
            hints.extend(_extract_noun_phrases(task_description))
        if hints:
            self._tracker.set_extra_tags(hints)
            self.logger.info("GSAM2 task hints injected: %s", hints)

    def set_frame_provider(self, provider: Callable[[], tuple]):
        self._frame_provider = provider

    def start(self):
        if self._running:
            return
        if self._frame_provider is None:
            raise ValueError("Frame provider not set. Call set_frame_provider() first.")
        self._running = True
        self.stats.is_running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._tracking_loop())
        self.logger.info("GSAM2ContinuousObjectTracker started")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        self.stats.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("GSAM2ContinuousObjectTracker stopped")

    async def _tracking_loop(self):
        while self._running:
            loop_start = time.time()
            try:
                provided = self._frame_provider()
                if isinstance(provided, tuple) and len(provided) == 4:
                    color_frame, depth_frame, intrinsics, robot_state = provided
                else:
                    color_frame, depth_frame, intrinsics = provided
                    robot_state = None

                detection_start = time.time()
                detected = await self._tracker.detect_objects(
                    color_frame, depth_frame, intrinsics, robot_state=robot_state
                )
                detection_time = time.time() - detection_start

                self.stats.total_frames += 1
                self.stats.total_detections += len(detected)
                self.stats.last_detection_time = detection_time
                alpha = 0.1
                self.stats.avg_detection_time = (
                    alpha * detection_time + (1 - alpha) * self.stats.avg_detection_time
                )

                # Cache detection bundle for save_snapshot compatibility
                buf = io.BytesIO()
                frame_img = color_frame if isinstance(color_frame, Image.Image) else Image.fromarray(color_frame)
                frame_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                with self._last_bundle_lock:
                    self._last_bundle = {
                        "timestamp": time.time(),
                        "color_png": png_bytes,
                        "depth": np.array(depth_frame, copy=True) if depth_frame is not None else None,
                        "intrinsics": intrinsics,
                        "objects": list(detected),
                        "robot_state": robot_state,
                    }

                if self.on_detection_complete:
                    if asyncio.iscoroutinefunction(self.on_detection_complete):
                        await self.on_detection_complete(len(detected))
                    else:
                        self.on_detection_complete(len(detected))

                # Stream annotated debug frame in background if enabled
                if self._debug_save_dir is not None:
                    frame_idx = self._debug_frame_index
                    self._debug_frame_index += 1
                    objects_snapshot = list(detected)
                    t = threading.Thread(
                        target=save_debug_frame,
                        args=(png_bytes, objects_snapshot, frame_idx, self._debug_save_dir, self._debug_lock),
                        daemon=True,
                    )
                    t.start()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("GSAM2 tracking loop error: %s", e, exc_info=True)

            elapsed = time.time() - loop_start
            if self.update_interval > elapsed:
                await asyncio.sleep(self.update_interval - elapsed)

    async def get_stats(self) -> TrackingStats:
        """Return a snapshot of current tracking statistics."""
        return TrackingStats(
            total_frames=self.stats.total_frames,
            total_detections=self.stats.total_detections,
            skipped_frames=self.stats.skipped_frames,
            avg_detection_time=self.stats.avg_detection_time,
            last_detection_time=self.stats.last_detection_time,
            cache_hit_rate=self.stats.cache_hit_rate,
            is_running=self.stats.is_running,
        )

    def get_all_objects(self) -> List[DetectedObject]:
        return self.registry.get_all_objects()

    def get_last_detection_bundle(self) -> Optional[Dict[str, Any]]:
        """Return the most recent detection frame and objects for snapshotting."""
        with self._last_bundle_lock:
            return dict(self._last_bundle) if self._last_bundle is not None else None
