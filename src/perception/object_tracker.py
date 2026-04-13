"""
Async object tracking system using Gemini Robotics VLM.

This module provides async object detection, affordance analysis, and interaction
point detection for robotic manipulation tasks using Google Gen AI's native async support.
"""

import copy
import time
import asyncio
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass
import io
import json
from urllib import response
import yaml

import numpy as np
from PIL import Image

# Import the new Google GenAI SDK
from google import genai
from google.genai import types

# Import coordinate conversion utilities
from .utils.coordinates import compute_3d_position

# Import object registry
from .object_registry import DetectedObject, InteractionPoint, DetectedObjectRegistry
from ..utils.logging_utils import get_structured_logger


class ObjectTracker:
    """
    Object tracking system using Gemini Robotics VLM.

    This class detects objects in camera frames, analyzes their affordances,
    and identifies interaction points for robot manipulation. It maintains
    a registry of detected objects.

    Example:
        >>> tracker = ObjectTracker(api_key="your_key")
        >>> tracker.detect_objects(color_frame, depth_frame, camera_intrinsics)
        >>> for obj in tracker.get_all_objects():
        ...     print(f"{obj.object_id}: {obj.affordances}")
        ...     for affordance, point in obj.interaction_points.items():
        ...         print(f"  {affordance} at {point.position_3d}")
    """

    # Model configurations
    ROBOTICS_MODEL = "gemini-robotics-er-1.5-preview"
    FLASH_MODEL = "gemini-1.5-flash"
    
    # Default prompts configuration path
    DEFAULT_PROMPTS_CONFIG = str(Path(__file__).parent.parent.parent / "config" / "object_tracker_prompts.yaml")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "auto",
        default_temperature: float = 0.5,
        thinking_budget: int = -1,
        max_parallel_requests: int = 5,
        crop_target_size: int = 512,
        enable_affordance_caching: bool = True,
        fast_mode: bool = False,
        pddl_predicates: Optional[List[str]] = None,
        pddl_types: Optional[List[str]] = None,
        pddl_actions: Optional[List[str]] = None,
        prompts_config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        task_context: Optional[str] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None,
        robot: Optional[Any] = None,
        llm_client: Optional[Any] = None,  # src.llm_interface.LLMClient
    ):
        """
        Initialize object tracker.

        Args:
            api_key: Google AI API key
            model_name: "auto" to use robotics model if available
            default_temperature: Generation temperature (0.0-1.0)
            thinking_budget: Token budget for extended thinking (0=disabled)
            max_parallel_requests: Max parallel VLM requests for object details
            crop_target_size: Resize crops to this size before VLM (0=disable, default 512)
            enable_affordance_caching: Cache affordance results for similar objects
            fast_mode: Skip detailed interaction points, only detect affordances
            pddl_predicates: List of PDDL predicate names to extract from objects (e.g., ["clean", "dirty", "opened"])
            task_context: Optional task description to ground object detection (e.g., "make coffee")
            available_actions: Optional list of available PDDL actions for context
        """
        self.logger = logger or get_structured_logger("ObjectTracker")
        self.logger.setLevel(logging.INFO)
        self.api_key = api_key
        self.max_parallel_requests = max_parallel_requests
        self.crop_target_size = crop_target_size
        self.enable_affordance_caching = enable_affordance_caching
        self.fast_mode = fast_mode
        self.robot = robot

        # PDDL predicate tracking
        self.pddl_predicates: List[str] = pddl_predicates or []
        self.pddl_types: List[str] = pddl_types or []
        self.pddl_actions: List[str] = pddl_actions or []
        # Task context for grounding
        self.task_context: Optional[str] = task_context
        self.available_actions: List[Dict[str, Any]] = available_actions or []
        self.goal_objects: List[str] = []  # Expected object names/IDs from the task
        
        # Load prompts configuration
        if prompts_config_path is None:
            prompts_config_path = self.DEFAULT_PROMPTS_CONFIG
        
        with open(prompts_config_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
        
        self.logger.info("Loaded prompts from %s", prompts_config_path)

        # Auto-select best model
        if model_name == "auto":
            self.model_name = self.ROBOTICS_MODEL
            self.logger.info("ObjectTracker using: %s", self.model_name)
        else:
            self.model_name = model_name

        # Initialize LLM client (abstract or genai fallback)
        self._llm_client = llm_client
        if llm_client is None:
            self.client = genai.Client(api_key=api_key)
            self.logger.info("ObjectTracker initialized with GenAI SDK")
        else:
            self.client = None
            self.logger.info("ObjectTracker initialized with injected LLMClient: %s", type(llm_client).__name__)

        self.default_temperature = default_temperature
        self.thinking_budget = thinking_budget

        # Object registry with thread safety
        self.registry = DetectedObjectRegistry()

        # Cache for current frame
        self._current_color_frame: Optional[Image.Image] = None
        self._current_depth_frame: Optional[np.ndarray] = None
        self._current_intrinsics: Optional[Any] = None

        # Performance optimization: image encoding cache
        self._encoded_image_cache: Optional[bytes] = None
        self._cache_image_id: Optional[int] = None

        # Affordance caching: object_type -> {affordances, properties}
        self._affordance_cache: Dict[str, Dict[str, Any]] = {}

        # Recent observations for re-identification prompts
        self._recent_observations: List[Dict[str, Any]] = []
        self._max_recent_observations = 3

        # Latest detection bundle for snapshotting (protected by lock)
        self._last_detection_lock = threading.RLock()
        self._last_detection_objects: List[DetectedObject] = []
        self._last_detection_png: Optional[bytes] = None
        self._last_detection_depth: Optional[np.ndarray] = None
        self._last_detection_intrinsics: Optional[Any] = None
        self._last_detection_timestamp: Optional[float] = None
        self._last_robot_state: Any = None

    def set_pddl_predicates(self, predicates: List[str]) -> None:
        """
        Set the list of PDDL predicates to track for objects.

        Args:
            predicates: List of predicate names (e.g., ["clean", "dirty", "opened", "filled"])
        """
        self.pddl_predicates = predicates
        self.logger.info("PDDL predicates updated: %s", predicates)

    def add_pddl_predicate(self, predicate: str) -> None:
        """
        Add a single PDDL predicate to track.

        Args:
            predicate: Predicate name to add
        """
        if predicate not in self.pddl_predicates:
            self.pddl_predicates.append(predicate)
            self.logger.info("Added PDDL predicate: %s", predicate)

    def remove_pddl_predicate(self, predicate: str) -> None:
        """
        Remove a PDDL predicate from tracking.

        Args:
            predicate: Predicate name to remove
        """
        if predicate in self.pddl_predicates:
            self.pddl_predicates.remove(predicate)
            self.logger.info("Removed PDDL predicate: %s", predicate)

    def set_pddl_actions(self, actions: List[str]) -> None:
        """
        Set the list of PDDL actions to track for objects.

        Args:
            actions: List of predicate names (e.g., ["clean", "dirty", "opened", "filled"])
        """
        self.pddl_actions = actions
        self.logger.info("PDDL actions updated: %s", actions)

    def add_pddl_action(self, action: str) -> None:
        """
        Add a single PDDL action to track.

        Args:
            action: Action name to add
        """
        if action not in self.pddl_actions:
            self.pddl_actions.append(action)
            self.logger.info("Added PDDL action: %s", action)

    def remove_pddl_action(self, action: str) -> None:
        """
        Remove a PDDL action from tracking.

        Args:
            action: Action name to remove
        """
        if action in self.pddl_actions:
            self.pddl_actions.remove(action)
            self.logger.info("Removed PDDL predicate: %s", action)

    def set_pddl_types(self, types: List[str]) -> None:
        """
        Set the list of PDDL types to track for objects.

        Args:
            types: List of type names (e.g., ["clean", "dirty", "opened", "filled"])
        """
        self.pddl_types = types
        print(f"ℹ PDDL types updated: {types}")

    def add_pddl_type(self, type: str) -> None:
        """
        Add a single PDDL type to track.

        Args:
            type: type name to add
        """
        if type not in self.pddl_types:
            self.pddl_types.append(type)
            print(f"ℹ Added PDDL type: {type}")

    def remove_pddl_type(self, type: str) -> None:
        """
        Remove a PDDL type from tracking.

        Args:
            type: type name to remove
        """
        if type in self.pddl_types:
            self.pddl_types.remove(type)

    def set_task_context(
        self,
        task_description: Optional[str] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None,
        goal_objects: Optional[List[str]] = None
    ) -> None:
        """
        Set or update the task context for grounded object detection.

        This helps the VLM understand what objects are relevant for the current task
        and what actions can be performed on them.

        Args:
            task_description: Natural language task description (e.g., "make coffee")
            available_actions: List of available PDDL actions with their parameters and descriptions
            goal_objects: List of expected object IDs/names from the task (e.g., ["water_bottle_1", "cup_2"])

        Example:
            >>> tracker.set_task_context(
            ...     task_description="make coffee",
            ...     available_actions=[
            ...         {"name": "fill_water", "params": ["?machine", "?source"], "description": "Fill machine with water"},
            ...         {"name": "insert_pod", "params": ["?pod", "?machine"], "description": "Insert coffee pod"}
            ...     ],
            ...     goal_objects=["coffee_machine_1", "water_cup_1"]
            ... )
        """
        if task_description is not None:
            self.task_context = task_description
            print(f"ℹ Task context updated: \"{task_description}\"")

        if available_actions is not None:
            self.available_actions = available_actions
            print(f"ℹ Available actions updated: {len(available_actions)} actions")

        if goal_objects is not None:
            self.goal_objects = goal_objects
            print(f"ℹ Goal objects updated: {len(goal_objects)} objects - {', '.join(goal_objects)}")

    def clear_task_context(self) -> None:
        """Clear the task context and available actions."""
        self.task_context = None
        self.available_actions = []
        self.goal_objects = []
        print("ℹ Task context cleared")

    def get_pddl_predicates(self) -> List[str]:
        """Get the current list of tracked PDDL predicates."""
        return self.pddl_predicates.copy()

    def _format_task_context_for_detection(self) -> Tuple[str, str]:
        """
        Format task context for detection prompts.

        Returns:
            Tuple of (task_context_section, task_priority_note)
        """
        if not self.task_context:
            return ("", "")

        task_context_section = self.prompts['task_context']['detection_section_template'].format(
            task_description=self.task_context
        )

        # Add goal objects information if available
        if self.goal_objects:
            # Filter out None/invalid values
            valid_goal_objects = [obj for obj in self.goal_objects if obj and obj != "None"]
            if valid_goal_objects:
                goal_objects_str = ", ".join(valid_goal_objects)
                task_context_section += f"\n    Expected goal objects: {goal_objects_str}\n    IMPORTANT: When you detect objects matching these goal objects, use these EXACT IDs as labels."

        task_priority_note = self.prompts['task_context']['detection_priority_template'].format(
            task_description=self.task_context
        )

        return (task_context_section, task_priority_note)

    def _format_task_context_for_analysis(self) -> str:
        """
        Format task context for object analysis prompts.

        Returns:
            Formatted task context section
        """
        if not self.task_context and not self.available_actions:
            return ""

        # Format available actions
        actions_list = ""
        if self.available_actions:
            actions_summary = []
            for action in self.available_actions[:10]:  # Limit to 10 actions to avoid token overflow
                name = action.get('name', 'unknown')
                desc = action.get('description', '')
                if desc:
                    actions_summary.append(f"- {name}: {desc}")
                else:
                    actions_summary.append(f"- {name}")
            actions_list = "\n       ".join(actions_summary)

        task_desc = self.task_context or "General manipulation"

        task_context_str = self.prompts['task_context']['analysis_section_template'].format(
            task_description=task_desc,
            actions_list=actions_list if actions_list else "General manipulation actions"
        )

        # Add goal objects information if available
        if self.goal_objects:
            # Filter out None/invalid values
            valid_goal_objects = [obj for obj in self.goal_objects if obj and obj != "None"]
            if valid_goal_objects:
                goal_objects_str = ", ".join(valid_goal_objects)
                task_context_str += f"\n\n    Goal Objects: {goal_objects_str}\n    Note: These objects are critical for completing the task. However, you are not allowed to use these placeholder names for the predicates (you can only use objects that have been detected)."

        return task_context_str
    
    def _get_robot_state_struct(self) -> Optional[Dict[str, Any]]:
        """
        Obtain robot state via duck-typed get_robot_state() on the provider.

        The orchestrator does not assume any internal structure beyond it being JSON-serializable.
        """
        try:
            raw_state = self.robot.get_robot_state()
            # Best-effort: ensure it's serializable (convert numpy arrays)
            def to_serializable(x):
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, (list, dict, str, int, float, type(None), bool)):
                    return x
                if hasattr(x, "__dict__"):
                    # dataclass / custom object; use dict
                    return {k: to_serializable(v) for k, v in vars(x).items()}
                return str(x)
            if isinstance(raw_state, dict):
                return {k: to_serializable(v) for k, v in raw_state.items()}
            # If provider returned a non-dict, wrap it for stability
            return {"data": to_serializable(raw_state)}
        except Exception:
            return None

    async def detect_objects(
        self,
        color_frame: Union[np.ndarray, Image.Image],
        depth_frame: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[Any] = None,
        temperature: Optional[float] = None,
        robot_state: Optional[Dict[str, Any]] = None
    ) -> List[DetectedObject]:
        """
        Detect all objects in scene with affordances and interaction points (async).

        Uses Google Gen AI's native async support for true concurrent VLM requests.

        This is the main entry point. It:
        1. Detects all object names in the scene
        2. For each object concurrently using async/await:
           - Analyzes affordances
           - Detects interaction points for each affordance
        3. Updates object registry

        Args:
            color_frame: RGB image from camera
            depth_frame: Optional depth image (same size as color)
            camera_intrinsics: Optional camera intrinsics for 3D projection
            temperature: Generation temperature

        Returns:
            List of detected objects with full information
        """
        self.logger.info("Detecting objects in scene (async)...")

        # Clear previous predicates before new detection
        self.registry.clear_predicates()

        start_time = time.time()

        # Prepare image
        pil_image = self._prepare_image(color_frame)

        # Capture frame timestamp and robot state as close to frame prep as possible
        # Prefer provider-supplied state when available to avoid race conditions.
        frame_timestamp = time.time()
        frame_robot_state = robot_state if robot_state is not None else self._get_robot_state_struct()

        # Store current frame for analysis
        self._current_color_frame = pil_image
        self._current_depth_frame = depth_frame
        self._current_intrinsics = camera_intrinsics

        # Performance optimization: Encode image once for all requests
        encode_start = time.time()
        self._cache_image_id = id(pil_image)
        # Pre-encode image (reused in parallel requests)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        self._encoded_image_cache = img_byte_arr.getvalue()
        encode_time = time.time() - encode_start

        # Build detection prompts with existing objects and prior observations for re-identification
        existing_objects = self.registry.get_all_objects()
        prior_observations = self._select_latest_reid_observations(
            existing_objects,
            list(self._recent_observations)
        )
        additional_image_parts = [
            types.Part.from_bytes(data=obs["image_bytes"], mime_type="image/png")
            for obs in prior_observations
            if obs.get("image_bytes")
        ]
        prior_prompt, current_prompt = self._format_detection_prompt_sections(
            existing_objects,
            prior_observations=prior_observations
        )
        prior_parts = list(additional_image_parts)
        prior_parts.append(types.Part.from_text(text=prior_prompt))
        current_image_part = types.Part.from_bytes(
            data=self._encoded_image_cache,
            mime_type="image/png"
        )
        current_parts = [
            current_image_part,
            types.Part.from_text(text=current_prompt)
        ]
        content_sequence = [
            types.Content(role="user", parts=prior_parts),
            types.Content(role="user", parts=current_parts),
        ]
        existing_ids = {obj.object_id for obj in existing_objects}

        # Step 1: Stream object names and collect them
        names_start = time.time()
        self.logger.debug("Detecting objects (async streaming)...")

        object_data_list = []

        async def on_object_detected(object_name: str, bounding_box: Optional[List[int]]):
            """Callback when object detected"""
            bbox_str = f"bbox={bounding_box}" if bounding_box else "no bbox"
            self.logger.debug("Found object: %s (%s)", object_name, bbox_str)
            object_data_list.append((object_name, bounding_box))

        # Use async streaming detection
        await self._detect_object_names_streaming(
            pil_image,
            temperature,
            on_object_detected,
            prompt=current_prompt,
            additional_image_parts=None,
            content_parts_override=None,
            contents_override=content_sequence
        )

        if not object_data_list:
            self.logger.warning("No objects detected")
            self._encoded_image_cache = None
            return []

        names_time = time.time() - names_start
        self.logger.info(
            "Detection phase complete in %.1fs (%s objects)",
            names_time,
            len(object_data_list),
        )
        self.logger.debug("Analyzing objects concurrently (async)...")

        # Step 2: Analyze all objects concurrently using asyncio.gather
        parallel_start = time.time()

        tasks = [
            self._analyze_single_object(
                object_name,
                pil_image,
                depth_frame,
                camera_intrinsics,
                temperature,
                bounding_box,
                existing_object_id=object_name if object_name in existing_ids else None,
                other_objects=object_data_list
            )
            for object_name, bounding_box in object_data_list
        ]

        # Run all analysis tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        detected_objects = []
        for i, result in enumerate(results):
            object_name = object_data_list[i][0]
            if isinstance(result, Exception):
                self.logger.warning("Failed to analyze %s: %s", object_name, result)
            elif result is not None:
                detected_objects.append(result)
                self.registry.add_object(result)
                self.logger.debug(
                    "Analyzed %s: %s affordances, %s points",
                    object_name,
                    len(result.affordances),
                    len(result.interaction_points),
                )

        parallel_time = time.time() - parallel_start
        total_time = time.time() - start_time

        # Cache the latest detection bundle for snapshot alignment
        with self._last_detection_lock:
            # Use the robot state and timestamp captured at frame acquisition
            self._last_robot_state = copy.deepcopy(frame_robot_state)
            self._last_detection_timestamp = frame_timestamp
            self._last_detection_objects = [copy.deepcopy(obj) for obj in detected_objects]
            if self._encoded_image_cache:
                self._last_detection_png = bytes(self._encoded_image_cache)
            else:
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                self._last_detection_png = buf.getvalue()
            self._last_detection_depth = np.array(depth_frame, copy=True) if depth_frame is not None else None
            self._last_detection_intrinsics = self._coerce_intrinsics_snapshot(camera_intrinsics)

        # Record observation for re-identification prompts
        self._store_recent_observation(self._encoded_image_cache, detected_objects)

        # Clear encoding cache
        self._encoded_image_cache = None
        self._cache_image_id = None

        # Performance summary
        self.logger.info("Detection complete in %.1fs", total_time)
        self.logger.debug(
            "Timing breakdown: encoding=%dms, names=%.1fs, analysis=%.1fs (%s objects)",
            int(encode_time * 1000),
            names_time,
            parallel_time,
            len(detected_objects),
        )
        if len(detected_objects) > 0:
            avg_time = parallel_time / len(detected_objects)
            self.logger.debug("Average per object: %.1fs", avg_time)

        return detected_objects

    async def _detect_object_names_streaming(
        self,
        image: Image.Image,
        temperature: Optional[float],
        callback,
        prompt: Optional[str] = None,
        additional_image_parts: Optional[List[types.Part]] = None,
        content_parts_override: Optional[List[Any]] = None,
        contents_override: Optional[List[Any]] = None
    ):
        """
        Detect names and bounding boxes with async streaming.

        Args:
            image: PIL Image
            temperature: Generation temperature
            callback: Async function called with (object_name, bounding_box)
        """
        if prompt is None:
            _, prompt = self._format_detection_prompt_sections(
                self.registry.get_all_objects()
            )

        # Add task context to the prompt
        task_context_section, task_priority_note = self._format_task_context_for_detection()
        if task_context_section:
            # Insert task context after the "Previously detected objects" section
            prompt = prompt.replace(
                "Instructions:",
                f"{task_context_section}\n\n      Instructions:"
            )
        if task_priority_note:
            # Add priority note to the instructions
            prompt = prompt.replace(
                "- Return bounding boxes",
                f"{task_priority_note}\n      - Return bounding boxes"
            )

        try:
            response_text = await self._generate_content(
                image,
                prompt,
                temperature,
                additional_image_parts=additional_image_parts,
                content_parts=content_parts_override,
                contents=contents_override
            )

            detections = self._parse_detection_response(response_text)
            for object_name, bounding_box in detections[:25]:
                await callback(object_name, bounding_box)

        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error("OBJECT DETECTION FAILED")
            self.logger.error("=" * 60)
            self.logger.error("Error type: %s", type(e).__name__)
            self.logger.error("Error message: %s", str(e))
            self.logger.exception("Full traceback:")
            self.logger.error("=" * 60)
            raise

    async def _analyze_single_object(
        self,
        object_name: str,
        image: Image.Image,
        depth_frame: Optional[np.ndarray],
        camera_intrinsics: Optional[Any],
        temperature: Optional[float],
        bounding_box: Optional[List[int]] = None,
        existing_object_id: Optional[str] = None,
        other_objects: Optional[List[Tuple[str, Optional[List[int]]]]] = None
    ) -> Optional[DetectedObject]:
        """
        Analyze a single object: affordances, properties, and interaction points.

        If bounding_box is provided, crops the image to that region and back-projects
        interaction points to full image coordinates.

        Args:
            object_name: Name of object to analyze
            image: RGB image (full size)
            depth_frame: Optional depth image (full size)
            camera_intrinsics: Optional camera intrinsics
            temperature: Generation temperature
            bounding_box: Optional bounding box [y1, x1, y2, x2] in 0-1000 normalized coords

        Returns:
            DetectedObject with full information
        """
        # Crop image if bounding box provided
        analysis_image = image
        crop_offset = None

        if bounding_box is not None:
            # Convert normalized bbox to pixel coordinates
            from .utils.coordinates import normalized_to_pixel
            img_height, img_width = image.height, image.width

            y1, x1, y2, x2 = bounding_box
            # Convert each corner to pixels
            py1, px1 = normalized_to_pixel([y1, x1], (img_height, img_width))
            py2, px2 = normalized_to_pixel([y2, x2], (img_height, img_width))

            # Ensure valid crop bounds
            px1, px2 = max(0, min(px1, px2)), min(img_width, max(px1, px2))
            py1, py2 = max(0, min(py1, py2)), min(img_height, max(py1, py2))

            # Crop the image (PIL uses left, upper, right, lower)
            analysis_image = image.crop((px1, py1, px2, py2))
            crop_offset = (py1, px1, py2, px2)  # Store for back-projection

            # Resize crop to target size for faster VLM processing
            if self.crop_target_size > 0:
                analysis_image = analysis_image.resize(
                    (self.crop_target_size, self.crop_target_size),
                    Image.Resampling.LANCZOS
                )

        # Check affordance cache for this object type
        object_type_guess = object_name.split()[-1] if ' ' in object_name else object_name
        cached_data = None
        if self.enable_affordance_caching and object_type_guess in self._affordance_cache:
            cached_data = self._affordance_cache[object_type_guess]

        # Build prompt based on mode and cache
        crop_note = " (This is a cropped region focusing on the object.)" if crop_offset else ""

        # Build predicates section if predicates are specified
        predicates_section = ""
        predicates_example = ""
        if self.pddl_predicates:
            pddl_list = ", ".join(self.pddl_predicates)

            # Get list of other detected objects for relational predicates
            other_objects_list = ""
            if other_objects:
                # other_objects is a list of tuples: (name, bbox)
                names = [name for name, _ in other_objects
                         if name != object_name and (existing_object_id is None or name != existing_object_id)]
                if names:
                    other_objects_list = "\n" + "".join(f"       - {name}\n" for name in names)
                else:
                    other_objects_list = "\n       (No other objects detected yet)"
            else:
                other_objects_list = "\n       (No other objects detected yet)"

            predicates_section = self.prompts['predicates']['section_template'].format(
                object_id=object_name,
                pddl_list=pddl_list,
                other_objects_list=other_objects_list
            )
            predicates_example = self.prompts['predicates']['example_template'].format(
                object_id=object_name
            )

        # Format task context for analysis
        task_context_section = self._format_task_context_for_analysis()
        analysis_schema = json.dumps(
            self.prompts["analysis"].get("response_schema", {}),
            ensure_ascii=True,
            indent=2
        )

        if self.fast_mode:
            # Fast mode: only detect affordances and properties, no interaction points
            prompt = self.prompts['analysis']['fast_mode'].format(
                object_name=object_name,
                crop_note=crop_note,
                predicates_section=predicates_section,
                predicates_example=predicates_example,
                task_context_section=task_context_section,
                analysis_schema=analysis_schema
            )
        elif cached_data:
            # Use cached affordances, only detect interaction points
            prompt = self.prompts['analysis']['cached_mode'].format(
                object_name=object_name,
                crop_note=crop_note,
                cached_affordances=cached_data['affordances'],
                object_type=cached_data['object_type'],
                analysis_schema=analysis_schema
            )
        else:
            # Full analysis
            prompt = self.prompts['analysis']['full'].format(
                object_name=object_name,
                crop_note=crop_note,
                predicates_section=predicates_section,
                predicates_example=predicates_example,
                task_context_section=task_context_section,
                analysis_schema=analysis_schema
            )

        try:
            response_text = await self._generate_content(
                analysis_image,
                prompt,
                temperature,
                response_schema=self.prompts["analysis"].get("response_schema")
            )
            data = self._parse_json_response(response_text)

            if not data:
                return None

            # Handle case where VLM returns a list instead of dict
            if isinstance(data, list):
                self.logger.warning("Unexpected list response for %s, skipping", object_name)
                return None

            raw_affordances = data.get("affordances", []) or []

            # Extract affordances (use cached if available)
            if cached_data:
                affordances = set(cached_data["affordances"])
            else:
                affordances: set[str] = set()

            # Helper function to back-project coordinates from crop to full image
            def backproject_to_full_image(crop_pos_2d: List[int]) -> List[int]:
                """Convert position from cropped image coords to full image coords."""
                if crop_offset is None:
                    return crop_pos_2d  # No cropping, return as-is

                from .utils.coordinates import normalized_to_pixel, pixel_to_normalized
                crop_y, crop_x = crop_pos_2d
                crop_height = crop_offset[2] - crop_offset[0]
                crop_width = crop_offset[3] - crop_offset[1]

                # Convert normalized coords to pixels in crop
                pixel_y, pixel_x = normalized_to_pixel([crop_y, crop_x], (crop_height, crop_width))

                # Add crop offset to get full image pixels
                full_pixel_y = pixel_y + crop_offset[0]
                full_pixel_x = pixel_x + crop_offset[1]

                # Convert back to normalized coords in full image
                img_height, img_width = image.height, image.width
                full_norm_y, full_norm_x = pixel_to_normalized(
                    [full_pixel_y, full_pixel_x],
                    (img_height, img_width)
                )

                return [full_norm_y, full_norm_x]

            # Extract affordances and interaction points from affordance entries
            interaction_points_result: Dict[str, InteractionPoint] = {}
            for entry in raw_affordances:
                if isinstance(entry, str):
                    affordances.add(entry)
                    continue
                if not isinstance(entry, dict):
                    continue
                affordance_name = entry.get("affordance") or entry.get("name")
                if isinstance(affordance_name, str):
                    affordances.add(affordance_name)
                else:
                    continue
                point_data = entry.get("interaction_point") if isinstance(entry.get("interaction_point"), dict) else entry
                if not isinstance(point_data, dict):
                    continue
                pos_value = point_data.get("position")
                if not isinstance(pos_value, (list, tuple)) or len(pos_value) < 2:
                    continue
                pos_2d_crop = list(pos_value[:2])
                # Back-project to full image coordinates
                pos_2d = backproject_to_full_image(pos_2d_crop)

                # Compute 3D position if depth available. Prefer explicit depth_frame
                # passed to the worker; otherwise fall back to the cached frame on
                # the tracker instance. We do NOT include depth in prompts to the LLM.
                pos_3d = None
                use_depth = depth_frame if depth_frame is not None else self._current_depth_frame
                use_intrinsics = camera_intrinsics if camera_intrinsics is not None else self._current_intrinsics
                if use_depth is not None and use_intrinsics is not None:
                    pos_3d = compute_3d_position(
                        pos_2d,
                        use_depth,
                        use_intrinsics
                    )

                interaction_points_result[affordance_name] = InteractionPoint(
                    position_2d=pos_2d,
                    position_3d=pos_3d,
                    alternative_points=[]
                )

            # Extract center position
            pos_2d_crop = data.get("position", [500, 500])
            pos_2d = backproject_to_full_image(pos_2d_crop)
            pos_3d = None
            use_depth = depth_frame if depth_frame is not None else self._current_depth_frame
            use_intrinsics = camera_intrinsics if camera_intrinsics is not None else self._current_intrinsics
            if use_depth is not None and use_intrinsics is not None:
                pos_3d = compute_3d_position(pos_2d, use_depth, use_intrinsics)

            # Create object ID
            object_type = data.get("object_type", object_name.split()[0])
            object_id = self._generate_object_id(object_name, object_type)

            # Use bounding box from initial detection, not from detailed analysis
            # (since detailed analysis was done on cropped image)
            final_bbox = bounding_box if bounding_box is not None else data.get("bounding_box")

            # Update affordance cache if enabled and not already cached
            if self.enable_affordance_caching and not cached_data and affordances:
                self._affordance_cache[object_type] = {
                    "object_type": object_type,
                    "affordances": list(affordances)
                }

            # Extract predicates if present
            object_predicates = data.get("predicates", [])

            # Validate and normalize predicates
            validated_predicates = []
            for pred in object_predicates:
                if isinstance(pred, str) and pred.strip():
                    # Normalize whitespace
                    normalized = " ".join(pred.strip().split())
                    validated_predicates.append(normalized)

            # Add predicates to registry's global predicate set
            if validated_predicates:
                self.registry.add_predicates(validated_predicates)
            print(f"ℹ Object '{object_name}' predicates: {validated_predicates}, original predicates: {object_predicates}")
            # Create DetectedObject (without predicates field - they're only at registry level)
            detected_obj = DetectedObject(
                object_type=object_type,
                object_id=existing_object_id or object_id,
                affordances=affordances,
                interaction_points=interaction_points_result,
                position_2d=pos_2d,
                position_3d=pos_3d,
                bounding_box_2d=final_bbox
            )

            return detected_obj

        except Exception as e:
            self.logger.error("Failed to analyze object '%s'", object_name)
            self.logger.error("  Error type: %s", type(e).__name__)
            self.logger.error("  Error message: %s", str(e))
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.exception("  Full traceback:")
            return None

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """
        Get object by ID from registry (thread-safe).

        Returns:
            DetectedObject or None if not found
        """
        return self.registry.get_object(object_id)

    def get_all_objects(self) -> List[DetectedObject]:
        """
        Get all objects from registry (thread-safe).

        Returns:
            List of all detected objects (snapshot)
        """
        return self.registry.get_all_objects()

    def get_objects_by_type(self, object_type: str) -> List[DetectedObject]:
        """
        Get all objects of a specific type (thread-safe).

        Args:
            object_type: Type to filter by (e.g., "cup", "bottle")

        Returns:
            List of objects matching the type
        """
        return self.registry.get_objects_by_type(object_type)

    def get_objects_with_affordance(self, affordance: str) -> List[DetectedObject]:
        """
        Get all objects that have a specific affordance (thread-safe).

        Args:
            affordance: Affordance to filter by (e.g., "graspable", "pourable")

        Returns:
            List of objects with the affordance
        """
        return self.registry.get_objects_with_affordance(affordance)

    def get_last_detection_bundle(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent detection frame and objects for snapshotting.

        Returns:
            Dict with PNG bytes, depth, intrinsics, objects, and timestamp; None if unavailable.
        """
        with self._last_detection_lock:
            if self._last_detection_png is None:
                return None

            depth_copy = (
                np.array(self._last_detection_depth, copy=True)
                if self._last_detection_depth is not None else None
            )
            intrinsics_copy = copy.deepcopy(self._last_detection_intrinsics)
            objects_copy = [
                copy.deepcopy(obj) for obj in (self._last_detection_objects or [])
            ]

            return {
                "timestamp": self._last_detection_timestamp,
                "color_png": self._last_detection_png,
                "depth": depth_copy,
                "intrinsics": intrinsics_copy,
                "objects": objects_copy,
                "robot_state": copy.deepcopy(self._last_robot_state)
            }

    def clear_registry(self):
        """Clear object registry (thread-safe)."""
        self.registry.clear()

    def save_detections(self, output_path: str, include_timestamp: bool = True):
        """
        Save detected objects to JSON file (thread-safe).

        Args:
            output_path: Path to save JSON file
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Path to saved file
        """
        return self.registry.save_to_json(output_path, include_timestamp)

    def load_detections(self, input_path: str):
        """
        Load detected objects from JSON file (thread-safe).

        Args:
            input_path: Path to JSON file

        Returns:
            List of loaded DetectedObject instances
        """
        return self.registry.load_from_json(input_path)

    async def update_interaction_point(
        self,
        object_id: str,
        affordance: str,
        task_context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[InteractionPoint]:
        """
        Update interaction point for a specific object and affordance (async).

        Useful for refining interaction points based on task context.

        Args:
            object_id: Object to update
            affordance: Which affordance to update
            task_context: Optional task description for context
            temperature: Generation temperature

        Returns:
            Updated InteractionPoint or None if failed
        """
        obj = self.get_object(object_id)
        if not obj:
            self.logger.warning("Object %s not found in registry", object_id)
            return None

        if self._current_color_frame is None:
            self.logger.warning("No current frame cached")
            return None

        # Build prompt for specific interaction point
        task_context_section = f"\nTask context: {task_context}" if task_context else ""
        
        prompt = self.prompts['interaction']['update'].format(
            affordance=affordance,
            object_id=obj.object_id,
            object_type=obj.object_type,
            task_context_section=task_context_section
        )

        try:
            response_text = await self._generate_content(
                self._current_color_frame,
                prompt,
                temperature
            )
            data = self._parse_json_response(response_text)

            pos_2d = data.get("position", [500, 500])

            # Compute 3D if depth available
            pos_3d = None
            if self._current_depth_frame is not None and self._current_intrinsics is not None:
                pos_3d = compute_3d_position(
                    pos_2d,
                    self._current_depth_frame,
                    self._current_intrinsics
                )

            # Create updated interaction point
            interaction_point = InteractionPoint(
                position_2d=pos_2d,
                position_3d=pos_3d
            )

            # Update in registry (thread-safe)
            obj = self.registry.get_object(object_id)
            if obj:
                obj.interaction_points[affordance] = interaction_point
                self.registry.update_object(object_id, obj)

            return interaction_point

        except Exception as e:
            self.logger.exception("Failed to update interaction point: %s", e)
            return None

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _generate_object_id(self, object_name: str, object_type: str) -> str:
        """Generate unique object ID (thread-safe)."""
        return self.registry.generate_unique_id(object_name, object_type)

    def _coerce_intrinsics_snapshot(self, intrinsics: Any) -> Optional[Dict[str, Any]]:
        """
        Build a JSON-serializable intrinsics payload for snapshot persistence.

        Returns a dict (or empty dict) when conversion fails, None when no intrinsics provided.
        """
        if intrinsics is None:
            return None
        try:
            return copy.deepcopy(intrinsics.to_dict())
        except Exception:
            pass
        if isinstance(intrinsics, dict):
            return copy.deepcopy(intrinsics)
        try:
            return dict(intrinsics)
        except Exception:
            return {}

    def _prepare_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Convert image to PIL Image format."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _build_detection_context_sections(
        self,
        existing_objects: List[DetectedObject],
        prior_observations: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, str]:
        """Create text sections for existing IDs and prior observations."""
        if existing_objects:
            lines = []
            for obj in existing_objects[:20]:
                parts = [f"id={obj.object_id}"]
                if obj.object_type:
                    parts.append(f"type={obj.object_type}")
                if obj.bounding_box_2d:
                    parts.append(f"bbox={obj.bounding_box_2d}")
                lines.append("- " + ", ".join(parts))

            if len(existing_objects) > 20:
                lines.append(f"- ... {len(existing_objects) - 20} more previously seen objects not listed")
            existing_section = "\n".join(lines)
        else:
            existing_section = "None observed yet."

        observations = prior_observations or []
        if observations:
            sections = []
            for idx, obs in enumerate(observations):
                image_label = f"prior_obs_{idx + 1}"
                sections.append(f"- image id: {image_label}")
                sections.append("  objects within:")
                if not obs.get("objects"):
                    sections.append("    • (none recorded)")
                else:
                    for obj in obs["objects"]:
                        pos = obj.get("position")
                        sections.append(
                            f"    • {obj.get('label', obj.get('id'))} (id={obj.get('id')}, position={pos})"
                        )
            prior_section = "\n".join(sections)
        else:
            prior_section = "None available."

        return existing_section, prior_section

    def _select_latest_reid_observations(
        self,
        existing_objects: List[DetectedObject],
        prior_observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Choose the minimal set of prior observations that cover the latest sighting of
        each known object. Newest frames are considered first so we keep only the last
        frame that contains each object ID.
        """
        if not prior_observations:
            return []

        object_ids = {obj.object_id for obj in existing_objects}
        # If registry is empty, fall back to all IDs present in observations
        if not object_ids:
            for obs in prior_observations:
                for obj in obs.get("objects", []):
                    oid = obj.get("id")
                    if oid:
                        object_ids.add(oid)

        selected_indices: set[int] = set()
        seen_ids: set[str] = set()

        # Walk from newest to oldest, selecting the first frame that contains each ID.
        for idx in range(len(prior_observations) - 1, -1, -1):
            obs = prior_observations[idx]
            includes_new_id = False
            for obj in obs.get("objects", []):
                oid = obj.get("id")
                if not oid:
                    continue
                if object_ids and oid not in object_ids:
                    continue
                if oid not in seen_ids:
                    seen_ids.add(oid)
                    includes_new_id = True
            if includes_new_id:
                selected_indices.add(idx)
            if object_ids and seen_ids.issuperset(object_ids):
                break

        if not selected_indices:
            return []

        return [prior_observations[i] for i in sorted(selected_indices)]

    def _format_detection_prompt_sections(
        self,
        existing_objects: List[DetectedObject],
        prior_observations: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, str]:
        """
        Build a two-turn detection prompt to separate reference media from the current frame.
        Turn 1: Prior IDs/images for re-ID (no detection yet).
        Turn 2: Current frame + detection instructions.
        """
        existing_section, prior_section = self._build_detection_context_sections(
            existing_objects, prior_observations
        )

        # Get task context sections for detection
        task_context_section, task_priority_section = self._format_task_context_for_detection()

        streaming = self.prompts['detection']['streaming']
        if isinstance(streaming, dict):
            prior_template = streaming.get('prior') or ""
            current_template = streaming.get('current') or ""
        else:
            prior_template = ""
            current_template = streaming

        prior_prompt = prior_template.format(
            existing_objects_section=existing_section,
            prior_images_section=prior_section,
            task_context_section=task_context_section
        ).strip()
        current_prompt = current_template.format(
            existing_objects_section=existing_section,
            prior_images_section=prior_section,
            task_priority_section=task_priority_section
        ).strip()

        # Ensure turn semantics are explicit even if templates are missing.
        if not prior_prompt:
            prior_prompt = "PRIOR CONTEXT ONLY: reuse IDs if visible. Await current frame for detection."
        if not current_prompt:
            current_prompt = (
                "CURRENT FRAME: Perform detection ONLY on this image, reusing IDs from prior context."
            )

        return prior_prompt, current_prompt

    def _parse_detection_response(
        self,
        response_text: str
    ) -> List[Tuple[str, List[int]]]:
        """
        Parse detection output. Preferred format: JSON array of objects with
        {"box_2d": [ymin, xmin, ymax, xmax], "label": "<id or name>"}.
        Falls back to legacy line-based parsing if JSON parsing fails.
        """
        detections: List[Tuple[str, List[int]]] = []

        def _coerce_bbox(bbox: Any) -> Optional[List[int]]:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return None
            try:
                coords = [int(round(float(v))) for v in bbox]
                # Clamp to 0-1000 normalized range
                return [max(0, min(1000, c)) for c in coords]
            except (TypeError, ValueError):
                return None

        parsed = self._parse_json_response(response_text)
        candidates: List[Any] = []
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            for key in ("objects", "detections", "items", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    candidates = parsed[key]
                    break

        for item in candidates:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("name") or item.get("id")
            bbox = item.get("box_2d") or item.get("bounding_box") or item.get("bbox")
            coerced = _coerce_bbox(bbox)
            if label and coerced:
                # Sanitize label: replace hyphens with underscores for PDDL compatibility
                sanitized_label = str(label).replace('-', '_')
                detections.append((sanitized_label, coerced))

        if detections:
            return detections

        # Fallback: legacy line-based parse (OBJECT: name | [bbox])
        lines = response_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line.startswith('OBJECT:'):
                continue
            content = line[7:].strip()
            if '|' in content:
                parts = content.split('|')
                object_name = parts[0].strip()
                # Sanitize name: replace hyphens with underscores for PDDL compatibility
                object_name = object_name.replace('-', '_')
                try:
                    bbox_str = parts[1].strip()
                    bounding_box = json.loads(bbox_str)
                    coerced = _coerce_bbox(bounding_box)
                    if object_name and coerced:
                        detections.append((object_name, coerced))
                except Exception:
                    if object_name:
                        detections.append((object_name, None))
            else:
                if content:
                    # Sanitize name: replace hyphens with underscores for PDDL compatibility
                    sanitized_content = content.replace('-', '_')
                    detections.append((sanitized_content, None))

        return detections

    def _store_recent_observation(
        self,
        image_bytes: Optional[bytes],
        detected_objects: List[DetectedObject]
    ) -> None:
        """Cache recent observation frames with object positions for re-ID prompts."""
        if not image_bytes or not detected_objects:
            return

        image_hash = hashlib.sha1(image_bytes).hexdigest()
        objects_summary = []
        for obj in detected_objects:
            pos = obj.bounding_box_2d or obj.position_2d
            objects_summary.append({
                "id": obj.object_id,
                "label": obj.object_type or obj.object_id,
                "position": pos
            })

        # Remove any existing entry with same hash
        self._recent_observations = [
            obs for obs in self._recent_observations
            if obs.get("image_hash") != image_hash
        ]

        self._recent_observations.append({
            "image_hash": image_hash,
            "image_bytes": image_bytes,
            "objects": objects_summary
        })

        # Keep only the most recent N observations
        if len(self._recent_observations) > self._max_recent_observations:
            self._recent_observations = self._recent_observations[-self._max_recent_observations:]

    async def _generate_content(
        self,
        image: Image.Image,
        prompt: str,
        temperature: Optional[float],
        response_schema: Optional[Dict[str, Any]] = None,
        additional_image_parts: Optional[List[types.Part]] = None,
        content_parts: Optional[List[Any]] = None,
        contents: Optional[List[Any]] = None
    ) -> str:
        """
        Generate content using async streaming API and collect all chunks.
        This is a convenience wrapper around _generate_content_streaming.
        """
        chunks = []
        async for chunk in self._generate_content_streaming(
            image,
            prompt,
            temperature,
            response_schema=response_schema,
            additional_image_parts=additional_image_parts,
            content_parts=content_parts,
            contents=contents
        ):
            chunks.append(chunk)
        return ''.join(chunks)

    async def _generate_content_streaming(
        self,
        image: Image.Image,
        prompt: str,
        temperature: Optional[float],
        response_schema: Optional[Dict[str, Any]] = None,
        additional_image_parts: Optional[List[types.Part]] = None,
        content_parts: Optional[List[Any]] = None,
        contents: Optional[List[Any]] = None
    ):
        """
        Generate content with async streaming support using client.aio.
        Yields text chunks as they arrive.
        """
        # Encode image (reuse cache when possible)
        if (self._encoded_image_cache is not None and
            self._cache_image_id == id(image)):
            img_bytes = self._encoded_image_cache
        else:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

        temp = temperature if temperature is not None else self.default_temperature

        # --- Abstract LLM client path ---
        if self._llm_client is not None:
            from src.llm_interface.base import GenerateConfig, ImagePart
            llm_parts = [ImagePart(data=img_bytes)]
            if additional_image_parts:
                # additional_image_parts are genai types — encode to bytes if needed
                for ap in additional_image_parts:
                    raw = getattr(ap, "inline_data", None)
                    if raw is not None:
                        llm_parts.append(ImagePart(data=raw.data, mime_type=raw.mime_type))
            llm_parts.append(prompt)
            cfg = GenerateConfig(
                temperature=temp,
                response_mime_type="application/json" if response_schema else None,
                response_json_schema=response_schema,
                thinking_budget=self.thinking_budget,
            )
            async for chunk in self._llm_client.generate_stream(llm_parts, config=cfg):
                yield chunk
            return

        # --- genai client path (original) ---
        if contents is None and content_parts is None:
            content_parts = [types.Part.from_bytes(data=img_bytes, mime_type='image/png')]
            if additional_image_parts:
                content_parts.extend(additional_image_parts)
            content_parts.append(prompt)
        payload = contents if contents is not None else content_parts

        config_kwargs: Dict[str, Any] = {
            "temperature": temp,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        }
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=payload,
            config=config
        )
        async for chunk in stream:
            if hasattr(chunk, 'text') and chunk.text:
                yield chunk.text

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, tolerating thinking tokens and markdown fences."""
        import json
        import re

        text = response_text

        # Strip <think>...</think> blocks (Qwen3 thinking models)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # Try bare parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from ```json ... ``` or ``` ... ``` fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Find first { } or [ ] block
        for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        self.logger.warning("Failed to parse JSON from response: %s", text[:200])
        return {}

    async def aclose(self):
        """Close async client resources."""
        if self.client:
            await self.client.aio.aclose()


@dataclass
class TrackingStats:
    """Statistics for the continuous tracker."""
    total_frames: int = 0
    total_detections: int = 0
    skipped_frames: int = 0
    avg_detection_time: float = 0.0
    last_detection_time: float = 0.0
    cache_hit_rate: float = 0.0
    is_running: bool = False


_DEBUG_PALETTE = [
    (255,  80,  80, 200),
    ( 80, 120, 255, 200),
    ( 80, 220,  80, 200),
    (255, 200,  50, 200),
    (220,  80, 220, 200),
    ( 80, 220, 220, 200),
    (255, 160,  50, 200),
]


def save_debug_frame(
    png_bytes: bytes,
    objects: List[Any],
    frame_index: int,
    save_dir: Path,
    lock: threading.Lock,
) -> None:
    """Render bounding boxes + labels onto a frame PNG and write to save_dir.

    Intended to be called from a daemon thread. Updates a ``latest.png`` symlink
    atomically so external viewers always see the most recent frame.
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for i, obj in enumerate(objects):
            color = _DEBUG_PALETTE[i % len(_DEBUG_PALETTE)]
            oid = obj.object_id
            bbox = obj.bounding_box_2d
            pos2d = obj.position_2d

            if bbox and len(bbox) == 4:
                y1n, x1n, y2n, x2n = bbox
                x1 = int(x1n / 1000 * w)
                y1 = int(y1n / 1000 * h)
                x2 = int(x2n / 1000 * w)
                y2 = int(y2n / 1000 * h)
                draw.rectangle([x1, y1, x2, y2], fill=(*color[:3], 50), outline=color, width=2)
                label_y = max(0, y1 - 18)
                draw.rectangle([x1, label_y, x1 + len(oid) * 8 + 6, label_y + 16],
                               fill=(*color[:3], 200))
                draw.text((x1 + 3, label_y + 1), oid, fill=(255, 255, 255, 255), font=font)
            elif pos2d and len(pos2d) >= 2:
                cx = int(pos2d[1] / 1000 * w)
                cy = int(pos2d[0] / 1000 * h)
                r = 6
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(*color[:3], 200), outline=(255, 255, 255, 255), width=2)
                draw.text((cx + r + 2, cy - 8), oid, fill=(*color[:3], 255), font=font)

            for ip_name, ip in (obj.interaction_points or {}).items():
                pos = ip.position_2d if hasattr(ip, "position_2d") else None
                if pos and len(pos) >= 2:
                    cx = int(pos[1] / 1000 * w)
                    cy = int(pos[0] / 1000 * h)
                    arm = 4
                    draw.line([cx - arm, cy, cx + arm, cy], fill=(255, 255, 255, 230), width=2)
                    draw.line([cx, cy - arm, cx, cy + arm], fill=(255, 255, 255, 230), width=2)
                    draw.text((cx + 5, cy - 6), ip_name, fill=(220, 220, 220, 200), font=font)

        stamp = f"frame {frame_index:04d}  |  objects: {len(objects)}"
        draw.rectangle([0, h - 20, w, h], fill=(0, 0, 0, 160))
        draw.text((4, h - 17), stamp, fill=(200, 200, 200, 255), font=font)

        out_path = save_dir / f"frame_{frame_index:04d}.png"
        img.save(out_path)
        latest = save_dir / "latest.png"
        with lock:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(out_path.name)
    except Exception:
        logging.getLogger(__name__).debug("Debug frame save failed", exc_info=True)


class ContinuousObjectTracker(ObjectTracker):
    """
    Background service for continuous object tracking.

    Runs in the background to periodically call `detect_objects` and keep the
    registry fresh.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "auto",
        max_parallel_requests: int = 5,
        crop_target_size: int = 512,
        enable_affordance_caching: bool = True,
        fast_mode: bool = False,
        update_interval: float = 1.0,
        on_detection_complete: Optional[Callable[[int], None]] = None,
        logger: Optional[logging.Logger] = None,
        robot: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        debug_save_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            max_parallel_requests=max_parallel_requests,
            crop_target_size=crop_target_size,
            enable_affordance_caching=enable_affordance_caching,
            fast_mode=fast_mode,
            logger=logger,
            robot=robot,
            llm_client=llm_client,
        )
        self.update_interval = update_interval
        self.on_detection_complete = on_detection_complete

        # Async control
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Frame source
        self._frame_provider: Optional[Callable[[], tuple]] = None

        # Statistics
        self.stats = TrackingStats()
        self._last_detection_count: int = 0

        # Debug image streaming
        self._debug_save_dir: Optional[Path] = Path(debug_save_dir) if debug_save_dir else None
        self._debug_frame_index: int = 0
        self._debug_executor = threading.Thread(target=lambda: None, daemon=True)  # placeholder
        self._debug_lock = threading.Lock()

    @property
    def should_detect(self) -> bool:
        """Return whether detection should run for the current frame."""
        return True

    def _save_debug_frame(self, png_bytes: bytes, objects: List[Any], frame_index: int) -> None:
        """Render bounding boxes + labels onto the frame and write to debug_save_dir (background thread)."""
        if self._debug_save_dir is None:
            return
        save_debug_frame(png_bytes, objects, frame_index, self._debug_save_dir, self._debug_lock)

    def set_frame_provider(
        self,
        provider: Callable[[], tuple]
    ):
        """Set the frame provider function.

        The provider should return either a 3-tuple `(color, depth, intrinsics)`
        or a 4-tuple `(color, depth, intrinsics, robot_state)` captured atomically.
        """
        self._frame_provider = provider

    def start(self):
        """Start the background tracking task."""
        if self._running:
            self.logger.warning("Tracker already running")
            return

        if self._frame_provider is None:
            raise ValueError("Frame provider not set. Call set_frame_provider() first.")

        self._running = True
        self.stats.is_running = True

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._tracking_loop())
        self.logger.info("Continuous tracker started")

    async def stop(self):
        """Stop the background tracking task."""
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
        self.logger.info("Continuous tracker stopped")

    async def _tracking_loop(self):
        """Main tracking loop (runs as async task)."""
        self.logger.debug("Tracking loop started")

        while self._running:
            loop_start = time.time()

            try:
                provided = self._frame_provider()
                # Support 3-tuple or 4-tuple with robot_state
                if isinstance(provided, tuple) and len(provided) == 4:
                    color_frame, depth_frame, intrinsics, robot_state = provided
                else:
                    color_frame, depth_frame, intrinsics = provided
                    robot_state = None

                if not self.should_detect:
                    async with self._lock:
                        self.stats.total_frames += 1
                        self.stats.skipped_frames += 1
                        total_attempted = self.stats.total_frames
                        self.stats.cache_hit_rate = (
                            self.stats.skipped_frames / total_attempted
                            if total_attempted > 0 else 0.0
                        )

                    if self.on_detection_complete:
                        if asyncio.iscoroutinefunction(self.on_detection_complete):
                            await self.on_detection_complete(self._last_detection_count)
                        else:
                            self.on_detection_complete(self._last_detection_count)
                    continue

                detection_start = time.time()
                detected_objects = await self.detect_objects(
                    color_frame,
                    depth_frame,
                    intrinsics,
                    robot_state=robot_state
                )
                detection_time = time.time() - detection_start
                self._last_detection_count = len(detected_objects)

                async with self._lock:
                    self.stats.total_frames += 1
                    self.stats.total_detections += len(detected_objects)
                    self.stats.last_detection_time = detection_time

                    alpha = 0.1  # Smoothing factor
                    self.stats.avg_detection_time = (
                        alpha * detection_time +
                        (1 - alpha) * self.stats.avg_detection_time
                    )

                    total_attempted = self.stats.total_frames
                    self.stats.cache_hit_rate = (
                        self.stats.skipped_frames / total_attempted
                        if total_attempted > 0 else 0.0
                    )

                if self.on_detection_complete:
                    if asyncio.iscoroutinefunction(self.on_detection_complete):
                        await self.on_detection_complete(len(detected_objects))
                    else:
                        self.on_detection_complete(len(detected_objects))

                # Stream annotated debug frame in background if enabled
                if self._debug_save_dir is not None and self._last_detection_png is not None:
                    frame_idx = self._debug_frame_index
                    self._debug_frame_index += 1
                    png_snapshot = bytes(self._last_detection_png)
                    objects_snapshot = list(detected_objects)
                    t = threading.Thread(
                        target=self._save_debug_frame,
                        args=(png_snapshot, objects_snapshot, frame_idx),
                        daemon=True,
                    )
                    t.start()

            except Exception as e:
                self.logger.error("=" * 60)
                self.logger.error("CONTINUOUS TRACKING ERROR")
                self.logger.error("=" * 60)
                self.logger.error("Error type: %s", type(e).__name__)
                self.logger.error("Error message: %s", str(e))
                self.logger.error("Frame: %d", self.stats.total_frames)
                self.logger.exception("Full traceback:")
                self.logger.error("=" * 60)
                # Continue tracking despite error
                await asyncio.sleep(self.update_interval)

            elapsed = time.time() - loop_start
            if elapsed < self.update_interval:
                await asyncio.sleep(self.update_interval - elapsed)

        self.logger.debug("Tracking loop stopped")

    async def get_stats(self) -> TrackingStats:
        """Get current tracking statistics (async-safe)."""
        async with self._lock:
            return TrackingStats(
                total_frames=self.stats.total_frames,
                total_detections=self.stats.total_detections,
                skipped_frames=self.stats.skipped_frames,
                avg_detection_time=self.stats.avg_detection_time,
                last_detection_time=self.stats.last_detection_time,
                cache_hit_rate=self.stats.cache_hit_rate,
                is_running=self.stats.is_running
            )

    def is_running(self) -> bool:
        """Check if tracker is currently running."""
        return self._running
