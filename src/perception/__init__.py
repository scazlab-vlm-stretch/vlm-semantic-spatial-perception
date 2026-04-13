"""
Perception module for VLM-based object detection and spatial understanding.

This module provides interfaces to Vision-Language Models (VLMs) like
Gemini Robotics-ER for:
- Object detection and localization
- Spatial reasoning
- Trajectory planning
- Task decomposition
"""


from .object_tracker import ObjectTracker, ContinuousObjectTracker, TrackingStats
from .object_registry import (
    DetectedObject,
    InteractionPoint,
    DetectedObjectRegistry
)
from .gsam2_object_tracker import GSAM2ObjectTracker, GSAM2ContinuousObjectTracker

__all__ = [
    # Object tracking system (focused on detection & affordances)
    "ObjectTracker",
    "DetectedObject",
    "InteractionPoint",
    "DetectedObjectRegistry",
    # Continuous background tracking
    "ContinuousObjectTracker",
    "TrackingStats",
    # GSAM2-based tracking (RAM+ + GroundingDINO + SAM2)
    "GSAM2ObjectTracker",
    "GSAM2ContinuousObjectTracker",
]
