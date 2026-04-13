"""
Simulation environment for xArm7 — PyBullet-based, no physical robot required.
"""

from src.kinematics.sim.scene_environment import SceneEnvironment, CAMERA_AIM_JOINTS, OBJECT_COLORS, OBJECT_HALF_EXTENTS
from src.kinematics.sim.pybullet_camera import PyBulletCamera
from src.kinematics.sim.transform_calculator import TransformCalculator

__all__ = [
    "SceneEnvironment",
    "CAMERA_AIM_JOINTS",
    "OBJECT_COLORS",
    "OBJECT_HALF_EXTENTS",
    "PyBulletCamera",
    "TransformCalculator",
]
