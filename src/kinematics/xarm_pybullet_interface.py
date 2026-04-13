"""xArm7 PyBullet interface — thin subclass of BasePybulletInterface with xArm7 defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from src.kinematics.base_pybullet_interface import BasePybulletInterface

_SIM_DIR = Path(__file__).parent / "sim"
_XARM7_URDF = _SIM_DIR / "urdfs" / "xarm7_camera" / "xarm7.urdf"
_CAMERA_LINK = "camera_color_optical_frame"
_TCP_LINK = "link_tcp"
_BASE_LINK = "link_base"
_N_ARM_JOINTS = 7


class XArmPybulletInterface(BasePybulletInterface):
    def __init__(
        self,
        urdf_path: Optional[Union[str, Path]] = None,
        camera_link_name: str = _CAMERA_LINK,
        tcp_link_name: str = _TCP_LINK,
        static_camera_tf: Optional[Any] = None,
    ) -> None:
        super().__init__(
            urdf_path=urdf_path or _XARM7_URDF,
            camera_link_name=camera_link_name,
            tcp_link_name=tcp_link_name,
            base_link_name=_BASE_LINK,
            n_arm_joints=_N_ARM_JOINTS,
            static_camera_tf=static_camera_tf,
        )


def create_sim_interface(
    camera_link: str = _CAMERA_LINK,
    static_camera_tf: Optional[Any] = None,
) -> XArmPybulletInterface:
    return XArmPybulletInterface(camera_link_name=camera_link, static_camera_tf=static_camera_tf)
