"""
Base PyBullet Simulation Interface

General-purpose base class for PyBullet FK/transform computation.
Robot-specific subclasses provide the URDF path and link names.

Duck-typed interface (same signatures as CuRoboMotionPlanner):
  get_robot_joint_state()      -> np.ndarray | None
  get_robot_tcp_pose()         -> (pos, quat_xyzw) | None
  get_camera_transform()       -> (pos, Rotation) | (None, None)
  get_robot_state()            -> dict
  get_forward_kinematics()     -> (pos, quat_wxyz) | None
  convert_cam_pose_to_base()   -> (pos, quat_xyzw)
  set_current_joint_state()    -> None

No robot connection, no CUDA, no curobo required.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None
    pybullet_data = None

logger = logging.getLogger(__name__)


class BasePybulletInterface:
    """
    General PyBullet kinematics interface.

    Maintains a simulated joint state (no physical robot). FK and camera
    transform queries run against the loaded URDF.
    """

    def __init__(
        self,
        urdf_path: Union[str, Path],
        camera_link_name: str,
        tcp_link_name: str,
        base_link_name: str,
        n_arm_joints: int,
        static_camera_tf: Optional[Any] = None,
    ) -> None:
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "PyBullet is required. Install with: pip install pybullet"
            )

        self.urdf_path = str(urdf_path)
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        self.camera_link_name = camera_link_name
        self.tcp_link_name = tcp_link_name
        self.base_link_name = base_link_name

        self._physics_client: Optional[int] = None
        self._robot_id: Optional[int] = None
        self._movable_joints: List[int] = []  # indices of non-fixed joints
        self._link_name_to_index: Dict[str, int] = {}

        # Current simulated joint state
        self._joints = np.zeros(n_arm_joints, dtype=np.float64)

        # Static camera transform (optional, same semantics as CuRoboMotionPlanner)
        self.static_camera_tf: Optional[np.ndarray] = None
        self.static_camera_position: Optional[np.ndarray] = None
        self.static_camera_rotation: Optional[Rotation] = None
        if static_camera_tf is not None:
            self._parse_static_camera_tf(static_camera_tf)

        self._init_pybullet()
        logger.info("%s ready (URDF=%s)", type(self).__name__, self.urdf_path)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_pybullet(self) -> None:
        if self._physics_client is not None:
            return
        self._physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
        )
        self._build_joint_map()
        logger.debug("PyBullet DIRECT loaded robot_id=%d", self._robot_id)

    def _build_joint_map(self) -> None:
        """Populate movable joint list and link-name → index map."""
        num = p.getNumJoints(self._robot_id)
        self._movable_joints = []
        self._link_name_to_index = {}
        for i in range(num):
            info = p.getJointInfo(self._robot_id, i)
            link_name: str = info[12].decode("utf-8")
            joint_type: int = info[2]
            self._link_name_to_index[link_name] = i
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self._movable_joints.append(i)

    def _get_link_index(self, link_name: str) -> Optional[int]:
        if link_name == self.base_link_name:
            return -1
        return self._link_name_to_index.get(link_name)

    def _apply_joints_to_sim(self) -> None:
        """Push self._joints into PyBullet for FK computation."""
        for i, joint_idx in enumerate(self._movable_joints):
            if i < len(self._joints):
                p.resetJointState(self._robot_id, joint_idx, float(self._joints[i]))

    def _get_link_world_pose(
        self, link_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion_xyzw) of a link in world frame."""
        if link_index == -1:
            pos, orn = p.getBasePositionAndOrientation(self._robot_id)
        else:
            state = p.getLinkState(self._robot_id, link_index)
            pos = state[4]
            orn = state[5]
        return np.array(pos, dtype=float), np.array(orn, dtype=float)

    # ------------------------------------------------------------------
    # Static camera transform (same as CuRoboMotionPlanner)
    # ------------------------------------------------------------------

    def _parse_static_camera_tf(self, static_camera_tf: Any) -> None:
        if isinstance(static_camera_tf, np.ndarray) and static_camera_tf.shape == (4, 4):
            self.static_camera_tf = static_camera_tf
            self.static_camera_position = static_camera_tf[:3, 3]
            self.static_camera_rotation = Rotation.from_matrix(static_camera_tf[:3, :3])
        elif isinstance(static_camera_tf, (tuple, list)) and len(static_camera_tf) == 2:
            pos, rot = static_camera_tf
            self.static_camera_position = np.array(pos, dtype=float)
            if isinstance(rot, Rotation):
                self.static_camera_rotation = rot
            elif isinstance(rot, np.ndarray) and rot.shape == (3, 3):
                self.static_camera_rotation = Rotation.from_matrix(rot)
            elif isinstance(rot, (list, tuple, np.ndarray)) and len(rot) == 4:
                self.static_camera_rotation = Rotation.from_quat(np.array(rot))
            else:
                logger.warning("Unrecognised rotation format in static_camera_tf")
                return
            # Build 4x4 matrix for reference
            T = np.eye(4)
            T[:3, :3] = self.static_camera_rotation.as_matrix()
            T[:3, 3] = self.static_camera_position
            self.static_camera_tf = T
        else:
            logger.warning("Unrecognised static_camera_tf format, ignoring")

    # ------------------------------------------------------------------
    # Duck-typed robot interface
    # ------------------------------------------------------------------

    def set_current_joint_state(self, joint_positions: Any) -> None:
        """Set the simulated joint state."""
        arr = np.asarray(joint_positions, dtype=float).flatten()
        self._joints = arr[: len(self._movable_joints)]

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Return current simulated joint positions (radians)."""
        return self._joints.copy()

    def get_robot_tcp_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (position, quaternion_xyzw) of the TCP in robot base frame.

        Uses PyBullet FK at the current joint state.
        """
        try:
            self._apply_joints_to_sim()
            tcp_idx = self._get_link_index(self.tcp_link_name)
            base_idx = self._get_link_index(self.base_link_name)

            if tcp_idx is None:
                logger.warning("TCP link '%s' not found in URDF", self.tcp_link_name)
                return None

            tcp_pos_w, tcp_orn_w = self._get_link_world_pose(tcp_idx)
            base_pos_w, base_orn_w = self._get_link_world_pose(base_idx)

            T_w_tcp = _build_T(tcp_pos_w, tcp_orn_w)
            T_w_base = _build_T(base_pos_w, base_orn_w)
            T_base_tcp = np.linalg.inv(T_w_base) @ T_w_tcp

            pos = T_base_tcp[:3, 3]
            rot = Rotation.from_matrix(T_base_tcp[:3, :3])
            quat_xyzw = rot.as_quat()  # [x, y, z, w]
            return pos, quat_xyzw
        except Exception:
            logger.exception("get_robot_tcp_pose failed")
            return None

    def get_camera_transform(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[Rotation]]:
        """
        Return (position, Rotation) of the camera link in robot base frame.

        Matches CuRoboMotionPlanner.get_camera_transform() signature.
        If a static transform was set, that is returned directly.
        Otherwise computes via FK.
        """
        if self.static_camera_tf is not None:
            return self.static_camera_position, self.static_camera_rotation

        try:
            self._apply_joints_to_sim()
            cam_idx = self._get_link_index(self.camera_link_name)
            base_idx = self._get_link_index(self.base_link_name)

            if cam_idx is None:
                logger.warning("Camera link '%s' not found in URDF", self.camera_link_name)
                return None, None

            cam_pos_w, cam_orn_w = self._get_link_world_pose(cam_idx)
            base_pos_w, base_orn_w = self._get_link_world_pose(base_idx)

            T_w_cam = _build_T(cam_pos_w, cam_orn_w)
            T_w_base = _build_T(base_pos_w, base_orn_w)
            T_base_cam = np.linalg.inv(T_w_base) @ T_w_cam

            pos = T_base_cam[:3, 3]
            rot = Rotation.from_matrix(T_base_cam[:3, :3])
            return pos, rot
        except Exception:
            logger.exception("get_camera_transform failed")
            return None, None

    def get_forward_kinematics(
        self, joint_positions: Optional[Any] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute FK for given joint positions (or current state if None).

        Returns:
            (position, quaternion_wxyz) matching CuRoboMotionPlanner convention,
            or None on failure.
        """
        try:
            if joint_positions is not None:
                saved = self._joints.copy()
                self.set_current_joint_state(joint_positions)
                result = self.get_robot_tcp_pose()
                self._joints = saved
            else:
                result = self.get_robot_tcp_pose()

            if result is None:
                return None
            pos, quat_xyzw = result
            # CuRoboMotionPlanner returns wxyz convention
            x, y, z, w = quat_xyzw
            quat_wxyz = np.array([w, x, y, z], dtype=float)
            return pos, quat_wxyz
        except Exception:
            logger.exception("get_forward_kinematics failed")
            return None

    def get_robot_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable state dict.

        Same structure as CuRoboMotionPlanner.get_robot_state() so
        ObjectTracker snapshots stay consistent.
        """
        state: Dict[str, Any] = {
            "stamp": time.time(),
            "provider": type(self).__name__,
        }

        joints = self.get_robot_joint_state()
        if joints is not None:
            state["joints"] = joints.tolist()

        tcp = self.get_robot_tcp_pose()
        if isinstance(tcp, tuple) and len(tcp) == 2:
            pos, quat = tcp
            state["tcp_pose"] = {
                "position": pos.tolist() if pos is not None else None,
                "quaternion_xyzw": quat.tolist() if quat is not None else None,
            }

        cam_pos, cam_rot = self.get_camera_transform()
        if cam_pos is not None:
            state["camera"] = {
                "position": cam_pos.tolist(),
                "quaternion_xyzw": cam_rot.as_quat().tolist() if cam_rot is not None else None,
            }

        if self.static_camera_position is not None:
            state["static_camera"] = {
                "position": self.static_camera_position.tolist(),
                "quaternion_xyzw": self.static_camera_rotation.as_quat().tolist()
                if self.static_camera_rotation is not None
                else None,
            }

        return state

    def convert_cam_pose_to_base(
        self,
        position: Any,
        orientation: Any,
        do_translation: bool = True,
        debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a pose expressed in camera frame to robot base frame.

        Matches CuRoboMotionPlanner.convert_cam_pose_to_base().

        Args:
            position: (3,) position in camera frame
            orientation: quaternion [x,y,z,w] or (3,3) rotation matrix
            do_translation: apply translation component (default True)
            debug: log the result

        Returns:
            (position_base, quaternion_xyzw_base)
        """
        cam_pos, cam_rot = self.get_camera_transform()
        if cam_rot is None:
            raise RuntimeError("Camera transform not available")

        pos = np.asarray(position, dtype=float)

        # Rotate position into base frame
        transformed_position = cam_rot.apply(pos)
        if do_translation and cam_pos is not None:
            transformed_position += cam_pos

        if debug:
            logger.debug("convert_cam_pose_to_base result: %s", transformed_position)

        # Handle orientation
        if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
            input_rotation = Rotation.from_matrix(orientation)
        elif isinstance(orientation, (np.ndarray, list, tuple)) and np.asarray(orientation).shape == (4,):
            input_rotation = Rotation.from_quat(np.asarray(orientation))
        else:
            raise ValueError(f"Unsupported orientation format: {type(orientation)}")

        combined = cam_rot * input_rotation
        return transformed_position, combined.as_quat()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_T(pos: np.ndarray, orn_xyzw: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from position and xyzw quaternion."""
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(orn_xyzw)).reshape(3, 3)
    T[:3, 3] = pos
    return T
