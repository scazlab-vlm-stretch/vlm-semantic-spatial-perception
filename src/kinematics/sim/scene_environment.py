"""
PyBullet Scene Environment

GUI-mode simulation environment for visualising the xArm7 robot alongside
scene objects (blocks, surfaces, etc.) and capturing synthetic RGB-D frames
from the robot's wrist camera.

All PyBullet calls are made with an explicit physicsClientId so the GUI client
is fully isolated from any DIRECT clients created elsewhere in the same process
(e.g. XArmPybulletInterface).

Usage:
    env = SceneEnvironment()
    env.start()
    env.set_robot_joints([0.1, -1.4, -0.1, 1.3, -0.4, 1.9, -0.1])
    env.add_scene_objects(scene_objects)
    color, depth, intrinsics = env.capture_camera_frame()
    env.set_status("Running L1…")
    env.step(2.0)
    env.stop()
"""

from __future__ import annotations

import logging
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None
    pybullet_data = None

from src.camera.base_camera import CameraFrame, CameraIntrinsics

logger = logging.getLogger(__name__)

_SIM_DIR = Path(__file__).parent
_DEFAULT_URDF = _SIM_DIR / "urdfs" / "xarm7_camera" / "xarm7.urdf"

# Camera aimed at the work surface in front of the robot
CAMERA_AIM_JOINTS = [0.100085, -1.407677, -0.098652, 1.314592, 0.0, 2.0, -0.112296]

# Default colours per object_id; fallback is grey
OBJECT_COLORS: Dict[str, List[float]] = {
    "red_block_1":      [0.85, 0.15, 0.15, 1.0],
    "red_block_2":      [0.85, 0.15, 0.15, 1.0],
    "blue_block_1":     [0.15, 0.35, 0.85, 1.0],
    "blue_block_2":     [0.15, 0.35, 0.85, 1.0],
    "red_container_1":  [0.80, 0.10, 0.10, 0.75],
    "blue_container_1": [0.10, 0.25, 0.80, 0.75],
    "table_1":          [0.55, 0.45, 0.30, 1.0],
}

# Half-extents (metres) per object_type
OBJECT_HALF_EXTENTS: Dict[str, List[float]] = {
    "block":     [0.03, 0.03, 0.03],
    "surface":   [0.25, 0.25, 0.01],
    "container": [0.07, 0.07, 0.02],   # flat open tray / bin
}

# RealSense D435 approximate intrinsics
_CAM_WIDTH  = 640
_CAM_HEIGHT = 480
_CAM_FX     = 610.0
_CAM_FY     = 610.0
_CAM_NEAR   = 0.05
_CAM_FAR    = 3.0


class SceneEnvironment:
    """
    PyBullet GUI environment with robot, scene objects, and wrist camera.

    Uses a single GUI physics client. Every p.* call explicitly passes
    physicsClientId=self._client so this class is safe to use alongside
    other PyBullet clients (e.g. XArmPybulletInterface in DIRECT mode).
    """

    def __init__(
        self,
        urdf_path: Optional[Path] = None,
        camera_link: str = "camera_color_optical_frame",
        initial_joints: Optional[List[float]] = None,
        n_arm_joints: int = 7,
        tcp_link_name: str = "link_tcp",
        camera_use_world_up: bool = False,
        viewer_target: Optional[List[float]] = None,
        viewer_distance: float = 1.1,
        viewer_yaw: float = 45,
        viewer_pitch: float = -30,
    ) -> None:
        self.urdf_path     = Path(urdf_path or _DEFAULT_URDF)
        self.camera_link   = camera_link
        self.initial_joints = initial_joints or CAMERA_AIM_JOINTS
        self._n_arm_joints = n_arm_joints
        self.tcp_link_name = tcp_link_name
        self._camera_use_world_up  = camera_use_world_up
        self._viewer_target   = viewer_target or [0.35, 0.0, 0.2]
        self._viewer_distance = viewer_distance
        self._viewer_yaw      = viewer_yaw
        self._viewer_pitch    = viewer_pitch

        self._client: Optional[int] = None
        self._robot_id: Optional[int] = None
        self._obj_ids: Dict[str, int] = {}
        self._text_ids: List[int] = []
        self._movable_joints: List[int] = []   # all revolute/prismatic joints
        self._arm_joints: List[int] = []        # arm-only (first 7 revolute)
        self._gripper_joints: List[int] = []    # gripper finger joints (rest)
        self._link_name_to_index: Dict[str, int] = {}
        self._object_colors: Dict[str, List[float]] = {}
        self._step_thread: Optional[threading.Thread] = None
        self._step_running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to PyBullet GUI, load ground plane and robot."""
        if not PYBULLET_AVAILABLE:
            logger.warning("PyBullet not available — running without visualisation")
            return

        self._client = p.connect(p.GUI)
        c = self._client  # shorthand for all calls below

        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
        p.loadURDF("plane.urdf", physicsClientId=c)

        p.resetDebugVisualizerCamera(
            cameraDistance=self._viewer_distance,
            cameraYaw=self._viewer_yaw,
            cameraPitch=self._viewer_pitch,
            cameraTargetPosition=self._viewer_target,
            physicsClientId=c,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=c)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=c)

        if self.urdf_path.exists():
            self._robot_id = p.loadURDF(
                str(self.urdf_path),
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True,
                physicsClientId=c,
            )
            self._build_joint_map()
            self.set_robot_joints(self.initial_joints)
            logger.info("Loaded URDF (robot_id=%d, client=%d)", self._robot_id, c)
        else:
            logger.warning("URDF not found at %s — skipping robot", self.urdf_path)

        # Background thread: keeps the GUI responsive by stepping the sim at ~30 Hz
        # when the main thread is busy with detection / planning.
        self._step_running = True
        self._step_thread = threading.Thread(target=self._background_step, daemon=True)
        self._step_thread.start()

    def _background_step(self) -> None:
        """Continuously step the simulation at ~30 Hz to keep the GUI live."""
        while self._step_running and self._client is not None:
            try:
                p.stepSimulation(physicsClientId=self._client)
            except Exception:
                break
            time.sleep(1.0 / 30.0)

    def stop(self) -> None:
        """Disconnect from PyBullet."""
        self._step_running = False
        if self._step_thread is not None:
            self._step_thread.join(timeout=1.0)
            self._step_thread = None
        if PYBULLET_AVAILABLE and self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ------------------------------------------------------------------
    # Robot control
    # ------------------------------------------------------------------

    def set_robot_joints(self, joint_positions: List[float]) -> None:
        """Set the arm joint configuration (radians). Gripper joints are unchanged."""
        if not PYBULLET_AVAILABLE or self._robot_id is None:
            return
        c = self._client
        joints = self._arm_joints if self._arm_joints else self._movable_joints
        for i, joint_idx in enumerate(joints):
            if i < len(joint_positions):
                p.resetJointState(self._robot_id, joint_idx, float(joint_positions[i]),
                                  physicsClientId=c)

    # ------------------------------------------------------------------
    # Scene objects
    # ------------------------------------------------------------------

    def add_scene_objects(self, scene_objects: List[Dict]) -> None:
        """Add coloured box bodies for each scene object at their position_3d."""
        if not PYBULLET_AVAILABLE or self._client is None:
            return
        c = self._client

        for obj in scene_objects:
            oid   = obj["object_id"]
            otype = obj.get("object_type", "block")
            pos   = obj.get("position_3d", [0, 0, 0])
            half  = OBJECT_HALF_EXTENTS.get(oid, OBJECT_HALF_EXTENTS.get(otype, [0.03, 0.03, 0.03]))
            color = OBJECT_COLORS.get(oid, [0.6, 0.6, 0.6, 1.0])
            self._object_colors[oid] = color

            vis_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=c)
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=c)
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos,
                physicsClientId=c,
            )
            self._obj_ids[oid] = body
            p.addUserDebugText(
                oid,
                [pos[0], pos[1], pos[2] + half[2] + 0.025],
                textColorRGB=[1, 1, 1],
                textSize=0.8,
                physicsClientId=c,
            )

        logger.info("Added %d scene objects (client=%d)", len(scene_objects), c)

    def move_object(self, object_id: str, position: List[float]) -> bool:
        """Teleport a scene object to a new position. Returns True if found."""
        if not PYBULLET_AVAILABLE or self._client is None:
            return False
        body = self._obj_ids.get(object_id)
        if body is None:
            return False
        p.resetBasePositionAndOrientation(body, position, [0, 0, 0, 1], physicsClientId=self._client)
        return True

    def get_object_position(self, object_id: str) -> Optional[List[float]]:
        """Return current [x, y, z] position of a scene object, or None if not found."""
        if not PYBULLET_AVAILABLE or self._client is None:
            return None
        body = self._obj_ids.get(object_id)
        if body is None:
            return None
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self._client)
        return list(pos)

    def highlight_objects(self, object_ids: List[str], duration: float = 0.3) -> None:
        """Briefly flash objects yellow then restore their original colour."""
        if not PYBULLET_AVAILABLE or self._client is None:
            return
        c = self._client
        for oid in object_ids:
            body = self._obj_ids.get(oid)
            if body is not None:
                p.changeVisualShape(body, -1, rgbaColor=[1.0, 0.9, 0.1, 1.0],
                                    physicsClientId=c)
        p.stepSimulation(physicsClientId=c)
        time.sleep(duration)
        for oid in object_ids:
            body = self._obj_ids.get(oid)
            if body is not None:
                orig = self._object_colors.get(oid, [0.6, 0.6, 0.6, 1.0])
                p.changeVisualShape(body, -1, rgbaColor=orig, physicsClientId=c)

    # ------------------------------------------------------------------
    # Status overlay
    # ------------------------------------------------------------------

    def set_status(self, text: str, color: Optional[List[float]] = None) -> None:
        """Display a status string as debug text in the scene."""
        if not PYBULLET_AVAILABLE or self._client is None:
            return
        c = self._client
        for tid in self._text_ids:
            p.removeUserDebugItem(tid, physicsClientId=c)
        self._text_ids.clear()

        rgb = color or [0.2, 0.9, 0.2]
        for i, line in enumerate(textwrap.wrap(text, width=60)):
            tid = p.addUserDebugText(
                line,
                [-0.1, 0.6, 0.55 - i * 0.06],
                textColorRGB=rgb,
                textSize=1.0,
                physicsClientId=c,
            )
            self._text_ids.append(tid)

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self, seconds: float = 1.0) -> None:
        """Run the simulation for `seconds`, keeping the GUI responsive."""
        if not PYBULLET_AVAILABLE or self._client is None:
            return
        c = self._client
        steps = max(1, int(seconds * 60))
        for _ in range(steps):
            p.stepSimulation(physicsClientId=c)
            time.sleep(1.0 / 60.0)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def capture_camera_frame(
        self,
        width: int = _CAM_WIDTH,
        height: int = _CAM_HEIGHT,
        fx: float = _CAM_FX,
        fy: float = _CAM_FY,
        near: float = _CAM_NEAR,
        far: float = _CAM_FAR,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[CameraIntrinsics]]:
        """
        Render an RGB + depth frame from the robot's camera_color_optical_frame.

        Returns:
            (color_uint8, depth_metres, intrinsics)  — all None if unavailable.
        """
        if not PYBULLET_AVAILABLE or self._robot_id is None:
            return None, None, None

        c = self._client
        cam_idx = self._link_name_to_index.get(self.camera_link)
        if cam_idx is None:
            logger.warning("Camera link '%s' not found in URDF", self.camera_link)
            return None, None, None

        state = p.getLinkState(self._robot_id, cam_idx, physicsClientId=c)
        pos = np.array(state[4], dtype=float)
        rot = np.array(p.getMatrixFromQuaternion(state[5], physicsClientId=c)).reshape(3, 3)

        forward = rot[:, 2]  # Z = forward in optical frame
        target  = pos + forward

        if self._camera_use_world_up:
            # Build an up vector from world +Z, orthogonalised against forward.
            # Avoids roll errors when the URDF camera Y axis is not aligned with world Y.
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
            if right_norm > 1e-6:
                right /= right_norm
                up = np.cross(right, forward)
            else:
                up = -rot[:, 1]  # degenerate: forward ≈ ±Z, fall back to URDF up
        else:
            up = -rot[:, 1]  # Y = down → up = -Y (standard optical frame)

        view = p.computeViewMatrix(
            cameraEyePosition=pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=c,
        )
        fov_y = 2.0 * np.degrees(np.arctan2(height / 2.0, fy))
        proj = p.computeProjectionMatrixFOV(
            fov=fov_y,
            aspect=width / height,
            nearVal=near,
            farVal=far,
            physicsClientId=c,
        )

        _, _, rgba, depth_buf, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=c,
        )

        color = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        d = np.array(depth_buf, dtype=np.float32).reshape(height, width)
        depth = (far * near / (far - d * (far - near))).astype(np.float32)

        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy,
            cx=width / 2.0, cy=height / 2.0,
            width=width, height=height,
        )

        valid = depth[depth < far * 0.99]
        logger.info(
            "Sim camera frame %dx%d — depth [%.2f, %.2f]m (client=%d)",
            width, height,
            float(depth.min()),
            float(valid.max()) if valid.size > 0 else 0.0,
            c,
        )
        return color, depth, intrinsics

    # ------------------------------------------------------------------
    # Robot state — duck-typed interface for config.robot
    # ------------------------------------------------------------------

    def get_robot_joint_state(self) -> Optional[np.ndarray]:
        """Return current arm joint positions (radians). Gripper joints excluded."""
        if not PYBULLET_AVAILABLE or self._robot_id is None:
            return None
        c = self._client
        joints = self._arm_joints if self._arm_joints else self._movable_joints
        angles = []
        for joint_idx in joints:
            state = p.getJointState(self._robot_id, joint_idx, physicsClientId=c)
            angles.append(state[0])
        return np.array(angles, dtype=float)

    def get_robot_tcp_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (position, quaternion_xyzw) of the TCP in world frame."""
        if not PYBULLET_AVAILABLE or self._robot_id is None:
            return None
        c = self._client
        tcp_idx = self._link_name_to_index.get(self.tcp_link_name)
        if tcp_idx is None:
            return None
        state = p.getLinkState(self._robot_id, tcp_idx, physicsClientId=c)
        pos = np.array(state[4], dtype=float)
        orn = np.array(state[5], dtype=float)  # xyzw
        return pos, orn

    def get_camera_transform(self) -> Tuple[Optional[np.ndarray], Optional[object]]:
        """Return (position, Rotation) of the wrist camera in world frame."""
        from scipy.spatial.transform import Rotation
        if not PYBULLET_AVAILABLE or self._robot_id is None:
            return None, None
        c = self._client
        cam_idx = self._link_name_to_index.get(self.camera_link)
        if cam_idx is None:
            return None, None
        state = p.getLinkState(self._robot_id, cam_idx, physicsClientId=c)
        pos = np.array(state[4], dtype=float)
        orn = np.array(state[5], dtype=float)  # xyzw
        rot = Rotation.from_quat(orn)
        return pos, rot

    def get_robot_state(self) -> Dict:
        """Return JSON-serialisable robot state dict (matches duck-typed interface)."""
        import time as _time
        state: Dict = {"stamp": _time.time(), "provider": "SceneEnvironment"}
        joints = self.get_robot_joint_state()
        if joints is not None:
            state["joints"] = joints.tolist()
        tcp = self.get_robot_tcp_pose()
        if tcp is not None:
            pos, quat = tcp
            state["tcp_pose"] = {
                "position": pos.tolist(),
                "quaternion_xyzw": quat.tolist(),
            }
        cam_pos, cam_rot = self.get_camera_transform()
        if cam_pos is not None:
            state["camera"] = {
                "position": cam_pos.tolist(),
                "quaternion_xyzw": cam_rot.as_quat().tolist(),
            }
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_joint_map(self) -> None:
        c = self._client
        num = p.getNumJoints(self._robot_id, physicsClientId=c)
        for i in range(num):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=c)
            link_name  = info[12].decode("utf-8")
            joint_type = info[2]
            self._link_name_to_index[link_name] = i
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self._movable_joints.append(i)
                if len(self._arm_joints) < self._n_arm_joints:
                    self._arm_joints.append(i)
                else:
                    self._gripper_joints.append(i)
