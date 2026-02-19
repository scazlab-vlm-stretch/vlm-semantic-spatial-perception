import numpy as np
import time
import torch
import math
import threading
from typing import List, Dict, Optional, Tuple, Union, Any
import os
import yaml
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
import pathlib
from urchin import URDF
from scipy.spatial.transform import Rotation as R


#Stretch imports
from stretch_body.device import Device
import stretch_body.base as base
import stretch_body.arm as arm
import stretch_body.lift as lift
import stretch_body.pimu as pimu
import stretch_body.head as head
import stretch_body.wacc as wacc
import stretch_body.hello_utils as hello_utils
import stretch_body.robot as stretch_robot

# # CuRobo imports -- all the imports I need stretch import equivalents
# from curobo.src.curobo.geom.sdf.world import CollisionCheckerType
# # from curobo.geom.types import Cuboid, WorldConfig, Mesh
# from curobo.src.curobo.types.base import TensorDeviceType
# # from curobo.types.math import Pose
# # from curobo.types.robot import JointState
# from curobo.src.curobo.types.robot import RobotConfig
# # from curobo.rollout.cost.pose_cost import PoseCostMetric
# from curobo.src.curobo.util_file import get_robot_configs_path, load_yaml
# from curobo.src.curobo.wrap.reacher.motion_gen import (
#     MotionGen, 
#     MotionGenConfig
#     # MotionGenPlanConfig, 
#     # MotionGenResult, 
#     # MotionGenStatus
# )
# from curobo.src.curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
# # from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGenerator, CudaRobotGeneratorConfig
# # xArm SDK import
# from xarm.wrapper import XArmAPI

#SIMPLIFY TO:
# end_of_arm

class RobotConfig:
    def __init__(self):
        self.dof = 1
        self.robot_type = "lite"
        self.prefix = ""
        self.arm_group_name = ""
        self.gripper_group_name = ""
        self.tcp_link = ""
        self.is_lite6 = True
        self.robot_ip = ""
        self.robot_cfg = None

class StretchMotionPlanner:
    def __init__(self, config=None, robot_ip=None, static_camera_tf=None):
        """
        Initialize CuRobo motion planner with Stretch SDK integration
        
        Args:
            config: Optional configuration object
            robot_ip: IP address of the xArm robot
            static_camera_tf: Optional SE(3) transform from static camera to robot base frame.
                             Can be provided as:
                             - 4x4 homogeneous transformation matrix (numpy array)
                             - tuple/list of (translation_vector, rotation) where rotation is 
                               quaternion [x,y,z,w] or 3x3 rotation matrix
        """
        # Initialize configuration
        self.config = config or self.initialize_default_config()
        
        # Set robot IP if provided
        if robot_ip is not None:
            self.config.robot_ip = robot_ip
        
        # Store static camera transform
        self.static_camera_tf = None
        self.static_camera_position = None
        self.static_camera_rotation = None
        
        # if static_camera_tf is not None:
        #     self._parse_static_camera_tf(static_camera_tf)
        
        # # Initialize TensorDeviceType for cuRobo
        # self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        
        # # Store collision objects
        # self.collision_objects = []
        # self.collision_object_lock = threading.Lock()
        
        # Current robot state
        self.current_joints = None
        self.joint_state_lock = threading.Lock()
        
        # Initialize Stretch SDK
        self.arm = None
        self.robot = None
        self.arm_lock = threading.Lock()
        
        # # Force/torque sensor settings for robust interactions
        # self.default_collision_sensitivity = 1  # Lower = more sensitive (0-5)
        # self.default_teach_sensitivity = 1       # Lower = more sensitive (0-5) 
        # self.pivot_collision_sensitivity = 1     # Less sensitive for pivot operations
        # self.pivot_teach_sensitivity = 1         # Less sensitive for pivot operations
        
        # # Dynamic sensitivity adjustment based on torque errors  
        # self.max_collision_sensitivity = 1       # Maximum collision sensitivity (least sensitive) 
        # self.max_teach_sensitivity = 1           # Maximum teach sensitivity (least sensitive)
        # self.min_collision_sensitivity = 0       # Minimum collision sensitivity (most sensitive)
        # self.min_teach_sensitivity = 0           # Minimum teach sensitivity (most sensitive)
        # self.current_pivot_collision_sensitivity = self.pivot_collision_sensitivity
        # self.current_pivot_teach_sensitivity = self.pivot_teach_sensitivity
        
        # # Alternative torque handling parameters
        # self.min_speed_factor = 0.1             # Minimum speed for torque issues
        # self.max_segments = 20                  # Maximum trajectory segments
        # self.torque_retry_segments = [10, 15, 20]  # Escalating segment counts
        
        self.connect_robot()
        
        # Configure initial robot sensitivity for normal operations
        # if self.arm:
        #     self._configure_initial_sensitivity()
        
        # Register callback for joint state updates
        # if self.arm is not None and hasattr(self.arm, "register_report_location_callback"):
        #     self.arm.register_report_location_callback(self.joint_state_callback)
        
        # Initialize cuRobo motion generator -- for now: CUDA ROBOT MODEL MOTION GENERATOR is probably unnecessary for our uses, but bring back if it proves difficult to integrate w/o it.
        # self.motion_gen = self.init_curobo()
        
        self.initial_position = None

        if self.arm is not None:
            # if not self.arm.is_homed():
            #     self.arm.home()
            self.initial_position = self.get_robot_joint_state()

            # if hasattr(self.arm, "get_servo_angle"):
            #     _, self.initial_position = self.arm.get_servo_angle(is_radian=True)
            # else:
            #     self.initial_position = self.get_robot_joint_state()
        
        # setup Stretch's urdf
        urdf_path = str((pathlib.Path(hello_utils.get_fleet_directory()) / 'exported_urdf' / 'stretch.urdf').absolute())
        self.urdf = URDF.load(urdf_path, lazy_load_meshes=True)

        # Initialize IK solver only if motion generator succeeded
        # if self.motion_gen is not None:
        #     self.ik_solver = self.init_ik_solver()
        # else:
        #     print("Cannot initialize IK solver: motion generator initialization failed")
        #     self.ik_solver = None
        
        print('CuRobo Motion Planner initialized with Stretch SDK')
        if self.static_camera_tf is not None:
            print('Using static camera transform for pose conversions')
            print(f"Transform: \n {self.static_camera_tf}")


    def initialize_default_config(self) -> RobotConfig:
        """Initialize default configuration for the planner"""
        config = RobotConfig()
        
        # Set default parameters -- adjust this to Stretch
        config.dof = 1 #for now, just wrist
        config.robot_type = "stretch"
        # config.prefix = ""
        # config.robot_ip = "192.168.1.224" #??
        
        # # Check if it's a Lite6 robot
        # config.is_lite6 = (config.robot_type == "lite" and config.dof == 6)
        
        # # Calculate planning group
        # config.arm_group_name = f"{config.prefix}{config.robot_type}{config.dof}"
        
        # # Similarly for gripper group
        # config.gripper_group_name = f"{config.prefix}{config.robot_type}_gripper"
        # config.tcp_link = f"{config.prefix}link_tcp"
        
        print(
            f"Configuration loaded: robot_type={config.robot_type}, dof={config.dof}, "
            # f"is_lite6={config.is_lite6}, robot_ip={config.robot_ip}"
        )
        return config

    def connect_robot(self): #FOR NOW - just focus on "arm" of Stretch
        """Connect to the physical robot using Stretch SDK"""
        try:
            with self.arm_lock:
                self.robot = stretch_robot.Robot()
                self.robot.startup()
                self.arm = self.robot

                joints = self._get_stretch_joint_positions()
                # code, angles = self.arm.get_servo_angle(is_radian=True)
                if joints is not None:
                    self.set_current_joint_state(joints)
                print("Stretch robot connected successfully")
        except Exception as e:
            print(f"Failed to connect to the Stretch robot: {str(e)}")
            self.robot = None
            self.arm = None
                
    def _get_stretch_joint_positions(self) -> Optional[List[float]]:
        #JUST THE END OF ARM -- may have to run self.arm.pull_status()
        if self.arm is None:
            return None
        self.arm.pull_status()

        status = getattr(self.arm, "status", None)
        if not status:
            return []
        end_of_arm = status.get("end_of_arm")
        if not end_of_arm:
            return []
        joints = []
        for name in ("wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"):
            joint = end_of_arm.get(name)
            if joint is not None:
                pos = joint.get("pos") #vel, effort
                if pos is not None:
                    joints.append(pos)
        return joints
        
        # if self.robot is None:
        #     return None
        # joints = []
        # status = getattr(self.robot, "status", {}) or {}
        # if "lift" in status and "pos" in status["lift"]:
        #     joints.append(status["lift"]["pos"])
        # if "arm" in status and "pos" in status["arm"]:
        #     joints.append(status["arm"]["pos"])
        # end_of_arm = status.get("end_of_arm", {}) or {}
        # for name in ("wrist_yaw", "wrist_pitch", "wrist_roll", "gripper"):
        #     if name in end_of_arm and "pos" in end_of_arm[name]:
        #         joints.append(end_of_arm[name]["pos"])
        # if self.config and self.config.dof and len(joints) < self.config.dof:
        #     joints.extend([0.0] * (self.config.dof - len(joints)))
        # return joints[: self.config.dof] if self.config and self.config.dof else joints


    def set_current_joint_state(self, joint_positions):
        """Set current joint state for planning"""
        with self.joint_state_lock:
            self.current_joints = np.array(joint_positions[: self.config.dof]) #need to splice to dof (?) or just show everything for debugging purposes


    def get_robot_state(self) -> Dict[str, Any]:
        """
        Duck-typed robot state for orchestrator snapshots.

        Returns a JSON-serializable dict capturing current joints, TCP pose,
        and camera transform when available. Fields are optional and omitted
        when unavailable; the orchestrator does not rely on any fixed schema.
        """
        state: Dict[str, Any] = {
            "stamp": time.time(),
            "provider": type(self).__name__,
        }
        try:
            joints = self.get_robot_joint_state()
            if joints is not None:
                state["joints"] = np.asarray(joints, dtype=float).tolist()
        except Exception:
            pass

        try:
            tcp = self.get_robot_tcp_pose() ###
            if isinstance(tcp, tuple) and len(tcp) == 2:
                pos, quat = tcp
                state["tcp_pose"] = {
                    "position": np.asarray(pos, dtype=float).tolist() if pos is not None else None,
                    "quaternion_xyzw": np.asarray(quat, dtype=float).tolist() if quat is not None else None,
                }
        except Exception:
            pass

        # try:
        #     cam_tf = self.get_camera_transform() ###
        #     if isinstance(cam_tf, tuple) and len(cam_tf) == 2:
        #         cam_pos, cam_quat = cam_tf
        #         # Provide a predictable sub-structure for camera transform
        #         state["camera"] = {
        #             "position": np.asarray(cam_pos, dtype=float).tolist() if cam_pos is not None else None,
        #             "quaternion_xyzw": np.asarray(cam_quat, dtype=float).tolist() if cam_quat is not None else None,
        #         }
        # except Exception:
        #     pass

        # Include static camera transform when set
        # if self.static_camera_position is not None or self.static_camera_rotation is not None:
        #     try:
        #         quat = None
        #         if self.static_camera_rotation is not None:
        #             quat_np = self.static_camera_rotation.as_quat()  # xyzw
        #             quat = np.asarray(quat_np, dtype=float).tolist()
        #         pos = np.asarray(self.static_camera_position, dtype=float).tolist() if self.static_camera_position is not None else None
        #         state["static_camera"] = {
        #             "position": pos,
        #             "quaternion_xyzw": quat,
        #         }
        #     except Exception:
        #         pass

        return state
    

    def get_robot_joint_state(self):
        """Get current joint state from the physical robot
        
        Returns:
            numpy.ndarray: Joint positions in radians or None if not available
        """
        if self.arm is None:
            print("Robot not connected")
            return None
            
        try:
            with self.arm_lock:
                if hasattr(self.arm, "_g et_stretch_join_positions"):
                    joints = self._get_stretch_joint_positions()
                    if joints is None:
                        print(f"Failed to get joint state, error code")
                        return None
                    self.set_current_joint_state(joints)
                    return np.array(joints)
        except Exception as e:
            print(f"Error getting robot joint state: {str(e)}")
            return None
    
    def get_robot_tcp_pose(self): #tcp = tool center point = likely the tip of gripper.
        """Get current TCP position and orientation from the physical robot
        
        Returns:
            tuple: (position, orientation) or None if not available
        """
        if self.arm is None:
            print("Robot not connected")
            return None
    
        try:
            with self.arm_lock:
                joints = self.get_robot_joint_state()
                if joints is None:
                    return None
                
                T = self.get_transform("base_link", "link_grasp_center")
                pose = T[:3, 3]
                quat = R.from_matrix(T[:3,3]).as_quat()
                #[[R R R x]
                #  [R R R y]
                #  [R R R z]
                #  [0 0 0 1]]
                # R is rotation matrix

                print(f"RObiot pose: {pose} quat: {quat}")
                return pose, quat
        except Exception as e:
            print(f"Error getting robot TCP pose: {str(e)}")
            return None
        

    def get_camera_transform(self):
            # Check if motion generator is properly initialized
            if self.motion_gen is None:
                print("Motion generator not initialized")
                return None, None
            
            if not hasattr(self.motion_gen, 'kinematics') or self.motion_gen.kinematics is None:
                print("Motion generator kinematics not available")
                return None, None
                
            joints = self.get_robot_joint_state()
            if joints is None:
                return None, None
            config = torch.from_numpy(np.array(joints))
            config = config.cuda("cuda")
            config = config.to(torch.float32)
            
            state = self.motion_gen.kinematics.get_state(config)
            # Extract camera pose and quaternion with better debugging
            camera_pose = state.links_position.cpu().numpy()[0][1]  # [x, y, z]
            camera_quat_raw = state.links_quaternion.cpu().numpy()[0][1]
             # Handle different quaternion formats that might be returned
            if len(camera_quat_raw.shape) == 1 and camera_quat_raw.shape[0] == 4:
                # Simple quaternion array [x, y, z, w]
                camera_quat = camera_quat_raw
            elif len(camera_quat_raw.shape) == 2:
                # Get the last quaternion if multiple are returned
                camera_quat = camera_quat_raw[-1]
            elif len(camera_quat_raw.shape) == 3:
                # 3D array - get the last element along the first dimension
                camera_quat = camera_quat_raw[-1]
                if len(camera_quat.shape) == 2:
                    # If still 2D, flatten or take appropriate element
                    if camera_quat.shape[0] == 1:
                        camera_quat = camera_quat[0]
                    else:
                        camera_quat = camera_quat.flatten()[:4]  # Take first 4 elements
            else:
                raise ValueError(f"Unexpected quaternion shape: {camera_quat_raw.shape}")
            
            # print(f"Debug - Final camera_quat shape: {camera_quat.shape}, value: {camera_quat}")
            
            # Ensure we have exactly 4 elements for quaternion
            if camera_quat.shape[0] != 4:
                raise ValueError(f"Expected 4 quaternion elements, got {camera_quat.shape[0]}")
            import copy
            # Create transformation matrix from end-effector to base
            camera_quat_copy = copy.deepcopy(camera_quat)
            camera_quat = np.array([camera_quat_raw[1], camera_quat_raw[2], camera_quat_raw[3], camera_quat_raw[0]])
            
            # print(f"Debug - joint state: {config}")
            # print(f"Debug - camera_pose shape: {camera_pose.shape}, value: {list(camera_pose)}")
            # print(f"Debug - camera_quat_raw shape: {camera_quat.shape}")
            # print(f"Debug - camera_quat_raw: {list(camera_quat)}")
            # print(f"Debug - camera_quat_raw norm: {np.linalg.norm(list(camera_quat_raw))}")
            # print()
            # print()
            # print()
            
            camera_rotation = Rotation.from_quat(camera_quat)
            return camera_pose, camera_rotation
        
    
    def _get_current_configuration(self):
        def bound_range(name, value):
            return min(max(value, self.urdf.joint_map[name].limit.lower), self.urdf.joint_map[name].limit.upper)

        tool = self.arm.end_of_arm.name
        if tool == 'tool_stretch_gripper':
            q_lift = bound_range('joint_lift', self.body.lift.status['pos'])
            q_arml = bound_range('joint_arm_l0', self.body.arm.status['pos'] / 4.0)
            q_yaw = bound_range('joint_wrist_yaw', self.body.end_of_arm.status['wrist_yaw']['pos'])
            q_pan = bound_range('joint_head_pan', self.body.head.status['head_pan']['pos'])
            q_tilt = bound_range('joint_head_tilt', self.body.head.status['head_tilt']['pos'])
            return {
                'joint_left_wheel': 0.0,
                'joint_right_wheel': 0.0,
                'joint_lift': q_lift,
                'joint_arm_l0': q_arml,
                'joint_arm_l1': q_arml,
                'joint_arm_l2': q_arml,
                'joint_arm_l3': q_arml,
                'joint_wrist_yaw': q_yaw,
                'joint_gripper_finger_left': 0.0,
                'joint_gripper_finger_right': 0.0,
                'joint_head_pan': q_pan,
                'joint_head_tilt': q_tilt
            }
        elif tool == 'tool_stretch_dex_wrist' or tool == 'eoa_wrist_dw3_tool_sg3':
            q_lift = bound_range('joint_lift', self.body.lift.status['pos'])
            q_arml = bound_range('joint_arm_l0', self.body.arm.status['pos'] / 4.0)
            q_yaw = bound_range('joint_wrist_yaw', self.body.end_of_arm.status['wrist_yaw']['pos'])
            q_pitch = bound_range('joint_wrist_pitch', self.body.end_of_arm.status['wrist_pitch']['pos'])
            q_roll = bound_range('joint_wrist_roll', self.body.end_of_arm.status['wrist_roll']['pos'])
            q_pan = bound_range('joint_head_pan', self.body.head.status['head_pan']['pos'])
            q_tilt = bound_range('joint_head_tilt', self.body.head.status['head_tilt']['pos'])
            return {
                'joint_left_wheel': 0.0,
                'joint_right_wheel': 0.0,
                'joint_lift': q_lift,
                'joint_arm_l0': q_arml,
                'joint_arm_l1': q_arml,
                'joint_arm_l2': q_arml,
                'joint_arm_l3': q_arml,
                'joint_wrist_yaw': q_yaw,
                'joint_wrist_pitch': q_pitch,
                'joint_wrist_roll': q_roll,
                'joint_gripper_finger_left': 0.0,
                'joint_gripper_finger_right': 0.0,
                'joint_head_pan': q_pan,
                'joint_head_tilt': q_tilt
            }
        else:
            raise ValueError(f"Cannot get configuration of {tool} tool")

    def get_transform(self, from_frame, to_frame):
        q_curr = self._get_current_configuration()
        fk_curr = self.urdf.link_fk(cfg=q_curr)

        # get frames w.r.t. base link
        baselink_to_fromframe = None
        baselink_to_toframe = None
        for l in self.urdf.links:
            if l.name == from_frame:
                baselink_to_fromframe = fk_curr[l]
            if l.name == to_frame:
                baselink_to_toframe = fk_curr[l]
        if from_frame == 'map':
            baselink_to_fromframe = np.eye(4)
        if to_frame == 'map':
            baselink_to_toframe = np.eye(4)

        # calculate to_frame w.r.t from_frame
        fromframe_to_toframe = np.linalg.inv(baselink_to_fromframe).dot(baselink_to_toframe)
        return fromframe_to_toframe
    

###

    # def joint_state_callback(self, data):
    #     """Callback function for joint state updates from the robot
        
    #     Args:
    #         data: Robot state data from robot SDK callback
    #     """
    #     if data and len(data) > 7:  # Make sure we have joint data
    #         # xArm SDK reports joint angles in the data
    #         joint_angles = data[:self.config.dof]  # xArm Lite6 has 6 joints
    #         with self.joint_state_lock:
    #             self.current_joints = np.array(joint_angles)


    # def _parse_static_camera_tf(self, static_camera_tf):
    #     """Parse the static camera transform into position and rotation components
        
    #     Args:
    #         static_camera_tf: SE(3) transform as 4x4 matrix or (translation, rotation) tuple
    #     """
    #     try:
    #         if isinstance(static_camera_tf, np.ndarray) and static_camera_tf.shape == (4, 4):
    #             # 4x4 homogeneous transformation matrix
    #             self.static_camera_position = static_camera_tf[:3, 3]
    #             self.static_camera_rotation = Rotation.from_matrix(static_camera_tf[:3, :3])
    #             print(f"Parsed 4x4 static camera transform: pos={self.static_camera_position}, rot={self.static_camera_rotation.as_quat()}")
                
    #         else:
    #             raise ValueError(f"static_camera_tf must be 4x4 matrix or (translation, rotation) tuple, got {type(static_camera_tf)}")
                
    #         # Store the original transform for reference
    #         self.static_camera_tf = static_camera_tf
    #     except Exception as e:
    #         print(f"Error parsing static camera transform: {e}")
    #         print("Static camera transform will be ignored, falling back to dynamic transform")
    #         self.static_camera_tf = None
    #         self.static_camera_position = None
    #         self.static_camera_rotation = None


    
    # def _configure_initial_sensitivity(self) -> bool:
    #     """
    #     Configure initial robot sensitivity settings for normal operations
        
    #     Returns:
    #         bool: True if configuration successful, False otherwise
    #     """
    #     try:
    #         if self.arm is None:
    #             print("Warning: Robot not connected, cannot configure initial sensitivity")
    #             return False
                
    #         with self.arm_lock:
    #             if hasattr(self.arm, "set_teach_sensitivity"):
    #                 teach_result = self.arm.set_teach_sensitivity(self.default_teach_sensitivity, wait=True)
    #                 if teach_result != 0:
    #                     print(f"Warning: Failed to set initial teach sensitivity (code: {teach_result})")
    #             print(f"✓ Configured initial sensitivity: collision={self.default_collision_sensitivity}, teach={self.default_teach_sensitivity}")
    #             return True
                
    #     except Exception as e:
    #         print(f"Error configuring initial sensitivity: {e}")
    #         return False

    # def init_curobo(self):
    #     """Initialize cuRobo motion generator"""
    #     try:
    #         # Get the robot configs path from cuRobo
    #         robot_configs_path = get_robot_configs_path()
    #         print(f"Robot configs path: {robot_configs_path}")
            
    #         # Create path for XArm Lite6 config
    #         xarm_config_dir = os.path.join(robot_configs_path, f"xarm" + "_lite6" if self.config.is_lite6 else f"{self.config.dof}")
    #         os.makedirs(xarm_config_dir, exist_ok=True)
    #         filename = "xarm_lite6" if self.config.is_lite6 else f"{self.config.robot_type}{self.config.dof}"
    #         # Define robot configuration file path
    #         robot_config_path = os.path.join(robot_configs_path, filename + ".yml")
            
    #         # Basic world configuration with just a table below the robot
    #         world_config = {
    #             "cuboid": {
    #                 "table": {
    #                     "dims": [1.0, 1.0, 0.05],  # x, y, z
    #                     "pose": [0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0],  # x, y, z, qw, qx, qy, qz
    #                 }
    #             }
    #         }
            
    #         # Set up motion generator configuration with optimized parameters
    #         print(f"Loading motion generator config from: {robot_config_path}")
    #         motion_gen_config = MotionGenConfig.load_from_robot_config(
    #             robot_cfg=robot_config_path,
    #             world_model=world_config,
    #             tensor_args=self.tensor_args,
    #             # Planning parameters - optimized for speed
    #             interpolation_dt=0.015,         # Increased from 0.01 for faster interpolation
    #             interpolation_steps=2500,      # Halved from 5000 for speed
    #             collision_checker_type=CollisionCheckerType.PRIMITIVE,
                
    #             # Optimization parameters - balanced for speed and reliability
    #             num_ik_seeds=12,        # Moderate reduction from 16 (was 32)
    #             num_graph_seeds=2,      # Restored from 1 (was 4)  
    #             num_trajopt_seeds=4,    # Restored from 2 (was 6)
                
    #             # Trajectory optimization parameters - faster convergence
    #             trajopt_tsteps=24,      # Reduced from default 32
    #             trajopt_dt=0.25,        # Reduced from 0.5 for faster optimization
    #             js_trajopt_dt=0.25,     # Reduced from 0.5 for faster optimization
                
    #             # Collision parameters - relaxed for speed
    #             collision_activation_distance=0.04,    # Slightly increased from 0.03
    #             collision_max_outside_distance=0.3,    # Reduced from 0.5
                
    #             # Quality parameters - restore some quality for reliability
    #             evaluate_interpolated_trajectory=True,  # Re-enabled for trajectory quality
    #             position_threshold=0.01,    # Restored to original for accuracy
    #             rotation_threshold=0.1,     # Restored to original
    #             cspace_threshold=0.05,      # Restored to original
                
    #             # Performance parameters
    #             use_cuda_graph=True,        # ENABLED: Cache CUDA graphs for massive speedup
    #             store_debug_in_result=False, # Keep disabled for speed
    #             minimum_trajectory_dt=0.015, # Must be >= interpolation_dt (0.015)
    #             finetune_dt_scale=0.85,      # Restored to original
                
    #             # Smoothness parameters - restore for quality
    #             minimize_jerk=True,          # Re-enabled for smooth trajectories
    #             filter_robot_command=True    # Re-enabled for command filtering
    #         )
            
    #         # Store robot config for IK solver - use the imported RobotConfig, not our class
    #         # Load the robot config from the YAML file for IK solver
    #         from curobo.types.robot import RobotConfig as CuroboRobotConfig
            
    #         self.config.robot_cfg = CuroboRobotConfig.from_dict(
    #             load_yaml(robot_config_path)["robot_cfg"],
    #             self.tensor_args
    #         )
            
    #         # Create motion generator
    #         print("Initializing cuRobo motion generator...")
    #         motion_gen = MotionGen(motion_gen_config)

    #         # Warmup the motion generator with common problem sizes
    #         # This pre-compiles CUDA graphs for faster subsequent planning
    #         print("Warming up motion generator (this may take 10-20 seconds)...")
    #         motion_gen.warmup(warmup_js_trajopt=True)  # Warm up joint space trajopt

    #         # Additional warmup for common batch sizes if needed
    #         # Uncomment if you see "warming up solver" messages during operation:
    #         # print("Performing extended warmup for batch planning...")
    #         # motion_gen.warmup(n_goalset=4, warmup_js_trajopt=True)

    #         print("cuRobo motion generator initialized successfully")
    #         return motion_gen
    #     except Exception as e:
    #         print(f"Failed to initialize cuRobo motion generator: {str(e)}")
    #         import traceback
    #         print(traceback.format_exc())
    #         return None
