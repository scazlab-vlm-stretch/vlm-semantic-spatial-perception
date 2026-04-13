# AMP Primitive Catalog

Ground-truth reference for every primitive in the AMP library. Use these **exact names** in skill plans — the executor validates names against this catalog and rejects unknown primitives.

Units: positions in **meters**, angles in **degrees**. Pixel coordinates use the **0-1000 normalized** convention (top-left origin, [y, x] order).

The executor back-projects `target_pixel_yx` into 3D using the snapshot depth map and intrinsics. Supply pixel coordinates when a snapshot is available; supply `point_label` (object_id) as fallback when no snapshot exists.

---

## move_gripper_to_pose

**What it does**: Moves the **gripper end-effector** to a target pose. Use this for grasping and placing — NOT for moving the robot base (that is `navigate_to_pose`, added later).

**Parameters** (all optional — provide at least one position source):

| Name | Type | Description |
|------|------|-------------|
| `target_pixel_yx` | `[y, x]` normalized 0-1000 | Pixel for the grasp/approach point. Executor back-projects to 3D. |
| `point_label` | string | Object ID fallback when no depth snapshot is available. Resolved from registry. |
| `preset_orientation` | `"top_down"` \| `"side"` | `top_down`: end-effector pointing toward ground. `side`: pointing forward, parallel to ground. Default: `"top_down"`. |
| `is_place` | boolean | Set `true` when placing — adds a small z-clearance above the surface. Default: `false`. |

---

## push_pull

**What it does**: Moves the end-effector along or into a surface (perpendicular push, lateral slide, button press, or articulated door/drawer open/close).

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `surface_label` | string | **required** | Object ID or surface label to interact with. |
| `force_direction` | `"perpendicular"` \| `"parallel"` | optional | `perpendicular`: push into the surface normal. `parallel`: slide across the surface. Default: `"perpendicular"`. |
| `is_button` | boolean | optional | Momentary press: push then immediately retract. Default: `false`. |
| `has_pivot` | boolean | optional | `true` for revolute joints (doors, drawers). Default: `false`. |
| `hinge_location` | string | optional | Surface boundary label for the hinge axis (required when `has_pivot=true`). |

---

## open_gripper

**What it does**: Opens the gripper and releases any held object.

**Parameters**: None required.

| Name | Type | Description |
|------|------|-------------|
| `wait` | boolean | Wait for completion before returning. Default: `true`. |
| `timeout` | float (s) | Max wait time. Default: `5.0`. |

---

## close_gripper

**What it does**: Closes the gripper to acquire a grasp.

**Parameters**: None required.

| Name | Type | Description |
|------|------|-------------|
| `wait` | boolean | Wait for completion before returning. Default: `true`. |
| `timeout` | float (s) | Max wait time. Default: `5.0`. |

---

## retract_gripper

**What it does**: Returns the arm to its home/standby joint configuration.

**Parameters**: None required.

| Name | Type | Description |
|------|------|-------------|
| `distance` | float (m) | Retract distance (implementation-defined). Default: `0.05`. |
| `speed_factor` | float | Speed multiplier 0.1–2.0. Default: `1.0`. |

---

## twist

**What it does**: Rotates the wrist joint by `rotation_angle_deg` degrees and returns. Useful for turning knobs, dials, or wiping surfaces.

**Parameters**: None required.

| Name | Type | Description |
|------|------|-------------|
| `direction` | `"clockwise"` \| `"counterclockwise"` | Rotation direction. Default: `"clockwise"`. |
| `rotation_angle_deg` | float (deg) | Angle to rotate. Must be > 0. Default: `90`. |
| `speed_factor` | float | Speed multiplier 0.1–2.0. Default: `1.0`. |
| `timeout` | float (s) | Max execution time. Default: `5.0`. |

---

## Quick recipes

**Pick object:**
```
move_gripper_to_pose  (target_pixel_yx or point_label, preset_orientation="top_down")
close_gripper
retract_gripper
```

**Place object:**
```
move_gripper_to_pose  (target_pixel_yx or point_label for target surface/object, is_place=true)
open_gripper
retract_gripper
```

**Press button:**
```
move_gripper_to_pose  (approach the button face)
push_pull             (surface_label=<button_id>, is_button=true)
retract_gripper
```
