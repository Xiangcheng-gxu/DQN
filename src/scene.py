"""MuJoCo scene creation, workspace definitions, and tabletop object generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from gripper import Gripper2F85
from robot import UR5eRobot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "models" / "scene.xml"

DEFAULT_JOINT_ANGLES_DEG = np.asarray((0.0, -90.0, 90.0, -90.0, -90.0, 0.0), dtype=float)
DEFAULT_GRIPPER_COMMAND = Gripper2F85.COMMAND_CLOSED

# [x, y, z] bounds in metres. Z values are relative to the tabletop reference.
WORKSPACE_LIMITS = np.asarray(((-0.75, -0.25), (-0.25, 0.25), (0.00, 0.40)), dtype=float)
WORKSPACE_REFERENCE_Z_M = 0.754
WORKSPACE_COLOR = (237, 234, 225)

OBSTACLE_COLOR = (100, 150, 255)  # Light blue
TARGET_COLOR = (255, 0, 0)
OBJECT_SIDE_RANGE_M = (0.03, 0.05)
OBJECT_HEIGHT_CHOICES_M = (0.01, 0.02, 0.03, 0.04)
# Target: 10 cm x 10 cm centred in the workspace.
TARGET_SPAWN_LIMITS = np.asarray(((-0.55, -0.45), (-0.05, 0.05)), dtype=float)
# Obstacles: the inner 40 cm x 40 cm area, excluding the target spawn area.
OBSTACLE_SPAWN_LIMITS = np.asarray(((-0.65, -0.35), (-0.15, 0.15)), dtype=float)


@dataclass(frozen=True)
class SceneObject:
    """A fixed square-prism object resting on the workspace reference plane."""

    kind: str
    color: tuple[int, int, int]
    position_xy: np.ndarray
    side_m: float
    height_m: float
    dynamic: bool = False

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return (*(channel / 255.0 for channel in self.color), 1.0)


@dataclass
class FixedScene:
    """A loaded scene and its persistent joint and gripper control targets."""

    model: mujoco.MjModel
    data: mujoco.MjData
    robot: UR5eRobot
    gripper: Gripper2F85
    joint_target_rad: np.ndarray
    gripper_target: float
    objects: tuple[SceneObject, ...]
    clutter_displacements_m: tuple[float, ...] = ()

    def step(self) -> None:
        """Reapply fixed robot controls and advance one physics step."""
        for actuator_id, target in zip(self.robot._actuator_ids, self.joint_target_rad):
            self.data.ctrl[actuator_id] = target
        self.data.ctrl[self.gripper.actuator_id] = self.gripper_target
        mujoco.mj_step(self.model, self.data)


def create_object(
    color: Sequence[int],
    *,
    rng: np.random.Generator | None = None,
    occupied: Sequence[SceneObject] = (),
    dynamic: bool = False,
) -> SceneObject:
    """Create one valid target or obstacle specification from its RGB colour.

    Red creates a target in the central 10 cm x 10 cm area. The obstacle colour
    creates an obstacle in the surrounding inner 40 cm x 40 cm area, excluding
    that target area. Every object has a square 3--5 cm base and a 1--4 cm height.
    """
    rgb = _validate_object_color(color)
    generator = np.random.default_rng() if rng is None else rng
    side_m = float(generator.uniform(*OBJECT_SIDE_RANGE_M))
    height_m = float(generator.choice(OBJECT_HEIGHT_CHOICES_M))
    kind = "target" if rgb == TARGET_COLOR else "obstacle"
    allowed = TARGET_SPAWN_LIMITS if kind == "target" else OBSTACLE_SPAWN_LIMITS
    excluded = () if kind == "target" else (TARGET_SPAWN_LIMITS,)
    position_xy = _sample_position(allowed, side_m, excluded, occupied, generator)
    return SceneObject(kind, rgb, position_xy, side_m, height_m, dynamic)


def create_fixed_scene(
    joint_angles_deg: np.ndarray = DEFAULT_JOINT_ANGLES_DEG,
    gripper_command: int = DEFAULT_GRIPPER_COMMAND,
    objects: Sequence[SceneObject] = (),
) -> FixedScene:
    """Load the project scene, add fixed objects, and initialize a fixed robot pose."""
    object_tuple = tuple(objects)
    model = _load_model(object_tuple)
    data = mujoco.MjData(model)
    robot = UR5eRobot(model)
    gripper = Gripper2F85(model)

    target_deg = np.asarray(joint_angles_deg, dtype=float)
    robot.set_joint_angles_deg(data, target_deg)
    data.qvel[:] = 0.0
    gripper.set_opening(data, gripper_command)
    mujoco.mj_forward(model, data)
    return FixedScene(model, data, robot, gripper, np.deg2rad(target_deg), gripper.mujoco_control(gripper_command), object_tuple)



def create_clutter_environment(
    object_count: int | tuple[int, int],
    *,
    rng: np.random.Generator | None = None,
    clustering_steps: int = 100,
    attraction_force_n: float = 1.0,
) -> FixedScene:
    """Create an initial clutter scene; call :func:`advance_clutter` to animate it.

    An integer specifies the obstacle count. An inclusive two-item tuple samples
    the obstacle count. The returned scene is intentionally unclustered so a
    viewer can display it before the attraction phase begins.
    """
    obstacle_count = _resolve_obstacle_count(object_count, rng)
    if clustering_steps < 1 or attraction_force_n <= 0:
        raise ValueError("clustering_steps must be positive and attraction_force_n must be positive")
    generator = np.random.default_rng() if rng is None else rng
    target = create_object(TARGET_COLOR, rng=generator, dynamic=True)
    objects: list[SceneObject] = [target]
    for _ in range(obstacle_count):
        objects.append(create_object(OBSTACLE_COLOR, rng=generator, occupied=objects, dynamic=True))
    scene = create_fixed_scene(objects=objects)
    _prepare_clutter(scene, clustering_steps, attraction_force_n)
    return scene


def advance_clutter(scene: FixedScene) -> bool:
    """Advance exactly one horizontal attraction step; return True when frozen."""
    if getattr(scene, "_clutter_finished", False):
        return True
    target_id = scene._clutter_target_id
    target_qpos_adr = scene._clutter_target_qpos_adr
    target_dof_adr = scene._clutter_target_dof_adr
    scene.data.qpos[target_qpos_adr:target_qpos_adr + 2] = scene._clutter_target_qpos
    scene.data.qvel[target_dof_adr:target_dof_adr + 2] = 0.0
    mujoco.mj_forward(scene.model, scene.data)
    target_xy = scene.data.xpos[target_id, :2]
    for body_id, dof_adr, obstacle in zip(scene._clutter_obstacle_ids, scene._clutter_obstacle_dof_addresses, scene.objects[1:]):
        obstacle_xy = scene.data.xpos[body_id, :2]
        offset_xy = target_xy - obstacle_xy
        distance = float(np.linalg.norm(offset_xy))
        if distance > 1e-6:
            # Continuous XY attraction; collision contact is the only stopping mechanism.
            scene.data.qfrc_applied[dof_adr:dof_adr + 2] = scene._clutter_force_n * offset_xy / distance
    scene.step()
    scene.data.qfrc_applied[:] = 0.0
    scene._clutter_remaining -= 1
    if scene._clutter_remaining > 0:
        return False
    _finish_clutter(scene)
    return True


def _prepare_clutter(scene: FixedScene, steps: int, force_n: float) -> None:
    target_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "target_0")
    target_joint_id = scene.model.body_jntadr[target_id]
    scene._clutter_target_id = target_id
    scene._clutter_target_qpos_adr = scene.model.jnt_qposadr[target_joint_id]
    scene._clutter_target_dof_adr = scene.model.jnt_dofadr[target_joint_id]
    scene._clutter_target_qpos = scene.data.qpos[scene._clutter_target_qpos_adr:scene._clutter_target_qpos_adr + 2].copy()
    scene._clutter_obstacle_ids = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{index}") for index in range(1, len(scene.objects))]
    scene._clutter_obstacle_dof_addresses = [scene.model.jnt_dofadr[scene.model.body_jntadr[body_id]] for body_id in scene._clutter_obstacle_ids]
    scene._clutter_initial_positions = [scene.data.xpos[body_id, :2].copy() for body_id in scene._clutter_obstacle_ids]
    scene._clutter_remaining = steps
    scene._clutter_force_n = force_n
    scene._clutter_finished = False


def _finish_clutter(scene: FixedScene) -> None:
    """End attraction without constraining objects: clear forces and velocity only."""
    # The target is restored to its original XY position, then all objects are
    # left as normal movable planar bodies with no residual velocity or force.
    scene.data.qpos[scene._clutter_target_qpos_adr:scene._clutter_target_qpos_adr + 2] = scene._clutter_target_qpos
    scene.data.qvel[:] = 0.0
    scene.data.qfrc_applied[:] = 0.0
    scene.data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(scene.model, scene.data)
    scene.clutter_displacements_m = tuple(float(np.linalg.norm(scene.data.xpos[body_id, :2] - initial)) for body_id, initial in zip(scene._clutter_obstacle_ids, scene._clutter_initial_positions))
    if any(displacement < 0.005 for displacement in scene.clutter_displacements_m):
        raise RuntimeError("Clutter generation failed: every obstacle must move at least 5 mm")
    scene._clutter_finished = True

def _resolve_obstacle_count(object_count: int | tuple[int, int], rng: np.random.Generator | None) -> int:
    if isinstance(object_count, (int, np.integer)):
        count = int(object_count)
        if count < 0:
            raise ValueError("object_count must be non-negative")
        return count
    if len(object_count) != 2:
        raise ValueError("object_count interval must contain exactly two integers")
    low, high = (int(value) for value in object_count)
    if low < 0 or low > high:
        raise ValueError("object_count interval must satisfy 0 <= low <= high")
    generator = np.random.default_rng() if rng is None else rng
    return int(generator.integers(low, high + 1))

def _load_model(objects: Sequence[SceneObject]) -> mujoco.MjModel:
    if not objects:
        return mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    tree = ET.parse(SCENE_PATH)
    root = tree.getroot()
    dynamic_scene = any(item.dynamic for item in objects)
    temporary_paths: list[Path] = []
    try:
        if dynamic_scene:
            # The included robot model has a robot-only keyframe. Free joints
            # increase nq, so compile a temporary copy without that keyframe.
            include = root.find("include")
            if include is None or not include.get("file"):
                raise ValueError("Scene XML does not include the robot model")
            robot_path = SCENE_PATH.parent / include.get("file")
            robot_tree = ET.parse(robot_path)
            keyframe = robot_tree.getroot().find("keyframe")
            if keyframe is not None:
                robot_tree.getroot().remove(keyframe)
            with NamedTemporaryFile(mode="wb", suffix=".xml", dir=SCENE_PATH.parent, delete=False) as temporary_robot:
                temporary_robot_path = Path(temporary_robot.name)
            temporary_paths.append(temporary_robot_path)
            robot_tree.write(temporary_robot_path, encoding="utf-8", xml_declaration=True)
            include.set("file", temporary_robot_path.name)
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        obstacle_rgba = " ".join(f"{channel / 255.0:.6f}" for channel in OBSTACLE_COLOR) + " 1"
        ET.SubElement(asset, "material", name="generated_obstacle_material", rgba=obstacle_rgba, specular="0", shininess="0", reflectance="0")
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Scene XML does not contain worldbody")
        for index, item in enumerate(objects):
            body = ET.SubElement(worldbody, "body", name=f"{item.kind}_{index}", pos=f"{item.position_xy[0]:.6f} {item.position_xy[1]:.6f} {WORKSPACE_REFERENCE_Z_M:.6f}")
            if item.dynamic:
                # Planar joints prevent gravity/collisions from lifting or tipping objects.
                ET.SubElement(body, "joint", name=f"{item.kind}_slide_x_{index}", type="slide", axis="1 0 0", frictionloss="0.1", damping="0.5")
                ET.SubElement(body, "joint", name=f"{item.kind}_slide_y_{index}", type="slide", axis="0 1 0", frictionloss="0.1", damping="0.5")
            attributes = {
                "name": f"{item.kind}_geom_{index}", "type": "box",
                "pos": f"0 0 {item.height_m / 2:.6f}",
                "size": f"{item.side_m / 2:.6f} {item.side_m / 2:.6f} {item.height_m / 2:.6f}",
                "rgba": " ".join(f"{value:.6f}" for value in item.rgba),
                "mass": "0.05", "contype": "1", "conaffinity": "1",
            }
            if item.kind == "obstacle":
                attributes["material"] = "generated_obstacle_material"
            if item.dynamic:
                # Dense clutter stays about 1 cm apart instead of interpenetrating.
                # Near-hard contact: zero margin and the smallest stable time constant.
                attributes["margin"] = "0"
                attributes["solref"] = "0.004 1"
                attributes["solimp"] = "0.9999 0.99999 0.00001 0.5 2"
            ET.SubElement(body, "geom", **attributes)
        with NamedTemporaryFile(mode="wb", suffix=".xml", dir=SCENE_PATH.parent, delete=False) as temporary_scene:
            temporary_scene_path = Path(temporary_scene.name)
        temporary_paths.append(temporary_scene_path)
        tree.write(temporary_scene_path, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(str(temporary_scene_path))
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)

def _validate_object_color(color: Sequence[int]) -> tuple[int, int, int]:
    rgb = tuple(color)
    if rgb not in (OBSTACLE_COLOR, TARGET_COLOR):
        raise ValueError(f"color must be OBSTACLE_COLOR {OBSTACLE_COLOR} or TARGET_COLOR {TARGET_COLOR}")
    return rgb


def _sample_position(allowed: np.ndarray, side_m: float, excluded: Sequence[np.ndarray], occupied: Sequence[SceneObject], rng: np.random.Generator) -> np.ndarray:
    half_side = side_m / 2
    low = allowed[:, 0] + half_side
    high = allowed[:, 1] - half_side
    if np.any(low > high):
        raise ValueError("Object does not fit inside its spawn region")
    for _ in range(1_000):
        position = rng.uniform(low, high)
        if any(_box_overlaps_limits(position, side_m, limits) for limits in excluded):
            continue
        if any(_boxes_overlap(position, side_m, other.position_xy, other.side_m) for other in occupied):
            continue
        return position
    raise RuntimeError("Could not sample a non-overlapping object position")



def _box_overlaps_limits(position: np.ndarray, side_m: float, limits: np.ndarray) -> bool:
    half_side = side_m / 2
    return bool(np.all(position + half_side > limits[:, 0]) and np.all(position - half_side < limits[:, 1]))


def _boxes_overlap(first_position: np.ndarray, first_side: float, second_position: np.ndarray, second_side: float) -> bool:
    # Require at least 5 cm edge-to-edge separation in the XY plane.
    clearance = (first_side + second_side) / 2 + 0.05
    return bool(abs(first_position[0] - second_position[0]) < clearance and abs(first_position[1] - second_position[1]) < clearance)