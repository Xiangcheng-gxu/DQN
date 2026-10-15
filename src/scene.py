"""场景定义与随机生成：工作空间、物体尺寸、初始摆放和水平聚拢。

学生修改工作区/物体颜色尺寸时先看本文件顶部常量。
注意这里的Z基准0.754m是世界坐标；动作里“离桌面0m”对应世界Z=0.754m。
"""

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

# X/Y是世界坐标范围，Z是相对桌面的高度范围；单位都是米。
WORKSPACE_LIMITS = np.asarray(((-0.75, -0.25), (-0.25, 0.25), (0.00, 0.40)), dtype=float)
WORKSPACE_REFERENCE_Z_M = 0.754
WORKSPACE_COLOR = (237, 234, 225)

OBSTACLE_COLOR = (100, 150, 255)  # 浅蓝色障碍物，RGB整数0到255。
TARGET_COLOR = (255, 0, 0)
OBJECT_SIDE_RANGE_M = (0.03, 0.05)
OBJECT_HEIGHT_CHOICES_M = (0.01, 0.02, 0.03, 0.04)
# 目标初始生成区：以工作区中心(-0.5,0)为中心的10×10cm区域。
TARGET_SPAWN_LIMITS = np.asarray(((-0.55, -0.45), (-0.05, 0.05)), dtype=float)
# 障碍物初始生成区：50cm工作区四周各内缩10cm，剩30×30cm，再排除目标区。
# 这是聚拢前的摆放范围；物体全底面必须在范围内，不只要求中心在里面。
OBSTACLE_SPAWN_LIMITS = np.asarray(((-0.65, -0.35), (-0.15, 0.15)), dtype=float)


@dataclass(frozen=True)
class SceneObject:
    """一个方柱物体的生成说明；不是仿真中每一步更新的实时位置。"""

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
    """加载后的模型和数据；Fixed指默认机械臂控制目标，不代表杂乱物体固定。"""

    model: mujoco.MjModel
    data: mujoco.MjData
    robot: UR5eRobot
    gripper: Gripper2F85
    joint_target_rad: np.ndarray
    gripper_target: float
    objects: tuple[SceneObject, ...]
    clutter_displacements_m: tuple[float, ...] = ()

    def step(self) -> None:
        """聚拢期间保持机械臂默认姿态及夹爪闭合，并计算一个物理步。"""
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
    """按颜色创建一个物体说明：红色为目标、浅蓝色为障碍物。

    随机正方形边长3到5cm，高度从1/2/3/4cm等概率选择。
    occupied是已有物体，初始摆放需保持至少5cm间隔；聚拢后不再保持该间隔。
    dynamic=True时添加XY滑动自由度，物体可推但不能升起或倾倒。
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
    """加载XML并加入物体，设置六关节默认姿态、闭合指令和零初速度。"""
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
    """创建聚拢前的场景，再由环境反复调用advance_clutter完成聚拢。

    整数表示障碍物数，不包含另加的1个目标；(最小数,最大数)包含两个端点。
    每次朝目标XY施加attraction_force_n牛的水平力，持续clustering_steps步。
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
    """聚拢推进一个物理步；结束返回True。聚拢后物体仍能移动，不固定它们。"""
    if getattr(scene, "_clutter_finished", False):
        return True
    # 聚拢阶段临时保持目标XY不动，让其余物体向它靠近。
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
            # 只给XY方向的力，不朝目标三维中心施力，不添加Z分量。
            scene.data.qfrc_applied[dof_adr:dof_adr + 2] = scene._clutter_force_n * offset_xy / distance
    scene.step()
    scene.data.qfrc_applied[:] = 0.0
    scene._clutter_remaining -= 1
    if scene._clutter_remaining > 0:
        return False
    _finish_clutter(scene)
    return True


def _prepare_clutter(scene: FixedScene, steps: int, force_n: float) -> None:
    """缓存物体地址与初始位置；聚拢完成后还要验证障碍物确实移动过。"""
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
    """恢复目标位置、清空残余外力和速度，取消临时保持，不锁住物体。"""
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
    """临时拼接随机物体XML并编译；只删本函数生成的临时XML，不改原模型。"""
    if not objects:
        return mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    tree = ET.parse(SCENE_PATH)
    root = tree.getroot()
    dynamic_scene = any(item.dynamic for item in objects)
    temporary_paths: list[Path] = []
    try:
        if dynamic_scene:
            # 原机器人关键帧只包含机器人关节；增加物体滑动关节后维数不再相同。
            # 临时副本移除旧关键帧，原XML不变，机械臂默认角由上层显式初始化。
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
                # 物体只有XY滑动自由度：教学简化，不模拟飞起、翻倒或绕Z旋转。
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
                # 零碰撞余量和较硬接触参数；数值求解仍可能有很小的穿透误差。
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
    """最多抽样1000次，要求整个底面在允许区内且与已有物体保持间隔。"""
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
    # 聚拢前要求XY底面边缘至少相隔5cm，不是中心间隔5cm。
    clearance = (first_side + second_side) / 2 + 0.05
    return bool(abs(first_position[0] - second_position[0]) < clearance and abs(first_position[1] - second_position[1]) < clearance)
