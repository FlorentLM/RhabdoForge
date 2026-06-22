from pathlib import Path
from typing import Union, Optional, Sequence, Dict
import numpy as np
from numpy.typing import ArrayLike
from pyglm import glm

from .world_utils import WORLD_RIGHT, WORLD_UP, WORLD_FORWARD
from insectvision.geometry.linalg import tangent_frames


class TransformMixin:
    """
    Mixin class for common transformation methods.
    """

    _transform: glm.mat4
    _transform_rev: int = 0

    def _euler_quat(self, yaw, pitch, roll, degrees=True):
        """Build a quaternion from YXZ Euler angles."""
        if degrees:
            yaw, pitch, roll = glm.radians(yaw), glm.radians(pitch), glm.radians(roll)
        return (glm.angleAxis(yaw, WORLD_UP)
                * glm.angleAxis(pitch, WORLD_RIGHT)
                * glm.angleAxis(roll, WORLD_FORWARD))

    @property
    def transform(self) -> glm.mat4:
        if not hasattr(self, '_transform'):
            self._transform = glm.mat4(1.0)
            self._transform_rev = 0
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value if isinstance(value, glm.mat4) else glm.mat4(value)
        self.touch()

    def touch(self) -> None:
        """Mark the object's transform as changed."""
        self._transform_rev += 1

    # Position

    @property
    def position(self) -> glm.vec3:
        return glm.vec3(self.transform[3])

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self.transform[3] = glm.vec4(glm.vec3(value), 1.0)
        self.touch()

    pos = position

    @property
    def scale(self):
        """Per-axis scale (column lengths of the linear part)."""
        return glm.vec3(
            glm.length(glm.vec3(self.transform[0])),
            glm.length(glm.vec3(self.transform[1])),
            glm.length(glm.vec3(self.transform[2])),
        )

    @property
    def rotation(self):
        """Orientation as a unit quaternion."""
        s = self.scale
        R = glm.mat4(
            self.transform[0] / s.x,
            self.transform[1] / s.y,
            self.transform[2] / s.z,
            glm.vec4(0, 0, 0, 1)
        )
        return glm.quat_cast(R)

    @rotation.setter
    def rotation(self, q):
        """Replace orientation."""
        if not isinstance(q, glm.quat):
            q = glm.quat(q)
        pos = self.position
        s = self.scale
        if not all(np.isfinite(v) and v > 1e-9 for v in (s.x, s.y, s.z)):
            s = glm.vec3(1.0)
        M = glm.mat4_cast(q)
        M[0] = glm.vec4(glm.vec3(M[0]) * s.x, 0.0)
        M[1] = glm.vec4(glm.vec3(M[1]) * s.y, 0.0)
        M[2] = glm.vec4(glm.vec3(M[2]) * s.z, 0.0)
        M[3] = glm.vec4(pos, 1.0)
        self.transform = M

    # Directional vectors

    @property
    def forward(self) -> glm.vec3:
        return glm.normalize(glm.vec3(self.transform * glm.vec4(WORLD_FORWARD, 0.0)))

    @property
    def backward(self) -> glm.vec3:
        return -self.forward

    @property
    def right(self) -> glm.vec3:
        return glm.normalize(glm.vec3(self.transform * glm.vec4(WORLD_RIGHT, 0.0)))

    @property
    def left(self) -> glm.vec3:
        return -self.right

    @property
    def up(self) -> glm.vec3:
        return glm.normalize(glm.vec3(self.transform * glm.vec4(WORLD_UP, 0.0)))

    @property
    def down(self) -> glm.vec3:
        return -self.up

    # Euler accessors

    @property
    def yaw(self):
        c2 = glm.normalize(glm.vec3(self.transform[2]))
        return glm.degrees(glm.atan2(c2.x, c2.z))

    @property
    def pitch(self):
        c2 = glm.normalize(glm.vec3(self.transform[2]))
        return glm.degrees(glm.asin(glm.clamp(-c2.y, -1.0, 1.0)))

    @property
    def roll(self):
        c0 = glm.normalize(glm.vec3(self.transform[0]))
        c1 = glm.normalize(glm.vec3(self.transform[1]))
        return glm.degrees(glm.atan2(c0.y, c1.y))

    @yaw.setter
    def yaw(self, v):
        self.rotation = self._euler_quat(v, self.pitch, self.roll)

    @pitch.setter
    def pitch(self, v):
        self.rotation = self._euler_quat(self.yaw, v, self.roll)

    @roll.setter
    def roll(self, v):
        self.rotation = self._euler_quat(self.yaw, self.pitch, v)

    def set_rotation(self, yaw=0.0, pitch=0.0, roll=0.0, degrees=True):
        """Replace orientation from Euler angles, all at once."""
        self.rotation = self._euler_quat(yaw, pitch, roll, degrees)
        return self

    # Aliases
    pan = horizontal_angle = yaw
    tilt = vertical_angle = pitch
    bank = roll

    # Transformations

    def translate(self, vec: Union[glm.vec3, ArrayLike]):
        """World-space translation."""
        self.transform[3] = glm.vec4(self.position + glm.vec3(vec), 1.0)
        self.touch()
        return self

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        """Rotate around a world-space axis passing through the object's position."""
        axis_map = {
            'x': WORLD_RIGHT, 'y': WORLD_UP, 'z': WORLD_FORWARD,
            'right': WORLD_RIGHT, 'left': -WORLD_RIGHT,
            'up': WORLD_UP, 'down': -WORLD_UP,
            'forward': WORLD_FORWARD, 'backward': -WORLD_FORWARD,
        }

        if isinstance(axis, str):
            rotation_axis = axis_map.get(axis.lower())
            if rotation_axis is None:
                raise ValueError(f"Unknown axis: '{axis}'.")
        else:
            rotation_axis = glm.normalize(glm.vec3(axis))

        a = glm.radians(angle) if degrees else angle

        pos = self.position
        self.transform[3] = glm.vec4(0, 0, 0, 1)

        R = glm.mat4_cast(glm.angleAxis(a, rotation_axis))
        self.transform = R * self.transform
        self.transform[3] = glm.vec4(pos, 1.0)
        self.touch()
        return self

    def rotate(self, yaw=0.0, pitch=0.0, roll=0.0, degrees=True):
        """Yaw around world up, pitch around local right, roll around local forward."""
        if pitch:
            self.rotate_axis(pitch, self.right, degrees=degrees)
        if roll:
            self.rotate_axis(roll, self.forward, degrees=degrees)
        if yaw:
            self.rotate_axis(yaw, WORLD_UP, degrees=degrees)
        return self

    def lookat(self, target, up=WORLD_UP):
        """Orient to face target."""

        target = glm.vec3(target)
        pos = self.position
        delta = target - pos
        if glm.length(delta) < 1e-6:
            return self
        look_dir = glm.normalize(delta)
        up = glm.vec3(up)

        if abs(glm.dot(look_dir, up)) > 0.9999:
            up = WORLD_FORWARD if abs(glm.dot(look_dir, WORLD_FORWARD)) < 0.9999 else WORLD_RIGHT

        s = self.scale
        if not all(np.isfinite(v) and v > 1e-9 for v in (s.x, s.y, s.z)):
            s = glm.vec3(1.0)

        M = glm.inverse(glm.lookAt(pos, target, up))
        M[0] = glm.vec4(glm.vec3(M[0]) * s.x, 0.0)
        M[1] = glm.vec4(glm.vec3(M[1]) * s.y, 0.0)
        M[2] = glm.vec4(glm.vec3(M[2]) * s.z, 0.0)
        M[3] = glm.vec4(pos, 1.0)
        self.transform = M
        return self

    def rescale(self, factors: Union[glm.vec3, ArrayLike, float]):
        s = glm.vec3(factors) if not isinstance(factors, (int, float)) else glm.vec3(factors)
        self.transform = glm.scale(self.transform, s)
        return self

    def follow(self, trajectory: 'Trajectory', dt: float, align: bool = True):
        new_pos, tangent, _, _ = trajectory.advance(dt)
        self.position = new_pos
        if align:
            self.lookat(new_pos - tangent)
        return self


class CollisionMixin:
    """
    Mixin class for objects that need BVH-based collision detection.
    Expects the class to also inherit from TransformMixin (or have a .position attribute).
    """

    @property
    def collider(self):
        """The pytinybvh.BVH object (TLAS) used for collisions."""
        return getattr(self, '_collider_tlas', None)

    @collider.setter
    def collider(self, tlas):
        self._collider_tlas = tlas

    @property
    def collider_radius(self):
        return getattr(self, '_collider_radius', 0.05)

    @collider_radius.setter
    def collider_radius(self, radius: float):
        self._collider_radius = max(0.001, radius)

    def move_and_slide(self, translation: Union[glm.vec3, ArrayLike]):
        """
        Attempt to translate the object. If collision detected, slide along the surface.
        """
        if self.collider is None:
            return self.translate(translation)

        new_pos = self.position + glm.vec3(translation)

        # Query closest geometry to the new position
        hit = self.collider.closest_point(tuple(new_pos))

        if hit is not None and hit['distance'] < self.collider_radius:
            surface_point = hit['point']
            new_pos_np = np.array(new_pos, dtype=np.float32)

            push_out_dir = new_pos_np - surface_point
            norm = np.linalg.norm(push_out_dir)

            if norm > 1e-6:
                push_out_dir /= norm
                # Push the agent out to the edge of its radius
                new_pos_np = surface_point + (push_out_dir * self.collider_radius)
                new_pos = glm.vec3(*new_pos_np)

        self.position = new_pos
        return self

    def snap_to_ground(self, down_dir=(0.0, -1.0, 0.0), max_dist=10.0, leg_height=0.02):
        """
        Casts a ray downwards and snaps the object on the floor.
        """
        if self.collider is None:
            return self

        from pytinybvh import Ray

        down_vec = glm.normalize(glm.vec3(down_dir))

        # start slightly above current position to avoid starting inside geometry
        ray_origin = self.position - down_vec * 0.1

        ray = Ray(origin=ray_origin, direction=down_vec, t=max_dist)
        t = self.collider.intersect(ray)

        if ray.prim_id != -1:
            hit_pos = ray_origin + down_vec * t
            self.position = hit_pos - down_vec * leg_height

        return self

    def tactile_query(self, directions, max_dist=0.1):
        """
        Casts rays in given directions (e.g. tactile sensing with left/right antennae).
        Returns list of booleans (True if an obstacle is closer than max_dist).
        """
        if self.collider is None:
            return [False] * len(directions)

        from pytinybvh import Ray
        results = []

        for d in directions:
            ray = Ray(origin=self.position, direction=np.asarray(glm.normalize(d)), t=max_dist)
            results.append(self.collider.is_occluded(ray))

        return results


class Curve:
    """
    Represents a static 3D path parameterised by arc length.
    """

    def __init__(self, points: Union[np.ndarray, ArrayLike]):

        self.points = np.asarray(points, dtype=np.float32)
        if len(self.points) < 2:
            raise ValueError("Curve must have at least 2 points")

        # Calculate segment lengths and cumulative distance (arc length)
        diffs = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(diffs, axis=1)
        self.cumulative_dists = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.total_length = self.cumulative_dists[-1]

    def get_sample_at(self, distance: float) -> tuple[glm.vec3, glm.vec3, glm.vec3, glm.vec3]:
        """
        Returns (position, tangent) at a specific distance along the curve.
        """

        dist = np.clip(distance, 0.0, self.total_length)
        # idx will is the index of the point *after* the current distance
        idx = np.searchsorted(self.cumulative_dists, dist)

        # Handle start or end of curve
        if idx == 0:
            p0, p1 = self.points[0], self.points[1]
            t = 0.0
        elif idx >= len(self.cumulative_dists):
            p0, p1 = self.points[-2], self.points[-1]
            t = 1.0
        else:
            idx_start = idx - 1
            idx_end = idx

            p0 = self.points[idx_start]
            p1 = self.points[idx_end]

            # Interpolate between p0 and p1
            seg_start_dist = self.cumulative_dists[idx_start]
            seg_len = self.segment_lengths[idx_start]

            if seg_len < 1e-6:
                t = 0.0
            else:
                t = (dist - seg_start_dist) / seg_len

        pos_np = (1.0 - t) * p0 + t * p1

        # Tangent is just the normalised vector of the current segment
        tangent_np = p1 - p0
        norm = np.linalg.norm(tangent_np)
        if norm > 1e-6:
            tangent_np /= norm
        else:
            tangent_np = np.array([0, 0, 1], dtype=np.float32)

        right_np, up_np = tangent_frames(tangent_np)

        return glm.vec3(pos_np), glm.vec3(tangent_np), glm.vec3(right_np), glm.vec3(up_np)


class Trajectory:
    """
    Stateful controller that moves along a Curve.
    """

    def __init__(self, curve: Curve, speed: float = 1.0, loop: bool = True, start_offset: float = 0.0):
        self.curve = curve
        self.speed = speed
        self.loop = loop
        self._current_dist = start_offset
        self._finished = False

    def advance(self, delta_time: float) -> tuple[glm.vec3, glm.vec3, glm.vec3, glm.vec3]:
        """
        Advances the internal state by delta_time * speed.
        Returns (new_position, new_tangent).
        """
        if self._finished and not self.loop:
            return self.curve.get_sample_at(self.curve.total_length)

        step = self.speed * delta_time
        self._current_dist += step

        if self.loop:
            self._current_dist %= self.curve.total_length
        else:
            if self._current_dist >= self.curve.total_length:
                self._current_dist = self.curve.total_length
                self._finished = True
            elif self._current_dist < 0:
                self._current_dist = 0

        return self.curve.get_sample_at(self._current_dist)

    @property
    def is_finished(self):
        return self._finished and not self.loop

    @property
    def progress(self):
        """Returns trajectory progress (0.0 to 1.0)."""
        return self._current_dist / self.curve.total_length


def extract_obj_curves(
        file_path,
        object_filter: Optional[Union[str, Sequence[str]]] = None,
        resample: int = None
    ) -> Dict[str, np.ndarray]:
    """
    Extracts curve coordinates from an .obj file.

    Args:
        file_path: Path to .obj file
        object_filter: Optional name(s) of objects to extract
        resample: Optionally resamples the curve to have this many evenly spaced points.
    """
    file_path = Path(file_path)
    vertices = []
    temp_indices = {}
    current_object = "Default"

    if object_filter is not None and isinstance(object_filter, str):
        target_objects = {object_filter}
    elif object_filter is not None:
        target_objects = set(object_filter)
    else:
        target_objects = object_filter

    with file_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            type_code = parts[0]

            if type_code == 'v':
                vertices.append([float(x) for x in parts[1:4]])

            elif type_code == 'o':
                current_object = parts[1]
                if (not target_objects or current_object in target_objects) and current_object not in temp_indices:
                    temp_indices[current_object] = []

            elif type_code == 'l':
                if not target_objects or current_object in target_objects:
                    indices = []
                    for idx in parts[1:]:
                        idx = int(idx)
                        real_idx = idx - 1 if idx > 0 else len(vertices) + idx
                        indices.append(real_idx)

                    current_list = temp_indices[current_object]

                    if not current_list:
                        current_list.extend(indices)
                    else:
                        start_idx = 1 if current_list[-1] == indices[0] else 0
                        current_list.extend(indices[start_idx:])
    final_curves = {}

    for obj_name, indices in temp_indices.items():
        if not indices:
            continue

        coords = np.array([vertices[i] for i in indices])

        if resample and resample > 1:
            coords = resample_path(coords, resample)

        final_curves[obj_name] = coords

    return final_curves


def resample_path(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Takes a path of points and returns a new path with 'num_samples' evenly spaced along the total arc length.
    """
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dist[-1]
    target_dists = np.linspace(0, total_length, num_samples)
    new_points = np.zeros((num_samples, 3))
    for i in range(3):
        new_points[:, i] = np.interp(target_dists, cum_dist, points[:, i])
    return new_points
