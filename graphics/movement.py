from typing import Union

import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import ArrayLike
from pyglm import glm


class TransformMixin:
    """
    Mixin class for common transformation methods.

    Currently classes using this mixin should have:
    For agent-like objects:
    - `_position` attribute
    - `yaw`, `pitch`, `roll` attributes
    - `orientation` property
    - `right`, `up`, `forward` properties

      OR

    For instance-like objects:
    - `transform` attribute (glm.mat4)

    In both cases:
    - A `dt()` method returning a DeltaTimeTransformer
    """

    @property
    def position(self):

        if hasattr(self, '_position'):
            return self._position

        elif hasattr(self, 'transform'):
            return glm.vec3(self.transform[3])

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must have either '_position' or 'transform' attribute"
            )

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):

        if hasattr(self, '_position'):
            self._position = glm.vec3(value)

        elif hasattr(self, 'transform'):
            self.transform[3] = glm.vec4(glm.vec3(value), 1.0)

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must have either '_position' or 'transform' attribute"
            )

    pos = position

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        """
        Translate by a given vector.
        """

        if hasattr(self, '_position'):
            self._position += glm.vec3(translation)

        elif hasattr(self, 'transform'):
            self.transform = glm.translate(self.transform, glm.vec3(translation))

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must have either '_position' or 'transform' attribute"
            )
        return self

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        """
        Rotate around a given axis.
        """

        if isinstance(axis, str):
            from graphics.utils import WORLD_RIGHT, WORLD_UP, WORLD_FORWARD

            axis_str = axis.lower()

            if hasattr(self, 'yaw') and hasattr(self, 'orientation'):
                axis_map = {
                    'x': self.right,
                    'y': self.up,
                    'z': self.forward,
                    'right': self.right,
                    'left': self.left,
                    'up': self.up,
                    'down': self.down,
                    'forward': self.forward,
                    'backward': self.backward,
                    'yaw': WORLD_UP,
                    'pitch': self.right,
                    'roll': self.forward,
                }
            else:
                axis_map = {
                    'x': WORLD_RIGHT,
                    'y': WORLD_UP,
                    'z': WORLD_FORWARD,
                    'right': WORLD_RIGHT,
                    'left': -WORLD_RIGHT,
                    'up': WORLD_UP,
                    'down': -WORLD_UP,
                    'forward': WORLD_FORWARD,
                    'backward': -WORLD_FORWARD,
                    'yaw': WORLD_UP,
                    'pitch': WORLD_RIGHT,
                    'roll': WORLD_FORWARD,
                }

            try:
                rotation_axis = axis_map[axis_str]
            except KeyError:
                raise ValueError(
                    f"Unknown axis identifier: '{axis}'. Valid options are: {list(axis_map.keys())}"
                )
        else:
            rotation_axis = glm.vec3(axis)

        angle_rad = glm.radians(angle) if degrees else angle

        # Agent-style: decompose to yaw/pitch/roll
        if hasattr(self, 'yaw') and hasattr(self, 'orientation'):
            new_orientation = glm.rotate(self.orientation, angle_rad, rotation_axis)
            self._decompose_orientation(new_orientation)

        # Instance-style: apply to transform matrix
        elif hasattr(self, 'transform'):
            self.transform = glm.rotate(self.transform, angle_rad, rotation_axis)

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} must have either 'orientation' or 'transform'"
            )

        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0, degrees: bool = True):
        """
        Rotate by yaw, pitch, and roll deltas.
        """

        if hasattr(self, 'yaw') and hasattr(self, 'pitch') and hasattr(self, 'roll'):
            self.yaw += yaw_delta if degrees else glm.degrees(yaw_delta)
            self.pitch += pitch_delta if degrees else glm.degrees(pitch_delta)
            self.roll += roll_delta if degrees else glm.degrees(roll_delta)

        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support yaw/pitch/roll rotation. "
                "Use rotate_axis() instead."
            )
        return self

    def _decompose_orientation(self, orientation_matrix: glm.mat4):
        """
        Decompose an orientation matrix into yaw, pitch, and roll.
        """
        if not (hasattr(self, 'yaw') and hasattr(self, 'pitch') and hasattr(self, 'roll')):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have yaw/pitch/roll attributes"
            )

        sin_pitch = -orientation_matrix[2][1]

        # Clamp to avoid errors with asin
        if sin_pitch >= 1.0:
            self.pitch = 90.0
        elif sin_pitch <= -1.0:
            self.pitch = -90.0
        else:
            self.pitch = glm.degrees(glm.asin(sin_pitch))

        cos_pitch = glm.cos(glm.radians(self.pitch))

        if abs(cos_pitch) > 1e-6:
            # Yaw
            sin_yaw = orientation_matrix[2][0] / cos_pitch
            cos_yaw = orientation_matrix[2][2] / cos_pitch
            self.yaw = glm.degrees(glm.atan2(sin_yaw, cos_yaw))

            # Roll
            sin_roll = orientation_matrix[0][1] / cos_pitch
            cos_roll = orientation_matrix[1][1] / cos_pitch
            self.roll = glm.degrees(glm.atan2(sin_roll, cos_roll))
        else:
            # Gimbal lock: roll and yaw axes are aligned
            self.roll = 0.0
            sin_yaw = -orientation_matrix[0][2]
            cos_yaw = orientation_matrix[0][0]
            self.yaw = glm.degrees(glm.atan2(sin_yaw, cos_yaw))

    def lookat(self, target_pos: Union[glm.vec3, ArrayLike]):
        """
        Orient to look at a target position.
        """
        if not (hasattr(self, 'yaw') and hasattr(self, 'pitch')):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support lookat (requires yaw/pitch)"
            )

        target_pos = glm.vec3(target_pos)

        if glm.distance(target_pos, self.position) < 1e-6:
            return self

        direction = glm.normalize(target_pos - self.position)
        self.pitch = glm.degrees(glm.asin(glm.clamp(direction.y, -1.0, 1.0)))
        self.yaw = glm.degrees(glm.atan2(direction.x, -direction.z))

        return self

    def scale(self, scale_factors: Union[glm.vec3, ArrayLike, float]):
        """
        Scale the object (for objects with transform matrices).
        """
        if not hasattr(self, 'transform'):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support scaling"
            )

        if isinstance(scale_factors, (int, float)):
            scale_vec = glm.vec3(scale_factors)
        else:
            scale_vec = glm.vec3(scale_factors)

        self.transform = glm.scale(self.transform, scale_vec)
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

        ray = Ray(origin=tuple(ray_origin), direction=tuple(down_vec), t=max_dist)
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
            ray = Ray(origin=tuple(self.position), direction=tuple(glm.normalize(d)), t=max_dist)
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

    def get_sample_at(self, distance: float) -> tuple[glm.vec3, glm.vec3]:
        """
        Returns (position, tangent) at a specific distance along the curve.
        """
        # Clamp distance
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

        return glm.vec3(pos_np), glm.vec3(tangent_np)


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

    def advance(self, delta_time: float) -> tuple[glm.vec3, glm.vec3]:
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
