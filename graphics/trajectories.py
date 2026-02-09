import numpy as np
from pyglm import glm
from typing import Union
from numpy.typing import ArrayLike


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