from typing import Any, Union
from numpy.typing import ArrayLike
from pyglm import glm


WORLD_RIGHT = WORLD_X = glm.vec3(1.0, 0.0, 0.0)
WORLD_UP = WORLD_Y = glm.vec3(0.0, 1.0, 0.0)
WORLD_FORWARD = WORLD_Z = glm.vec3(0.0, 0.0, -1.0)

WORLD_LEFT = -WORLD_RIGHT
WORLD_DOWN = -WORLD_UP
WORLD_BACKWARD = -WORLD_FORWARD


class DeltaTimeTransformer:
    """
    A proxy object for applying framerate-independent transforms.

    It wraps a target object (like an Agent or Instance) and scales all
    subsequent chained transformation calls by a delta_time value.
    """
    def __init__(self, target: Any, delta_time: float):
        self._target = target
        self._delta_time = delta_time

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        scaled_translation = glm.vec3(translation) * self._delta_time
        self._target.translate(scaled_translation)
        return self

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        scaled_angle = angle * self._delta_time
        self._target.rotate_axis(scaled_angle, axis, degrees=degrees)
        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0, degrees: bool = True):
        self._target.rotate(
            yaw_delta * self._delta_time,
            pitch_delta * self._delta_time,
            roll_delta * self._delta_time,
            degrees=degrees
        )
        return self

    def scale(self, scale_factors: Union[glm.vec3, ArrayLike]):
        """
        Applies scaling over time (i.e. a factor of 1.1 will scale *towards* 10% larger, not instantly become 1.1x as large).
        """
        scale_vec = glm.vec3(scale_factors)

        # interpolate between no-scale (1, 1, 1) and target scale
        interpolated_scale = glm.mix(glm.vec3(1.0), scale_vec, self._delta_time)
        self._target.scale(interpolated_scale)
        return self

    def follow(self, trajectory, align_orientation: bool = True):
        """
        Updates the target's position based on the Trajectory state.

        Args:
            trajectory: An instance of geometry.paths.Trajectory
            align_orientation: If True, calls lookat() (or equivalent) to face movement direction.
        """
        new_pos, tangent, right, up = trajectory.advance(self._delta_time)

        self._target.position = new_pos

        if align_orientation:
            # look at a point slightly ahead in tangent direction
            target_look = new_pos - tangent

            if hasattr(self._target, 'lookat'):
                self._target.lookat(target_look)

            elif hasattr(self._target, 'direction'):
                self._target.direction = tangent

        return self

    def move_and_slide(self, translation: Union[glm.vec3, ArrayLike]):
        scaled_translation = glm.vec3(translation) * self._delta_time
        self._target.move_and_slide(scaled_translation)
        return self

