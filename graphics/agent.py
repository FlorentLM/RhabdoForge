from typing import Sequence
from numpy.typing import ArrayLike
from pyglm import glm
import numpy as np
from graphics.camera import Camera


class Agent:
    """
    Represents an agent (an insect) in the 3D world
    Pretty much just a wrapper around the camera (for now)
    """

    def __init__(self, position: Sequence[float | int] = (0.0, 0.0, 0.0), ratio: float = 16/9):

        self.camera = Camera(position=glm.vec3(position), ratio=ratio)
        self.move_sensitivity = 0.01
        self.mouse_sensitivity = 0.25

    def move(self, direction: glm.vec3 | ArrayLike):
        """ Moves the agent in a specified direction vector """

        direction = glm.vec3(direction)

        if glm.length(direction) > 0:
            displacement = glm.normalize(direction) * self.move_sensitivity
            self.camera.pos += displacement

    def rotate(self, yaw_delta: float, pitch_delta: float):
        """ Rotates the agent's view """

        self.camera.yaw -= yaw_delta * self.mouse_sensitivity
        self.camera.pitch -= pitch_delta * self.mouse_sensitivity
        self.camera.pitch = np.clip(self.camera.pitch, -89.0, 89.0)

    # TODO: add automatic getters to avoid wrapping every single property

    @property
    def position(self):
        return self.camera.pos

    @property
    def forward(self):
        return self.camera.forward

    @property
    def backward(self):
        return self.camera.backward