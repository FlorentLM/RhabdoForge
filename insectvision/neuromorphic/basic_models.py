from typing import Tuple
import numpy as np
from insectvision.compound_eyes import Eye


class HassensteinReichardtEMD:
    """
    Elementary Motion Detector based on Hassenstein-Reichardt correlator.

    - GPU temporal accumulation: Photoreceptors (R1-R6) - low-pass integration
    - Python high-pass: Lamina monopolar cells (L1/L2) - Luminance adaptation
    - Python delay/correlator: Medulla (e.g., Mi1/Tm3 to T4/T5) - Delay and multiplication
    """

    def __init__(self,
            eye: Eye, direction: Tuple[float, float, float],
            delay_hz: float = 8.0,
            highpass_hz: float = 2.0,
            coordinate='cartesian'
        ):
        self.eye = eye

        # Convert Hz to time constants
        self.tau_delay = 1.0 / (2.0 * np.pi * delay_hz)
        self.tau_hp = 1.0 / (2.0 * np.pi * highpass_hz)

        self.targets, self.weights = eye.directed_neighbours(
            direction=direction, k=1, coordinate=coordinate, return_weights=True
        )

        self._mean_lum = None   # For high-pass L1/L2 adaptation
        self._delayed_A = None  # LP[A(t)] (correlator)
        self._delayed_B = None  # LP[B(t)] (correlator)

    def process(self, ommatidia_data: np.ndarray, dt: float) -> np.ndarray:

        # GPU output (must be already low-pass filtered)
        eye_data = ommatidia_data[self.eye.global_indices]
        luminance = eye_data[:, :3].mean(axis=1)

        # Lamina L1/L2 high-pass equivalent (luminance adaptation / contrast)
        alpha_hp = dt / (self.tau_hp + dt)
        if self._mean_lum is None:
            self._mean_lum = luminance.copy()
            return np.zeros(len(self.eye), dtype=np.float32)
        else:
            self._mean_lum += alpha_hp * (luminance - self._mean_lum)

        # Convert to contrast (removes DC, normalises by local mean)
        contrast = (luminance - self._mean_lum) / (self._mean_lum + 1e-6)

        # Medulla delay lines
        alpha_delay = dt / (self.tau_delay + dt)

        signal_A = contrast  # direct channel
        signal_B = contrast[self.targets]  # neighbour channel

        if self._delayed_A is None:
            self._delayed_A = signal_A.copy()
            self._delayed_B = signal_B.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        # Update delay lines
        self._delayed_A += alpha_delay * (signal_A - self._delayed_A)
        self._delayed_B += alpha_delay * (signal_B - self._delayed_B)

        # Correlator: preferred arm - anti-preferred arm
        motion = signal_B * self._delayed_A - signal_A * self._delayed_B

        return motion * self.weights
