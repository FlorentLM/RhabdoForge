import numpy as np
from numpy.typing import ArrayLike

from insectvision.compound_eyes import Eye


class HassensteinReichardtEMD:
    """
    Elementary Motion Detector based on Hassenstein-Reichardt correlator.

    - GPU temporal accumulation: Photoreceptors (R1-R6) - low-pass integration
    - Python high-pass: Lamina monopolar cells (L1/L2) - Luminance adaptation
    - Python delay/correlator: Medulla (e.g., Mi1/Tm3 to T4/T5) - Delay and multiplication
    """

    def __init__(self,
            eye: Eye,
            direction: ArrayLike,
            delay_hz: float = 8.0,
            highpass_hz: float = 2.0,
            coordinate='cartesian'
        ):
        self.eye = eye
        self.self_indices = eye.lens_indices

        # Convert Hz to time constants
        self.tau_delay = 1.0 / (2.0 * np.pi * delay_hz)
        self.tau_hp = 1.0 / (2.0 * np.pi * highpass_hz)

        # Lens-level directed neighbours (eye-local indices)
        self.targets, self.weights = eye.lenses.directed_neighbours(
            direction=direction, k=1, coordinate=coordinate, return_weights=True
        )

        self._mean_lum = None   # High-pass state (L1/L2 adaptation), kept global for simplicity
        self._delayed_A = None  # Delay line A (correlator), local to this eye
        self._delayed_B = None  # Delay line B (correlator), local to this eye

    def process(self, view, dt: float) -> np.ndarray:
        """
        Process one frame.

        Args:
            - global_view: The full (non sliced) per-lens buffer (N_total, R, channels)
        """

        # Luminance (for the whole animal)
        luminance = view[:, :, :3].mean(axis=(1, 2))

        # Lamina L1/L2 high-pass equivalent (luminance adaptation / contrast)
        alpha_hp = dt / (self.tau_hp + dt)
        if self._mean_lum is None:
            self._mean_lum = luminance.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        self._mean_lum += alpha_hp * (luminance - self._mean_lum)
        global_contrast = (luminance - self._mean_lum) / (self._mean_lum + 1e-6)

        # signal_A: contrast at the lenses of this eye (direct channel)
        # signal_B: contrast at the neighbours
        signal_A = global_contrast[self.self_indices]
        signal_B = global_contrast[self.targets]

        # Medulla delay lines (this eye)
        alpha_delay = dt / (self.tau_delay + dt)

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