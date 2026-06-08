import numpy as np
from numpy.typing import ArrayLike

from insectvision.compound_eyes import Eye


class HassensteinReichardtEMD:
    """
    Elementary Motion Detector (based on Hassenstein-Reichardt correlator), with ON/OFF motion pathways.

    - Photoreceptors: Uses pooled R1-R6 peripheral signals (neural superposition).
    - Lamina L1/L2: High-pass filtering for luminance adaptation and contrast extraction.
    - Rectification: Splits contrast into ON (brighter) and OFF (darker) parallel pathways.
    - Medulla (T4/T5 cells): Delay lines and cross-multiplication.
        T4 cells correlate ON signals, T5 cells correlate OFF signals.
    - Output: Recombines T4 and T5 responses into a directionally selective motion vector.

    Args:
        eye (Eye): The single eye to process.
        direction (ArrayLike): The motion direction to correlate against

        coordinate (str): 'spherical' or 'cartesian' for the direction parameter.
    """

    def __init__(self,
        eye: Eye,
        direction: ArrayLike,
        delay_coeff: float = 0.20,      # delay-line blend (smaller = longer delay/memory)
        highpass_coeff: float = 0.10,   # LMC adaptation blend
        coordinate='cartesian'
        ):

        self.eye = eye
        self.self_indices = eye.lens_indices

        self.delay_coeff = delay_coeff
        self.highpass_coeff = highpass_coeff

        # Lens-level directed neighbours (eye-local indices)
        self.targets, self.weights = eye.lenses.directed_neighbours(
            direction=direction, k=1, coordinate=coordinate, return_weights=True
        )

        self._mean_lum = None

        # Split ON/OFF delay lines
        self._delayed_ON_A = None
        self._delayed_ON_B = None
        self._delayed_OFF_A = None
        self._delayed_OFF_B = None

    def process(self, visual_output: 'VisualOutput') -> np.ndarray:

        lmc_signal = visual_output.lmc_input

        # Radiance/Luminance
        luminance = lmc_signal[:, :3].mean(axis=-1)

        if self._mean_lum is None:
            self._mean_lum = luminance.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        # Lamina L1/L2 high-pass (luminance adaptation / contrast)
        self._mean_lum += self.highpass_coeff * (luminance - self._mean_lum)
        global_contrast = (luminance - self._mean_lum) / (self._mean_lum + 1e-6)

        # Split into ON (L1->T4) and OFF (L2->T5) pathways
        signal_ON = np.maximum(global_contrast, 0.0)
        signal_OFF = np.maximum(-global_contrast, 0.0)

        sig_ON_A = signal_ON[self.self_indices]
        sig_ON_B = signal_ON[self.targets]
        sig_OFF_A = signal_OFF[self.self_indices]
        sig_OFF_B = signal_OFF[self.targets]

        # Medulla delay lines
        if self._delayed_ON_A is None:
            self._delayed_ON_A = sig_ON_A.copy()
            self._delayed_ON_B = sig_ON_B.copy()
            self._delayed_OFF_A = sig_OFF_A.copy()
            self._delayed_OFF_B = sig_OFF_B.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        a = self.delay_coeff
        self._delayed_ON_A += a * (sig_ON_A - self._delayed_ON_A)
        self._delayed_ON_B += a * (sig_ON_B - self._delayed_ON_B)
        self._delayed_OFF_A += a * (sig_OFF_A - self._delayed_OFF_A)
        self._delayed_OFF_B += a * (sig_OFF_B - self._delayed_OFF_B)

        # Correlate ON with ON, OFF with OFF
        motion_ON = sig_ON_B * self._delayed_ON_A - sig_ON_A * self._delayed_ON_B
        motion_OFF = sig_OFF_B * self._delayed_OFF_A - sig_OFF_A * self._delayed_OFF_B

        # Recombine T4 and T5
        total_motion = (motion_ON + motion_OFF) * self.weights

        return total_motion