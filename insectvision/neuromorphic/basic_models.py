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
                 tau_delay: float = 0.020,      # 20 ms
                 tau_highpass: float = 0.08,    # 80 ms
                 coordinate='cartesian'
                 ):
        self.eye = eye
        self.self_indices = eye.lens_indices

        self.tau_delay = tau_delay
        self.tau_hp = tau_highpass

        # Lens-level directed neighbours (eye-local indices)
        self.targets, self.weights = eye.ommatidia.directed_neighbours(
            direction=direction, k=1, coordinate=coordinate, return_weights=True
        )

        self._mean_lum = None
        self.last_estimate = 0.0

        # Split ON/OFF delay lines
        self._delayed_ON_A = None
        self._delayed_ON_B = None
        self._delayed_OFF_A = None
        self._delayed_OFF_B = None

    def process(self, visual_output: 'VisualOutput', dt: float) -> np.ndarray:

        lmc_signal = visual_output.lmc_input

        # Radiance/Luminance
        luminance = lmc_signal[:, :3].mean(axis=-1)

        # Lamina L1/L2 high-pass (luminance adaptation / contrast)
        alpha_hp = dt / (self.tau_hp + dt)
        if self._mean_lum is None:
            self._mean_lum = luminance.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        self._mean_lum += alpha_hp * (luminance - self._mean_lum)
        global_contrast = (luminance - self._mean_lum) / (self._mean_lum + 1e-6)

        # Split into ON (L1->T4) and OFF (L2->T5) pathways
        signal_ON = np.maximum(global_contrast, 0.0)
        signal_OFF = np.maximum(-global_contrast, 0.0)

        sig_ON_A = signal_ON[self.self_indices]
        sig_ON_B = signal_ON[self.targets]
        sig_OFF_A = signal_OFF[self.self_indices]
        sig_OFF_B = signal_OFF[self.targets]

        # Medulla delay lines
        alpha_delay = dt / (self.tau_delay + dt)

        if self._delayed_ON_A is None:
            self._delayed_ON_A = sig_ON_A.copy()
            self._delayed_ON_B = sig_ON_B.copy()
            self._delayed_OFF_A = sig_OFF_A.copy()
            self._delayed_OFF_B = sig_OFF_B.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        self._delayed_ON_A += alpha_delay * (sig_ON_A - self._delayed_ON_A)
        self._delayed_ON_B += alpha_delay * (sig_ON_B - self._delayed_ON_B)
        self._delayed_OFF_A += alpha_delay * (sig_OFF_A - self._delayed_OFF_A)
        self._delayed_OFF_B += alpha_delay * (sig_OFF_B - self._delayed_OFF_B)

        # Correlate ON with ON, OFF with OFF
        motion_ON = sig_ON_B * self._delayed_ON_A - sig_ON_A * self._delayed_ON_B
        motion_OFF = sig_OFF_B * self._delayed_OFF_A - sig_OFF_A * self._delayed_OFF_B

        # Recombine T4 and T5 (this is per lens)
        total_motion = (motion_ON + motion_OFF) * self.weights

        self.last_estimate = np.mean(total_motion)  # estimate is the mean over the whole eye

        return total_motion


class GradientFlowDetector:
    """
    Gradient (ratio-based) optic-flow estimator.

    Estimates true angular velocity via the local optic-flow constraint
        v = -(dI/dt) / (dI/dx),
    in which contrast and spatial frequency cancel.

    It thus balances on actual image speed and should centre independently
    of wall texture density (cf. Srinivasan et al. 1991), where the correlator does not.
    """

    def __init__(self,
        eye: Eye,
        direction: ArrayLike,
        coordinate: str = 'cartesian',
        eps: float = 1e-9,
        tau_smooth: float = 0.05    # 50 ms smoothing time constant
        ):

        self.eye = eye
        self.self_indices = eye.lens_indices

        self.eps = eps
        self.tau_smooth = tau_smooth

        self.targets, self.weights = eye.ommatidia.directed_neighbours(
            direction=direction, k=1, coordinate=coordinate, return_weights=True
        )

        self._prev = None            # previous-frame home-lens luminance
        self.last_estimate = 0.0     # pooled Lucas-Kanade velocity

        self._num_ema = 0.0
        self._den_ema = 0.0

    def process(self, visual_output: 'VisualOutput', dt: float) -> np.ndarray:

        luminance = visual_output.lmc_input[:, :3].mean(axis=-1)
        I_self = luminance[self.self_indices]

        if self._prev is None:
            self._prev = I_self.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        # Spatial gradient along flow axis (current frame), temporal gradient (home lens)
        I_x = luminance[self.targets] - I_self
        I_t = (I_self - self._prev) / dt
        self._prev = I_self.copy()

        # Pooled (Lucas-Kanade) estimate: ratio of sums -> A/k cancellation
        alpha = dt / (self.tau_smooth + dt)

        inst_num = np.sum(self.weights * I_t * I_x)
        inst_den = np.sum(self.weights * I_x * I_x)
        self._num_ema += alpha * (inst_num - self._num_ema)
        self._den_ema += alpha * (inst_den - self._den_ema)

        self.last_estimate = float(abs(-self._num_ema / (self._den_ema + self.eps)))  # currently lens/s
        # TODO: divide by angular neighbour spacing to get rad/s

        # Per-lens local velocity (for the heatmap overlay and the mean balance)
        v_local = -(I_t * I_x) / (I_x * I_x + 1e-3)

        return (v_local * self.weights).astype(np.float32)