from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from rhabdoforge.compound_eyes import EyeView
    from rhabdoforge.renderers.helpers import VisualOutput


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
        eye (EyeView): The single eye to process.
        direction (ArrayLike): The motion direction to correlate against
        coordinate (str): 'spherical' or 'cartesian' for the direction parameter.
    """

    def __init__(self,
            eye: 'EyeView',
            direction: ArrayLike,
            tau_delay: float = 0.020,      # 20 ms
            tau_highpass: float = 0.08,    # 80 ms
            pooling_k: Optional[int] = None,
            coordinate='cartesian'
        ):

        self.eye = eye
        self.local_indices = np.arange(len(eye))

        # Baseline selection:
        # it should be large enough to capture motion but constrained by the eye's own optical 'blur'
        median_ioa = np.median(eye.interommatidial_angles[:, 0])
        # Baseline: at least 2 ommatidia, but not less than 1.5 degrees
        self.baseline = max(median_ioa * 2.0, np.radians(1.5))

        # Pooling:
        if pooling_k is None:
            # Auto: pool over the same area as the correlation span
            self.pooling_k = max(1, int(round(self.baseline / median_ioa)))
        else:
            self.pooling_k = max(1, pooling_k)

        self.pool_graph = eye._get_neighbour_graph(k=self.pooling_k)['neighbour_indices']

        target_indices, weights = eye.directed_neighbours(
            direction=direction,
            distance=self.baseline,
            k=1,
            coordinate=coordinate,
            return_weights=True,
            k_search=20
        )
        self.local_targets, _ = eye._to_local(target_indices)

        # Angular distance for every pair
        dirs_self = self.eye.directions
        dirs_target = self.eye.model.directions[target_indices]
        self.delta_phi = np.arccos(np.clip(np.sum(dirs_self * dirs_target, axis=1), -1.0, 1.0))

        # Normalise by the distance sampled
        self.weights = weights / (self.delta_phi + 1e-4)

        self.tau_delay, self.tau_hp = tau_delay, tau_highpass
        self._mean_lum = None
        self.last_estimate = 0.0

        # Split ON/OFF delay lines
        self._delayed_ON_A = None
        self._delayed_ON_B = None
        self._delayed_OFF_A = None
        self._delayed_OFF_B = None

    def process(self, visual_output: 'VisualOutput', dt: float) -> np.ndarray:

        # Radiance/Luminance
        raw_luminance = visual_output.lmc_input[self.eye.indices, :3].mean(axis=-1)
        luminance = np.mean(raw_luminance[self.pool_graph], axis=1)

        # Lamina L1/L2 high-pass (luminance adaptation / contrast)
        alpha_hp = dt / (self.tau_hp + dt)
        if self._mean_lum is None:
            self._mean_lum = luminance.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        self._mean_lum += alpha_hp * (luminance - self._mean_lum)

        # Contrast normalisation (L1-style)
        contrast = (luminance - self._mean_lum) / (self._mean_lum + 1e-2)

        # Split into ON (L1->T4) and OFF (L2->T5) pathways
        signal_ON = np.maximum(contrast, 0.0)
        signal_OFF = np.maximum(-contrast, 0.0)

        A_ON = signal_ON[self.local_indices]
        B_ON = signal_ON[self.local_targets]
        A_OFF = signal_OFF[self.local_indices]
        B_OFF = signal_OFF[self.local_targets]

        # Correlation with delay
        alpha_delay = dt / (self.tau_delay + dt)

        if self._delayed_ON_A is None:
            self._delayed_ON_A = A_ON.copy()
            self._delayed_ON_B = B_ON.copy()
            self._delayed_OFF_A = A_OFF.copy()
            self._delayed_OFF_B = B_OFF.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        self._delayed_ON_A += alpha_delay * (A_ON - self._delayed_ON_A)
        self._delayed_ON_B += alpha_delay * (B_ON - self._delayed_ON_B)
        self._delayed_OFF_A += alpha_delay * (A_OFF - self._delayed_OFF_A)
        self._delayed_OFF_B += alpha_delay * (B_OFF - self._delayed_OFF_B)

        motion_ON = B_ON * self._delayed_ON_A - A_ON * self._delayed_ON_B
        motion_OFF = B_OFF * self._delayed_OFF_A - A_OFF * self._delayed_OFF_B

        # Saturation and output
        total_motion = np.tanh(motion_ON + motion_OFF) * self.weights

        # Pooling for the control estimate
        self.last_estimate = np.median(total_motion)

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
            eye: 'EyeView',
            direction: ArrayLike,
            tau_smooth: float = 0.05,  # 50 ms smoothing time constant
            pooling_k: Optional[int] = None,
            coordinate: str = 'cartesian'
        ):

        self.eye = eye
        self.local_indices = np.arange(len(eye))

        # Resolution-appropriate baseline for the derivative
        median_ioa = np.median(eye.interommatidial_angles[:, 0])
        # Baseline: at least 2 ommatidia, but not less than 1.5 degrees
        self.baseline = max(median_ioa * 2.0, np.radians(1.5))

        if pooling_k is None:
            # Auto: pool over the distance used for the spatial derivative
            self.pooling_k = max(1, int(round(self.baseline / median_ioa)))
        else:
            self.pooling_k = max(1, pooling_k)

        self.pool_graph = eye._get_neighbour_graph(k=self.pooling_k)['neighbour_indices']

        target_indices, weights = eye.directed_neighbours(
            direction=direction,
            distance=self.baseline,
            k=1,
            coordinate=coordinate,
            return_weights=True
        )

        self.weights = weights  # delta_phi used inside process()
        self.local_targets, _ = eye._to_local(target_indices)

        # delta phi between pooled pairs
        dirs_self = self.eye.directions
        dirs_target = self.eye.model.directions[target_indices]
        self.delta_phi = np.maximum(np.arccos(np.clip(np.sum(dirs_self * dirs_target, axis=1), -1.0, 1.0)), 1e-4)

        self.tau_smooth = tau_smooth

        self._prev_self = None      # previous-frame home-lens luminance
        self._prev_target = None    # previous-frame target lens luminance
        self.last_estimate = 0.0    # pooled Lucas-Kanade velocity

        self._num_ema = 0.0
        self._den_ema = 0.0

    def process(self, visual_output: 'VisualOutput', dt: float) -> np.ndarray:

        raw_luminance = visual_output.lmc_input[self.eye.indices, :3].mean(axis=-1)
        luminance = np.mean(raw_luminance[self.pool_graph], axis=1)

        I_self = luminance[self.local_indices]
        I_target = luminance[self.local_targets]

        if self._prev_self is None:
            self._prev_self = I_self.copy()
            self._prev_target = I_target.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        # Average spatial gradient across current and previous frame
        I_x = 0.5 * ((I_target - I_self) + (self._prev_target - self._prev_self)) / self.delta_phi

        # Average temporal gradient across self and target lenses
        I_t_grad = 0.5 * ((I_self - self._prev_self) + (I_target - self._prev_target)) / dt

        self._prev_self = I_self.copy()
        self._prev_target = I_target.copy()

        # Pooled (Lucas-Kanade) estimate: ratio of sums -> A/k cancellation
        alpha = dt / (self.tau_smooth + dt)

        inst_num = np.sum(self.weights * I_t_grad * I_x)
        inst_den = np.sum(self.weights * I_x * I_x)
        self._num_ema += alpha * (inst_num - self._num_ema)
        self._den_ema += alpha * (inst_den - self._den_ema)

        # Dynamic stability floor
        den_floor = 1e-4 + 0.05 * np.max(self._den_ema)
        self.last_estimate = float(abs(-self._num_ema / (self._den_ema + den_floor)))

        # Per-lens local velocity (for the heatmap overlay and the mean balance)
        v_local = -(I_t_grad * I_x) / (I_x * I_x + 1e-3)

        return (v_local * self.weights).astype(np.float32)