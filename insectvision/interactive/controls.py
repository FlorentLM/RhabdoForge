from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

if TYPE_CHECKING:
    from insectvision.engine import Context


@dataclass
class Action:
    id: str
    name: str
    category: str
    description: str
    callback: 'Callable'
    keyboard_hint: str = ''
    gamepad_hint: str = ''


class ActionRegistry:
    def __init__(self, ctx: 'Context'):
        self.ctx = ctx
        self.actions: Dict[str, 'Action'] = {}

        self._setup()

    def register(self,
        id: str,
        name: str,
        category: str,
        description: str,
        callback: 'Callable',
        k_hint: str = '',
        g_hint: str = ''
        ) -> None:

        self.actions[id] = Action(id, name, category, description, callback, k_hint, g_hint)

    def trigger(self, action_id: str) -> None:
        if action_id in self.actions:
            self.actions[action_id].callback()

    def _setup(self):

        # Rendering
        self.register(
            'view_cycle',
            'Change camera',
            'Rendering',
            'Cycle through camera modes',
            self.ctx.cycle_view_mode,
            'C',
            'R-Stick Click'
        )
        self.register(
            'voronoi_toggle',
            'Toggle tiled view',
            'Rendering',
            'Tiled Vornoi view',
            self.ctx.toggle_voronoi,
            'V',
            'B / Circle'
        )
        self.register(
            'proj_toggle',
            'Change projection mode',
            'Rendering',
            'Physical vs. Acceptance',
            self.ctx.toggle_projection_mode,
            'P',
            'Y / Triangle'
        )
        self.register(
            'heatmap_toggle',
            'Toggle Heatmap',
            'Rendering',
            'Enable heatmap overlay',
            self.ctx.toggle_heatmap,
            'H',
            ''
        )
        self.register(
            'dither_toggle',
            'Toggle time dithering',
            'Rendering',
            'Temporal randomness',
            self.ctx.toggle_time_dithering,
            'T',
            'D-Pad Left'
        )
        self.register(
            'dither_once',
            'Dither Once',
            'Rendering',
            'Randomise once',
            self.ctx.dither_once,
            'X',
            ''
        )

        # Sampling
        self.register(
            'samples_inc',
            'Increase Samples',
            'Sampling',
            'More rays/receptor',
            self.ctx.increase_samples,
            '+',
            'D-Pad Up'
        )
        self.register(
            'samples_dec',
            'Decrease Samples',
            'Sampling',
            'Fewer rays/receptor',
            self.ctx.decrease_samples,
            '-',
            'D-Pad Down'
        )

        # Environment / Sun
        self.register(
            'sun_ctrl_toggle',
            'Sun Control',
            'Environment',
            'Mouse/Stick controls sun',
            self.ctx.toggle_sun_control,
            'L',
            'L-Stick Click'
        )

        # Agent
        self.register(
            'reset_pos',
            'Reset Position',
            'Agent',
            'Teleport to origin',
            self.ctx.reset_position,
            'O',
            'Start'
        )
        self.register(
            'reset_rot',
            'Reset Rotation',
            'Agent',
            'Reset rotation',
            self.ctx.reset_rotation,
            'Backspace',
            'LB+RB'
        )

        # Dynamics
        self.register(
            'saccade_toggle',
            'Toggle Actuation',
            'Dynamics',
            'Rhabdomere saccades',
            self.ctx.toggle_saccades,
            'R',
            'D-Pad Right'
        )

        # UI
        self.register(
            'hud_toggle',
            'Toggle HUD',
            'UI',
            'Show/hide text overlay',
            self.ctx.toggle_hud,
            'I',
            'Select'
        )
        self.register(
        'debug_toggle',
            'Toggle Debug',
            'UI',
            'Show debug geometry',
            self.ctx.toggle_debug,
            'G',
            ''
        )
        self.register(
            'mouse_lock_toggle',
            'Toggle Mouse Lock',
            'UI',
            'Lock/Unlock mouse to window',
            self.ctx.toggle_mouse_capture,
            'Tab',
            ''
        )

    def get_by_category(self, category: str) -> List['Action']:
        return [a for a in self.actions.values() if a.category == category]


class Controls(ABC):
    """
    Abstract interface for input controllers.
    """

    @abstractmethod
    def setup(self, ctx: 'Context') -> None:
        """Register GLFW callbacks, detect hardware, etc"""
        ...

    @abstractmethod
    def poll(self, ctx: 'Context') -> None:
        """Read inputs and do action."""
        ...

    def free(self) -> None:
        """Release resources, deregister callbacks."""
        pass
