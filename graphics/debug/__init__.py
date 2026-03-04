"""
Examples:

    debug.add(DebugGrid())
    debug.add(AxisGizmo(size=0.4, track=agent))
    debug.add(GazeArrow(agent))
    debug.add(LabeledPoint((1, 2, 3)))
    debug.add(LabeledPoint(agent, label="agent"))
    debug.add(LabeledPoint.from_vertex(my_asset, 42))
    debug.add(LabeledPoint.from_vertex(my_instance, 42))
    debug.add(DebugBox(source=my_asset))
    debug.add(DebugBox(source=my_bvh))
    debug.add(DebugPath(my_curve))
    debug.add(DebugPath(my_trajectory, show_progress=True))
    debug.add(DebugFrustum(agent, far=2.0))

    # in draw loop
    debug.draw(context)

    # or a specific POV object
    debug.draw(agent)
    debug.draw(observer)

    # or explicit matrices
    debug.draw(view=view_mat, proj=proj_mat)

    # toggle
    some_item.visible = False

    debug.remove(some_item)

    debug.free()
"""

from typing import Optional
from graphics.debug.debug_renderer import DebugRenderer
from graphics.debug.debug_objects import (
    DebugDrawable,
    LineObject,
    ArrowObject,
    MarkerObject,
    AxesGizmo,
    GazeArrow,
    DirectionArrow,
    DebugGrid,
    LabeledPoint,
    DebugLine,
    DebugRay,
    DebugPath,
    DebugBox,
    DebugFrustum,
)


class DebugOverlay:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._renderer: Optional[DebugRenderer] = None  # lazy loaded
        self._items = []

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = DebugRenderer()

    def add(self, *items):
        for item in items:
            if item not in self._items:
                self._items.append(item)
        return items[0] if len(items) == 1 else list(items)

    def remove(self, item):
        try:
            self._items.remove(item)
        except ValueError:
            pass

    def clear(self):
        self._items.clear()

    def draw(self, context_or_pov=None, view=None, proj=None, pov=None):
        """
        Draw all debug objects.

        Can be called with:
            debug.draw(ctx)                             # Context (uses current view mode)
            debug.draw(agent)                           # Agent or OrbitCamera
            debug.draw(view=view_mat, proj=proj_mat)    # explicit matrices
        """
        if not self.enabled or not self._items:
            return

        self._ensure_renderer()

        if context_or_pov is not None:
            if hasattr(context_or_pov, 'view_mode'):

                ctx = context_or_pov
                from graphics.utils import ViewMode
                if ctx.view_mode == ViewMode.third_person and ctx.observer is not None:
                    ctx.observer.ratio = ctx.window_size[0] / ctx.window_size[1]
                    ctx.observer.update()
                    pov = ctx.observer
                else:
                    pov = ctx.agent
            else:
                pov = context_or_pov

        if pov is not None:
            view = pov.view
            proj = pov.projection

        if view is None or proj is None:
            raise ValueError("Provide the Context, a POV object (Agent / OrbitCamera), or explicit matrices (view, proj)")

        for item in self._items:
            if getattr(item, "visible", True):
                item.draw(self._renderer)

        self._renderer.flush(view, proj)

    def free(self):
        if self._renderer is not None:
            self._renderer.free()
            self._renderer = None


__all__ = [
    "DebugOverlay",
    "DebugDrawable",
    "LineObject",
    "ArrowObject",
    "MarkerObject",
    "AxesGizmo",
    "GazeArrow",
    "DirectionArrow",
    "DebugGrid",
    "LabeledPoint",
    "DebugLine",
    "DebugRay",
    "DebugPath",
    "DebugBox",
    "DebugFrustum",
]