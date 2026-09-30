import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from contextlib import contextmanager, nullcontext
import time


class NoProfiler(nullcontext):
    def __call__(self, *_, **__):
        return self
    def tick(self) -> None:
        pass
    def free(self) -> None:
        pass


class Profiler:
    """CPU/GPU timer, prints a summary `every` seconds."""

    def __init__(self, every: float = 2.0):
        self.every = every
        self._t0 = time.perf_counter()
        self._cpu = {}
        self._gpu = {}
        self._pend = []
        self._free = []

    @contextmanager
    def __call__(self, name: str, gpu: bool = False):

        q = None
        if gpu:
            q = self._free.pop() if self._free else int(glGenQueries(1))
            glBeginQuery(GL_TIME_ELAPSED, q)

        t = time.perf_counter()
        try:
            yield
        finally:
            self._add(self._cpu, name, (time.perf_counter() - t) * 1e3)
            if q is not None:
                glEndQuery(GL_TIME_ELAPSED)
                self._pend.append((name, q))

    @staticmethod
    def _add(d: dict, name: str, ms: float) -> None:
        s = d.setdefault(name, [0.0, 0.0, 0])
        s[0] += ms
        s[1] = max(s[1], ms)
        s[2] += 1

    def tick(self) -> None:

        left = []
        for name, q in self._pend:
            if glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE):
                self._add(self._gpu, name, int(glGetQueryObjectuiv(q, GL_QUERY_RESULT)) * 1e-6)
                self._free.append(q)
            else:
                left.append((name, q))
        self._pend = left

        now = time.perf_counter()
        if now - self._t0 < self.every:
            return

        print(f'[ Profiling | cpu (avg / max) | gpu (avg / max) ]')
        for name in sorted({*self._cpu, *self._gpu}):
            row = f'  {name:<10}'
            for d in (self._cpu, self._gpu):
                s = d.get(name)
                row += f'| {s[0] / s[2]:6.2f} / {s[1]:6.2f} ' if s else '|' + ' ' * 17
            print(row)

        self._cpu.clear()
        self._gpu.clear()
        self._t0 = now

    def free(self) -> None:
        for _, q in self._pend:
            self._free.append(q)
        self._pend.clear()
        if self._free:
            glDeleteQueries(len(self._free), self._free)
            self._free.clear()
