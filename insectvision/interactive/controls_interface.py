from abc import ABC, abstractmethod


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
