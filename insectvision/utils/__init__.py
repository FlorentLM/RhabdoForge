from enum import IntEnum


class EyeOutput(IntEnum):
    """
    Eye colours output mode (for visualisation only)
    """
    Raw = 0          # Render receptors individually (scaled down)
    Ommatidium = 1   # One tile per lens (averaging R1-R8)
    Cartridge = 2    # One tile per lens (averaging optically superimposed receptors)


class OmmatidiaProjection(IntEnum):
    """
    Eye rendering projection (acceptance vs. lens position)
    """
    Position = 0     # Positions on the curved eye surface
    OpticalAxis = 1  # Positions from optical axis directions


class Colormap(IntEnum):
    """
    Colours for the overlay view (heatmap)
    """
    Diverging = 0    # Blue, white, red (signed and centred on zero)
    Sequential = 1   # Viridis-like
    Thermal = 2      # Black, red, white


class DisplayMode(IntEnum):
    """
    Camera mode
    """
    Compound = 0
    Panoramic = 1
    Third_person = 2
    Perspective = 3


class RandomnessMode(IntEnum):
    Pseudo = 0      # Standard PCG White Noise
    Halton = 1      # Quasi-random low-discrepancy
    Stratified = 2  # Grid-based jittered sampling


class SamplingMode(IntEnum):
    Gaussian = 0    # Default approximation
    Airy = 1        # Physical diffraction pattern