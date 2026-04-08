import numpy as np


def checkerboard_texture(width, height, block_size=1, ratio=0.5):
    """
    Generate a full contrast chequerboard pattern.

    Parameters:
        width (int): Texture width (pixels)
        height (int): Texture height (pixels)
        block_size (float): Size of each block
        ratio (float): Ratio of black vs. white squares
    """

    low_res_w = width // block_size
    low_res_h = height // block_size

    random_grid = np.random.random((low_res_w, low_res_h))
    small_pattern = (random_grid < ratio).astype(np.uint8) * 255

    pattern = np.repeat(np.repeat(small_pattern, block_size, axis=0), block_size, axis=1)

    return pattern.astype(np.uint8)


def grating_texture(width, height, num_bands=18, orientation='vertical', angle=None, wave_type='square'):
    """
    Generate a full contrast grating texture.

    Parameters:
        width (int): Texture width (pixels)
        height (int): Texture height (pixels)
        num_bands (float): Number of repeating periods in the texture
        orientation (str): 'vertical' or 'horizontal' (ignored if angle is provided)
        angle (float, optional): Angle of the bands in degrees (0 = vertical bands).
                                 Overrides `orientation` if provided.
        wave_type (str): 'square' (hard black/white bands) or 'sine' (smooth gradient)
    """
    if angle is not None:
        theta = np.deg2rad(angle)

        # We base spatial frequency on the width so the thickness of the bars
        # remains physically constant regardless of the angle
        freq = num_bands / width

        # Create a 2D coordinate grid for every pixel
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        # Calculate the phase at each pixel based on the rotated wave vector
        coords = 2 * np.pi * freq * (X * np.cos(theta) + Y * np.sin(theta))

        if wave_type == 'square':
            pattern = (np.sin(coords) > 0).astype(np.uint8) * 255
        elif wave_type == 'sine':
            pattern = ((np.sin(coords) + 1.0) * 127.5).astype(np.uint8)
        else:
            raise ValueError("wave_type must be 'square' or 'sine'")

        return pattern

    else:
        pattern = np.zeros((height, width), dtype=np.uint8)

        if orientation == 'vertical':
            coords = np.linspace(0, num_bands * 2 * np.pi, width, endpoint=False)
        elif orientation == 'horizontal':
            coords = np.linspace(0, num_bands * 2 * np.pi, height, endpoint=False)
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        if wave_type == 'square':
            wave_1d = (np.sin(coords) > 0).astype(np.uint8) * 255
        elif wave_type == 'sine':
            wave_1d = ((np.sin(coords) + 1.0) * 127.5).astype(np.uint8)
        else:
            raise ValueError("wave_type must be 'square' or 'sine'")

        if orientation == 'vertical':
            pattern[:] = wave_1d
        else:
            pattern[:] = wave_1d[:, np.newaxis]

        return pattern


def chirp_texture(resolution: int, f_start: float, f_end: float, phase: float, angle_deg: float) -> np.ndarray:
    """Generates a high-res grating where spacing gets progressively smaller (chirp)."""

    theta = np.deg2rad(angle_deg)

    # Coordinate grid centered on 0
    x = np.linspace(-0.5, 0.5, resolution)
    X, Y = np.meshgrid(x, x)

    # Rotate coordinates
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    u = X_rot + 0.5

    chirp_phase = f_start * u + 0.5 * (f_end - f_start) * (u ** 2)
    total_phase = chirp_phase + phase
    pattern = (np.sin(2 * np.pi * total_phase) > 0).astype(np.uint8) * 255

    return pattern
