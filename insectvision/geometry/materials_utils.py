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


def grating_texture(width, height, num_bands=18, orientation='vertical', wave_type='square'):
    """
    Generate a full contrast grating texture.

    Parameters:
        width (int): Texture width (pixels)
        height (int): Texture height (pixels)
        num_bands (float): Number of repeating periods in the texture
        orientation (str): 'vertical' or 'horizontal'
        wave_type (str): 'square' (hard black/white bands) or 'sine' (smooth gradient)
    """
    pattern = np.zeros((height, width), dtype=np.uint8)

    if orientation == 'vertical':
        coords = np.linspace(0, num_bands * 2 * np.pi, width, endpoint=False)

    elif orientation == 'horizontal':
        coords = np.linspace(0, num_bands * 2 * np.pi, height, endpoint=False)

    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # 1D wave pattern
    if wave_type == 'square':
        wave_1d = (np.sin(coords) > 0).astype(np.uint8) * 255

    elif wave_type == 'sine':
        # map sine values from [-1, 1] to [0, 255]
        wave_1d = ((np.sin(coords) + 1.0) * 127.5).astype(np.uint8)

    else:
        raise ValueError("wave_type must be 'square' or 'sine'")

    if orientation == 'vertical':
        pattern[:] = wave_1d
    else:
        pattern[:] = wave_1d[:, np.newaxis]

    return pattern