import numpy as np


def checkerboard_texture(width, height, block_size=1, ratio=0.5):
    low_res_w = width // block_size
    low_res_h = height // block_size
    random_grid = np.random.random((low_res_w, low_res_h))
    small_pattern = (random_grid < ratio).astype(np.uint8) * 255
    pattern = np.repeat(np.repeat(small_pattern, block_size, axis=0), block_size, axis=1)
    return pattern.astype(np.uint8)
