from typing import Optional
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


# TODO: Move these somewhere else maybe

def aces_tonemap(x):
    """
    ACES tonemapping curve approx
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def exr_to_cubemap(input_path, output_size: Optional[int] = None, exposure=1.0, contrast=1.05, fmt='jpg'):

    from pathlib import Path
    import cv2
    import os

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    input_path = Path(input_path)

    # Load EXR
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
    if img is None:
        print(f"Error: Could not load {input_path}")
        return

    # Define the 6 faces and their rotations
    faces = ['right', 'left', 'top', 'bottom', 'front', 'back']

    output_folder = Path('assets/textures') / input_path.stem
    output_folder.mkdir(exist_ok=True, parents=True)

    fmt = fmt.strip('.').lower()
    output_size = output_size or min(img.shape[:2])

    print(f'Converting {input_path.name} ({img.shape[1]}x{img.shape[0]}) to {output_size}x{output_size} {fmt} cubemap...')

    for i, face in enumerate(faces):

        # Create coordinates for the cube face
        grid = np.indices((output_size, output_size), dtype=np.float32)
        y, x = grid[0], grid[1]

        xx = 2.0 * x / (output_size - 1) - 1.0
        yy = 2.0 * y / (output_size - 1) - 1.0

        if face == 'right':  # Right
            vx, vy, vz = np.ones_like(xx), -yy, -xx
        elif face == 'left':  # Left
            vx, vy, vz = -np.ones_like(xx), -yy, xx
        elif face == 'top':  # Top
            vx, vy, vz = xx, np.ones_like(xx), yy
        elif face == 'bottom':  # Bottom
            vx, vy, vz = xx, -np.ones_like(xx), -yy
        elif face == 'front':  # Front
            vx, vy, vz = xx, -yy, np.ones_like(xx)
        elif face == 'back':  # Back
            vx, vy, vz = -xx, -yy, -np.ones_like(xx)

        # To spherical coordinates
        mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        vx, vy, vz = vx / mag, vy / mag, vz / mag

        phi = np.arctan2(vx, vz)
        theta = np.arcsin(vy)

        # Back to equirect UV
        out_x = (phi / (2 * np.pi) + 0.5) * (img.shape[1] - 1)
        out_y = (0.5 - theta / np.pi) * (img.shape[0] - 1)

        face_img = cv2.remap(img, out_x, out_y, cv2.INTER_LINEAR)

        # Tonemapping & Gamma correction (HDR -> LDR)

        # Exposure
        face_img = face_img * exposure

        # ACES tonemapping
        face_img = aces_tonemap(face_img)

        # Contrast adjust (S curve)
        face_img = np.clip((face_img - 0.5) * contrast + 0.5, 0, 1)

        # Gamma (linear to sRGB)
        face_img = np.power(face_img, 1.0 / 2.2)

        face_img = (np.clip(face_img, 0, 1) * 255).astype(np.uint8)

        cv2.imwrite((output_folder / face).with_suffix('.' + fmt), face_img)

    print('Done.')
