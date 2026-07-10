from typing import Sequence, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path


def checkerboard_texture(width: int, height: int, block_size: int = 1, ratio: float = 0.5) -> np.ndarray:
    """
    Generate a full contrast chequerboard pattern.

    Args:
        - width (int): Texture width (pixels)
        - height (int): Texture height (pixels)
        - block_size (float): Size of each block
        - ratio (float): Ratio of black vs. white squares
    """

    low_res_w = width // block_size
    low_res_h = height // block_size

    random_grid = np.random.random((low_res_w, low_res_h))
    small_pattern = (random_grid < ratio).astype(np.uint8) * 255

    pattern = np.repeat(np.repeat(small_pattern, block_size, axis=0), block_size, axis=1)

    return pattern.astype(np.uint8)


def grating_texture(width: int, height: int, num_bands: int = 18, orientation: str='vertical', angle: Optional[float] = None, wave_type: str = 'square') -> np.ndarray:
    """
    Generate a full contrast grating texture.

    Args:
        - width (int): Texture width (pixels)
        - height (int): Texture height (pixels)
        - num_bands (float): Number of repeating periods in the texture
        - orientation (str): 'vertical' or 'horizontal' (ignored if angle is provided)
        - angle (float, optional): Angle of the bands in degrees (0 = vertical bands). Overrides `orientation` if provided.
        - wave_type (str): 'square' (hard black/white bands) or 'sine' (smooth gradient)
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


def load_exr_equirect(input_path: str | Path, max_height: int = 2048) -> np.ndarray:
    """
    Load an HDR equirectangular EXR as a linear float32 RGB array (H, W, 3). No tonemapping.
    """

    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    img = cv2.imread(str(Path(input_path)), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load EXR: {input_path}")

    img = img[:, :, :3][:, :, ::-1]  # BGR -> RGB
    if not np.issubdtype(img.dtype, np.floating):  # LDR (8/16-bit) source
        img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
        img = np.power(img, 2.2)  # sRGB -> linear (approx)
    else:
        img = img.astype(np.float32)  # EXR/HDR already linear

    img = np.ascontiguousarray(img, dtype=np.float32)   # just to be sure

    MAX_RADIANCE = 1.0e4
    img = np.nan_to_num(img, nan=0.0, posinf=MAX_RADIANCE, neginf=0.0)
    img = np.clip(img, 0.0, MAX_RADIANCE).astype(np.float32)

    h, w = img.shape[:2]
    if h > max_height:
        new_h = max_height
        new_w = int(round(w * new_h / h))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img, dtype=np.float32)

    return img


def sh_irradiance(equirect_rgb: ArrayLike) -> np.ndarray:
    """
    Project an HDR equirect (H, W, 3, linear) onto 9 SH coeffs scaled for diffuse irradiance.
    Returns (9, 3) float32 array such that  irradiance(N) = sum_i c_i * Y_i(N)  (multiply by albedo).
    """

    equirect_rgb = np.asarray(equirect_rgb, dtype=np.float32)
    H, W = equirect_rgb.shape[:2]

    # Coordinates generation
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    phi = (u - 0.5) * (2.0 * np.pi)
    el = (0.5 - v) * np.pi

    cos_el = np.cos(el)
    sin_el = np.sin(el)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Weight the image by the solid angle (dw)
    dw = cos_el * ((2.0 * np.pi / W) * (np.pi / H))
    img_weighted = (equirect_rgb * dw[:, None, None]).reshape(-1, 3)

    X = np.outer(cos_el, sin_phi).ravel()
    Y = np.repeat(sin_el, W)
    Z = np.outer(cos_el, cos_phi).ravel()

    basis = np.empty((9, H * W), dtype=X.dtype)

    basis[0] = 0.282095
    basis[1] = 0.32573533 * Y
    basis[2] = 0.32573533 * Z
    basis[3] = 0.32573533 * X
    basis[4] = 0.27313700 * X * Y
    basis[5] = 0.27313700 * Y * Z
    basis[6] = 0.07884800 * (3.0 * Z * Z - 1.0)
    basis[7] = 0.27313700 * X * Z
    basis[8] = 0.13656850 * (X * X - Y * Y)

    coeffs = basis @ img_weighted

    return coeffs.astype(np.float32)


def constant_sh(bg_color: Sequence[float]) -> np.ndarray:
    c = np.zeros((9, 3), np.float32)
    c[0] = np.power(np.asarray(bg_color, np.float32), 2.2) / 0.282095  # linear bg as constant irradiance
    return c


def get_exr_sun(exr_path: str | Path, sun_percentile: float = 99.95)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import os
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2

    img = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)[:, :, :3][:, :, ::-1]
    img = np.nan_to_num(img.astype(np.float32), posinf=1e4, neginf=0.0)
    H, W = img.shape[:2]
    L = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    # Locate the sun
    thr = np.percentile(L, sun_percentile)
    mask = L >= thr
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()

    # Pixel -> direction
    phi = (cx / W - 0.5) * 2.0 * np.pi
    theta = (0.5 - cy / H) * np.pi
    azimuth = np.degrees(np.arctan2(np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)))
    elevation = np.degrees(np.arcsin(np.clip(np.sin(theta), -1.0, 1.0)))

    # Sun radiance -> directional-light colour & intensity (solid-angle weighted)
    sin_t = np.sin((np.arange(H) + 0.5) / H * np.pi)[:, None]      # equirect area weight
    w = mask * sin_t
    sun_rgb = (img * w[..., None]).reshape(-1, 3).sum(0) / max(w.sum(), 1e-6)

    intensity = np.max(sun_rgb)
    color = sun_rgb / max(intensity, 1e-6)
    ang_radius = np.sqrt(mask.sum() / np.pi) / H * np.pi     # rough disk radius (rad)

    return azimuth, elevation, color, intensity, ang_radius