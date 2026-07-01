"""
Exact replica of model from Stürlz et al., 2010 (10.1088/1748-3182/5/3/036002) with smooth boundary interpolation.

Azimuth values taken from https://github.com/BioroboticsLab/bee_view/blob/master/data/azimuth_max.csv
Original code from Polster, 2017 (bachelor thesis): https://github.com/BioroboticsLab/bee_view/blob/master/data/calc_ommatidial_array.R

Reproduces figures from Stürlz et al., 2010
"""
from pathlib import Path
from typing import Callable, Tuple, List

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d, Akima1DInterpolator
from insectvision.lattice_fitting.plots import plot_eyes_3d


IOA_H_MIN = 2.4
IOA_H_MID = 3.7
IOA_H_MAX = 4.6
IOA_V_MIN = 1.5
IOA_V_MAX = 4.5

# TODO: Lookup more accurate values
# Honeybee head ~4.5 mm, compound eye diameter ~2.5-3 mm?
# Eye radius ~ 1.25 mm?
BEE_EYE_RADIUS = 1250.0         # micrometres

# Eye separation (centre-to-centre distance)
BEE_EYE_SEPARATION = 3000.0     # micrometres


def spherical_to_cartesian_sturzl(
        azimuth: ArrayLike,
        elevation: ArrayLike,
        radius: float = 1.0,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Converts spherical coordinates to cartesian coordinates (in the correct frame for this script).
    """
    # TODO: unify this with the rest of the codebase

    az_rad = np.radians(azimuth) if degrees else np.array(azimuth, dtype=np.float64)
    el_rad = np.radians(elevation) if degrees else np.array(elevation, dtype=np.float64)

    x = radius * np.cos(el_rad) * np.cos(az_rad)
    y = radius * np.cos(el_rad) * np.sin(az_rad)
    z = radius * np.sin(el_rad)

    return np.stack([x, y, z])


def akima_interp_fn(x: ArrayLike, y: ArrayLike, fill_value: float) -> 'Callable':
    """
    Akima interpolator that returns 'fill_value' for queries outside range.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    akima_fn = Akima1DInterpolator(x, y)

    def wrapper(query_x):
        query_x = np.asarray(query_x)
        mask_oob = (query_x < x.min()) | (query_x > x.max())
        vals = akima_fn(query_x)
        return np.where(mask_oob, fill_value, vals)

    return wrapper


def _get_azimuth_delta(ioa_h: ArrayLike, elevation: ArrayLike, degrees: bool = True) -> np.ndarray:

    ioa_h = np.asarray(ioa_h, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)

    if degrees:
        ioa_h = np.deg2rad(ioa_h)
        elevation = np.deg2rad(elevation)

    cos_elev = np.cos(elevation)

    # if cos_elev is near 0, delta is undefined/180
    valid_mask = np.abs(cos_elev) > 1e-9

    cos_elev = np.where(valid_mask, cos_elev, 1.0)

    arg = np.sin(ioa_h / 2.0) / cos_elev
    arg = np.clip(arg, -1.0, 1.0) # clip arcsin to [-1, 1] to avoid errors near poles

    delta = 2 * np.arcsin(arg)
    delta = np.where(valid_mask, delta, np.pi)

    if degrees:
        delta = np.rad2deg(delta)

    return delta


def _get_horizontal_IOA(azim: ArrayLike, elev: ArrayLike) -> np.ndarray:

    azim = np.asarray(azim, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)
    abs_e = np.abs(elev)
    abs_a = np.abs(azim)

    factor_e_pos = np.where(abs_e > 50, (90 - abs_e) / 40.0, 1.0)
    factor_e_neg = np.where((elev > -50) & (elev < 50), 1.0, (90 - elev) / 40.0)

    # Conditions for positive azimuth (a >= 0)
    cond_p1 = (azim >= 0) & (azim <= 45)
    val_p1 = IOA_H_MID + (azim / 45.0) * (IOA_H_MIN - IOA_H_MID)

    cond_p2 = (azim > 45) & (azim <= 90)
    val_p2 = IOA_H_MIN + ((azim - 45.0) / 45.0) * (IOA_H_MID - IOA_H_MIN)

    cond_p3 = (azim > 90) & (azim <= 150)
    val_p3 = IOA_H_MID + ((azim - 90.0) / 60.0) * (IOA_H_MAX - IOA_H_MID) * factor_e_pos

    cond_p4 = (azim > 150) & (azim <= 180)
    val_p4 = np.where(abs_e > 50, IOA_H_MID + (IOA_H_MAX - IOA_H_MID) * factor_e_pos, IOA_H_MAX)

    cond_p5 = (azim > 180) & (azim <= 270)
    val_p5 = IOA_H_MAX

    # Conditions for negative azimuth (a < 0)
    cond_n1 = (azim >= -45) & (azim < 0)
    val_n1 = IOA_H_MID + (abs_a / 45.0) * (IOA_H_MAX - IOA_H_MID) * factor_e_neg

    cond_n2 = (azim >= -90) & (azim < -45)
    factor_n2 = np.where(elev <= -50, (90 - abs_e) / 40.0, 1.0)
    val_n2 = IOA_H_MID + ((abs_a - 45.0) / 45.0) * (IOA_H_MAX - IOA_H_MID) * factor_n2

    # Choose
    conditions = [cond_p1, cond_p2, cond_p3, cond_p4, cond_p5, cond_n1, cond_n2]
    choices = [val_p1, val_p2, val_p3, val_p4, val_p5, val_n1, val_n2]

    return np.select(conditions, choices, default=IOA_H_MID)  # default mostly for a < -90


def _get_vertical_IOA(elev: ArrayLike) -> float:
    return IOA_V_MIN + (IOA_V_MAX - IOA_V_MIN) * (np.abs(elev) / 90.0)


def _generate_zone(
        zone: int,
        az_fn_12: 'Callable',
        az_fn_34: 'Callable',
        eye_factor: float = 1.1,
        packing_f: float = np.sqrt(2) / 2.0
    ) -> List[np.ndarray]:
    """Generates ommatidia for a specific zone."""

    zone_ommatidia = []

    # Zone-specific growth directions and boundaries
    sign_a = 1 if zone in [1, 2] else -1
    sign_e = 1 if zone in [1, 4] else -1
    boundary_fn = az_fn_12 if zone in [1, 2] else az_fn_34

    e = 0.0
    is_odd = False

    # Grow outwards from equator until pole is reached
    while np.abs(e) <= 90:

        # Stagger rows
        row_offset = _get_azimuth_delta(IOA_H_MID, e) / 2.0
        a = row_offset * sign_a if is_odd else 0.0

        def in_bounds(current_a, current_e):
            if np.abs(current_e) > 89.99:
                return False
            az_boundary = boundary_fn(current_e)

            if zone in [1, 2]:  # grow right
                return current_a < min(az_boundary, 270)
            else:  # grow left
                return current_a > max(az_boundary, -90)

        while in_bounds(a, e):

            # IOA for current position
            ioa_h = _get_horizontal_IOA(a, e)
            ioa_v = _get_vertical_IOA(e)

            delta_rho = eye_factor * np.sqrt(ioa_h * ioa_v)

            zone_ommatidia.append((a, e, ioa_h, ioa_v, delta_rho))

            delta_a = _get_azimuth_delta(ioa_h, e)

            # Stop if step is negligible (pole convergence)
            if delta_a < 1e-6:
                break

            a += delta_a * sign_a

        # Move to next elevation
        e += sign_e * _get_vertical_IOA(e) * packing_f
        is_odd = not is_odd

    return zone_ommatidia


def reconstruct_sturzl_data(
        csv_file: str | Path,
        eye_factor: float = 1.1,
        packing_f: float = np.sqrt(2) / 2.0,
        interp: str = 'akima',
        show_plots: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a single eye in internal coordinate system (the right one).

    Returns:
        directions: (N, 2) array of (azimuth, elevation) in degrees
        interommatidial_angles: (N, 2) array of (horizontal_IOA, vertical_IOA) in degrees
        acceptance_angles: (N,) array of acceptance angles in degrees
    """

    data = np.genfromtxt(csv_file, delimiter=',', encoding="utf-8")[1:]

    zone_12 = data[:, :2]
    zone_34 = data[:, 2:]

    if interp == 'akima':
        interp_fn_12 = akima_interp_fn(*zone_12.T, fill_value=300.0)
        interp_fn_34 = akima_interp_fn(*zone_34.T, fill_value=-200.0)
    else:
        interp_fn_12 = interp1d(*zone_12.T, kind=interp, bounds_error=False, fill_value=300.0)
        interp_fn_34 = interp1d(*zone_34.T, kind=interp, bounds_error=False, fill_value=-200.0)

    ommatidia_data = np.concatenate([
        _generate_zone(z, interp_fn_12, interp_fn_34, eye_factor=eye_factor, packing_f=packing_f)
        for z in [1, 2, 3, 4]
    ])

    # Dedup (round for floating point error)
    _, unique_indices = np.unique(ommatidia_data[:, :2].round(decimals=3), axis=0, return_index=True)
    ommatidia_data = ommatidia_data[unique_indices]

    directions = ommatidia_data[:, :2]
    interommatidial_angles = ommatidia_data[:, 2:4]
    acceptance_angles = ommatidia_data[:, 4]

    if show_plots:
        from species_models.bee_Sturzl2010._sturzl_figures import fig7_eye_zones, fig8_ortho_projection, fig10_receptive_fields
        fig7_eye_zones(directions, interp_fn_12, interp_fn_34, zone_12, zone_34)
        fig8_ortho_projection(directions)
        fig10_receptive_fields(directions, acceptance_angles)

    return directions, interommatidial_angles, acceptance_angles


if __name__ == "__main__":

    SHOW_PLOTS = True

    csv_file = 'species_models/bee_Sturzl2010/sturzl2010_azimuth_max.csv'

    # Ommatidia packing seems to be inconsistent in various figures of the paper:

    # packing_f = 0.5  # significant overlap, needed to get ~5420 om, which is what the paper says it generates
    packing_f = np.sqrt(2) / 2.0  # ~ 45 degree lattice, generates ~3840 om, and is what the paper shows in Fig. 10
    # packing_f = np.sqrt(3) / 2.0    # ~ 60 degree lattice (hexagon), generates ~3150 om

    directions, ioas, acceptances = reconstruct_sturzl_data(
        csv_file=csv_file,
        eye_factor=1.1,         # eye parameter p
        packing_f=packing_f,
        show_plots=SHOW_PLOTS
    )

    # TODO: get rid of this
    def to_opengl(coords: np.ndarray) -> np.ndarray:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return np.stack([y, z, -x], axis=1)

    R_dirs = to_opengl(spherical_to_cartesian_sturzl(directions[:, 0], directions[:, 1], degrees=True).T)
    R_dirs /= np.linalg.norm(R_dirs, axis=1, keepdims=True)
    R_positions = R_dirs * BEE_EYE_RADIUS

    # Offset right eye along X
    offset = BEE_EYE_SEPARATION / 2
    R_positions[:, 0] += offset

    # Mirror for left eye
    L_positions = R_positions.copy()
    L_positions[:, 0] *= -1
    L_dirs = R_dirs.copy()
    L_dirs[:, 0] *= -1

    all_directions = np.vstack([R_dirs, L_dirs])
    all_positions = np.vstack([R_positions, L_positions])
    eye_ids = np.concatenate([np.ones(len(R_dirs)), np.zeros(len(L_dirs))])

    print(f"\nFinal model:  L={len(L_positions)}  R={len(R_positions)}")

    np.savez_compressed(
        'species_models/bee_Sturzl.npz',
        directions=all_directions,
        positions=all_positions,
        eye_id=eye_ids,
        acceptance_angles_rad=np.deg2rad(np.concatenate([acceptances, acceptances]))
    )

    if SHOW_PLOTS:
        plot_eyes_3d(
            all_positions,
            all_directions,
            eye_ids,
            title='Bee eyes (from Stürzl et al., 2010)'
        )