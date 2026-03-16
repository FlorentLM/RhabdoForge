import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator
from species_models.bee_Sturzl2010.plots_Sturzl2010 import plot_eye_zones, plot_ortho_projection, plot_receptive_fields
from species_models.plots import plot_eyes_3d
from insectvision.utils.math import spherical_to_cartesian

# Exact replica of model from Stürlz et al., 2010 (10.1088/1748-3182/5/3/036002) with smooth boundary interpolation.
# Azimuth values taken from https://github.com/BioroboticsLab/bee_view/blob/master/data/azimuth_max.csv
# Original code from Polster, 2017 (bachelor thesis): https://github.com/BioroboticsLab/bee_view/blob/master/data/calc_ommatidial_array.R
# Reproduces figures from Stürlz et al., 2010


IOA_H_MIN = 2.4
IOA_H_MID = 3.7
IOA_H_MAX = 4.6
IOA_V_MIN = 1.5
IOA_V_MAX = 4.5

# TODO: Lookup more accurate values
# Honeybee head ~4.5 mm, compound eye diameter ~2.5-3 mm?
# Eye radius ~ 1.25 mm?
BEE_EYE_RADIUS = 1.25

# Eye separation (center-to-center distance)
BEE_EYE_SEPARATION = 3.0  # mm


##


def akima_interpolator(x, y, fill_value: float):
    """
    Akima interpolator that returns `fill_value` for queries outside range.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    akima_fn = Akima1DInterpolator(x, y)

    def wrapper(query_x):
        query_x = np.asarray(query_x)
        mask_oob = (query_x < x.min()) | (query_x > x.max())
        vals = akima_fn(query_x)
        return np.where(mask_oob, fill_value, vals)

    return wrapper


def load_azimuth_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', encoding="utf-8")[1:]

    zone_12 = data[:, :2]
    zone_34 = data[:, 2:]

    return zone_12, zone_34


def get_azimuth_delta(ioa_h, elevation):

    ioa_h_rad = np.radians(ioa_h)
    elevation_rad = np.radians(elevation)

    cos_elev = np.cos(elevation_rad)

    # Avoid div by zero at the poles (if cos_elev is near 0, delta is undefined/180)
    valid_mask = np.abs(cos_elev) > 1e-9

    cos_elev = np.where(valid_mask, cos_elev, 1.0)

    arg = np.sin(ioa_h_rad / 2.0) / cos_elev
    arg = np.clip(arg, -1.0, 1.0) # clip arcsin to [-1, 1] to avoid errors near poles

    delta = np.degrees(2 * np.arcsin(arg))
    delta = np.where(valid_mask, delta, 180.0)

    return delta


def get_horizontal_IOA(a, e):

    a = np.asarray(a)
    e = np.asarray(e)
    abs_e = np.abs(e)
    abs_a = np.abs(a)

    factor_e_pos = np.where(abs_e > 50, (90 - abs_e) / 40.0, 1.0)
    factor_e_neg = np.where((e > -50) & (e < 50), 1.0, (90 - e) / 40.0)

    # Conditions for positive azimuth (a >= 0)
    cond_p1 = (a >= 0) & (a <= 45)
    val_p1 = IOA_H_MID + (a / 45.0) * (IOA_H_MIN - IOA_H_MID)

    cond_p2 = (a > 45) & (a <= 90)
    val_p2 = IOA_H_MIN + ((a - 45.0) / 45.0) * (IOA_H_MID - IOA_H_MIN)

    cond_p3 = (a > 90) & (a <= 150)
    val_p3 = IOA_H_MID + ((a - 90.0) / 60.0) * (IOA_H_MAX - IOA_H_MID) * factor_e_pos

    cond_p4 = (a > 150) & (a <= 180)
    val_p4 = np.where(abs_e > 50, IOA_H_MID + (IOA_H_MAX - IOA_H_MID) * factor_e_pos, IOA_H_MAX)

    cond_p5 = (a > 180) & (a <= 270)
    val_p5 = IOA_H_MAX

    # Conditions for negative azimuth (a < 0)
    cond_n1 = (a >= -45) & (a < 0)
    val_n1 = IOA_H_MID + (abs_a / 45.0) * (IOA_H_MAX - IOA_H_MID) * factor_e_neg

    cond_n2 = (a >= -90) & (a < -45)
    factor_n2 = np.where(e <= -50, (90 - abs_e) / 40.0, 1.0)
    val_n2 = IOA_H_MID + ((abs_a - 45.0) / 45.0) * (IOA_H_MAX - IOA_H_MID) * factor_n2

    # Choose
    conditions = [cond_p1, cond_p2, cond_p3, cond_p4, cond_p5, cond_n1, cond_n2]
    choices = [val_p1, val_p2, val_p3, val_p4, val_p5, val_n1, val_n2]

    return np.select(conditions, choices, default=IOA_H_MID)  # default mostly for a < -90


def get_vertical_IOA(e):
    return IOA_V_MIN + (IOA_V_MAX - IOA_V_MIN) * (np.abs(e) / 90.0)


def generate_zone(zone, az_fn_12, az_fn_34, eye_factor=1.1, packing_f=np.sqrt(2) / 2.0):
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
        row_offset = get_azimuth_delta(IOA_H_MID, e) / 2.0
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
            ioa_h = get_horizontal_IOA(a, e)
            ioa_v = get_vertical_IOA(e)

            delta_rho = eye_factor * np.sqrt(ioa_h * ioa_v)

            zone_ommatidia.append((a, e, ioa_h, ioa_v, delta_rho))

            delta_a = get_azimuth_delta(ioa_h, e)

            # Stop if step is negligible (pole convergence)
            if delta_a < 1e-6:
                break

            a += delta_a * sign_a

        # Move to next elevation
        e += sign_e * get_vertical_IOA(e) * packing_f
        is_odd = not is_odd

    return zone_ommatidia


def get_interp(zone_12, zone_34, interp='akima'):

    if interp == 'akima':
        interp_fn_12 = akima_interpolator(*zone_12.T, fill_value=300.0)
        interp_fn_34 = akima_interpolator(*zone_34.T, fill_value=-200.0)
    else:
        interp_fn_12 = interp1d(*zone_12.T, kind=interp, bounds_error=False, fill_value=300.0)
        interp_fn_34 = interp1d(*zone_34.T, kind=interp, bounds_error=False, fill_value=-200.0)

    return interp_fn_12, interp_fn_34


def build_eye(interp_fn_12, interp_fn_34, eye_factor=1.1, packing_f=np.sqrt(2) / 2.0):
    """
    Build a single eye in internal coordinate system (the right one).

    Returns:
        directions: (N, 2) array of (azimuth, elevation) in degrees
        interommatidial_angles: (N, 2) array of (horizontal_IOA, vertical_IOA) in degrees
        acceptance_angles: (N,) array of acceptance angles in degrees
    """
    ommatidia_data = np.concatenate([
        generate_zone(z, interp_fn_12, interp_fn_34, eye_factor=eye_factor, packing_f=packing_f)
        for z in [1, 2, 3, 4]
    ])

    # Dedup (round for floating point error)
    _, unique_indices = np.unique(ommatidia_data[:, :2].round(decimals=3), axis=0, return_index=True)
    ommatidia_data = ommatidia_data[unique_indices]

    directions = ommatidia_data[:, :2]
    interommatidial_angles = ommatidia_data[:, 2:4]
    acceptance_angles = ommatidia_data[:, 4]

    return directions, interommatidial_angles, acceptance_angles


def generate_eyes(right_eye_dirs):
    """
    Generate both eyes in OpenGL coordinate system (X=right, Y=up, Z=back).
    """

    pts_internal = spherical_to_cartesian(
        right_eye_dirs[:, 0],
        right_eye_dirs[:, 1],
        degrees=True
    ).T

    def to_opengl(coords):
        x_int, y_int, z_int = coords[:, 0], coords[:, 1], coords[:, 2]
        return np.stack([y_int, z_int, -x_int], axis=1)

    r_dir = to_opengl(pts_internal)
    r_dir /= np.linalg.norm(r_dir, axis=1, keepdims=True)
    r_ori = r_dir * BEE_EYE_RADIUS

    # Offset right eye along X (OpenGL)
    offset = BEE_EYE_SEPARATION / 2
    r_ori[:, 0] += offset

    #Generate left eye by mirroring right eye
    l_ori = r_ori.copy()
    l_ori[:, 0] *= -1

    l_dir = r_dir.copy()
    l_dir[:, 0] *= -1

    directions = np.vstack([r_dir, l_dir])
    positions = np.vstack([r_ori, l_ori])
    eye_id = np.concatenate([np.ones(len(r_dir)), np.zeros(len(l_dir))])

    return directions, positions, eye_id


if __name__ == "__main__":

    PLOT_EYES = True
    REPRODUCE_PAPERS_PLOT = False

    file_path = "species_models/bee_Sturzl2010/sturzl2010_azimuth_max.csv"

    zone_12, zone_34 = load_azimuth_data(file_path)
    interp_fn_12, interp_fn_34 = get_interp(zone_12, zone_34, interp='akima')

    # Ommatidia packing seems to be inconsistent in various figures of the paper:
    # packing_f = 0.5  # significant overlap, needed to get ~5420 om, which is what the paper says it generates
    packing_f = np.sqrt(2) / 2.0  # ~ 45 degree lattice, generates ~3840 om, and is what the paper shows in Fig. 10
    # packing_f = np.sqrt(3) / 2.0    # ~ 60 degree lattice (hexagon), generates ~3150 om

    right_eye_dirs, right_eye_ioas, right_eye_acceptance = build_eye(
        interp_fn_12, interp_fn_34,
        eye_factor=1.1,  # parameter p
        packing_f=packing_f
    )

    if REPRODUCE_PAPERS_PLOT:
        plot_eye_zones(right_eye_dirs, interp_fn_12, interp_fn_34, zone_12, zone_34)  # Fig. 7
        plot_ortho_projection(right_eye_dirs)  # Fig. 8
        plot_receptive_fields(right_eye_dirs, right_eye_acceptance)  # Fig. 10

    # Create the other eye
    directions, positions, eye_id = generate_eyes(right_eye_dirs)

    output_filename = "species_models/bee_Sturzl.npz"
    np.savez_compressed(
        output_filename,
        directions=directions,
        positions=positions,
        eye_id=eye_id
    )

    if PLOT_EYES:
        plot_eyes_3d(
            positions,
            directions,
            eye_id,
            title='Bee eyes (from Stürzl et al., 2010)'
        )