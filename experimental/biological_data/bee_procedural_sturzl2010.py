import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Akima1DInterpolator
import matplotlib.pyplot as plt
import time


# Loaders

def akima_interpolator(x, y, fill_value):
    """
    Creates an Akima interpolator that returns a specific fill_value for queries outside the original range
    (prevents premature termination of the generation loop at the poles)
    """

    akima_fn = Akima1DInterpolator(x, y)

    min_x, max_x = x.min(), x.max()

    # Wrapper function that enforces boundary condition
    def wrapper(query_x):
        query_x = np.asarray(query_x)
        interpolated_y = akima_fn(query_x)
        # Substitute the fill value for values outside the range
        interpolated_y[query_x < min_x] = fill_value
        interpolated_y[query_x > max_x] = fill_value
        return interpolated_y

    return wrapper


def load_azimuth_data(file_path="experimental/biological_data/azimuth_max.csv", interp='akima'):
    """ Loads and prepares the azimuth boundary data using the robust interpolator """

    try:
        azimuth_df = pd.read_csv(file_path, encoding="utf-8").dropna()
    except FileNotFoundError:
        print(f"Error: Azimuth data file not found at '{file_path}'")
        return None, None, None

    # Sort data by elevation
    df_12 = azimuth_df[['elevation_1_2', 'azimuth_max_1_2']].dropna().sort_values('elevation_1_2')
    df_34 = azimuth_df[['elevation_3_4', 'azimuth_max_3_4']].dropna().sort_values('elevation_3_4')

    if interp == 'akima':
        azimuth_max_fn_12 = akima_interpolator(df_12['elevation_1_2'].values,
                                               df_12['azimuth_max_1_2'].values,
                                               fill_value=300.0)
        azimuth_max_fn_34 = akima_interpolator(df_34['elevation_3_4'].values,
                                               df_34['azimuth_max_3_4'].values,
                                               fill_value=-200.0)
    else:
        azimuth_max_fn_12 = interp1d(df_12['elevation_1_2'].values, df_12['azimuth_max_1_2'].values,
                                     kind=interp, bounds_error=False, fill_value=300.0)
        azimuth_max_fn_34 = interp1d(df_34['elevation_3_4'].values, df_34['azimuth_max_3_4'].values,
                                     kind=interp, bounds_error=False, fill_value=-200.0)

    return azimuth_max_fn_12, azimuth_max_fn_34, (df_12, df_34)


# Helpers

IOA_H_MIN = 2.4
IOA_H_MID = 3.7
IOA_H_MAX = 4.6
IOA_V_MIN = 1.5
IOA_V_MAX = 4.5
DEG2RAD = np.pi / 180


def ioa_h_to_azimuth_delta(ioa_h, elevation):
    """ Calculates the change in azimuth in degrees """
    if abs(np.cos(elevation * DEG2RAD)) < 1e-9:
        return 180.0
    return 2 * np.arcsin(np.sin((ioa_h / 2) * DEG2RAD) / np.cos(elevation * DEG2RAD)) / DEG2RAD


def get_ioa_h_scalar(a, e):
    """ Scalar version of the horizontal IOA calculation """

    abs_e = abs(e)
    if a >= 0:
        if a <= 45: return IOA_H_MID + (a / 45) * (IOA_H_MIN - IOA_H_MID)
        if a <= 90: return IOA_H_MIN + ((a - 45) / 45) * (IOA_H_MID - IOA_H_MIN)
        factor_e = (90 - abs_e) / 40 if abs_e > 50 else 1.0
        if a <= 150: return IOA_H_MID + ((a - 90) / 60) * (IOA_H_MAX - IOA_H_MID) * factor_e
        if a <= 180: return IOA_H_MID + (IOA_H_MAX - IOA_H_MID) * factor_e if abs_e > 50 else IOA_H_MAX
        if a <= 270: return IOA_H_MAX
    else:
        abs_a = abs(a)
        if a >= -45:
            factor_e = 1.0 if abs_e < 50 and e > -50 else (90 - e) / 40
            return IOA_H_MID + (abs_a / 45) * (IOA_H_MAX - IOA_H_MID) * factor_e
        if a >= -90:
            factor_e = (90 - abs_e) / 40 if e <= -50 else 1.0
            return IOA_H_MID + ((abs_a - 45) / 45) * (IOA_H_MAX - IOA_H_MID) * factor_e
    return IOA_H_MID


def get_ioa_v(e):
    return IOA_V_MIN + (IOA_V_MAX - IOA_V_MIN) * (abs(e) / 90)


# Core ommatidia generation logic

def _generate_zone(zone, az_fn_12, az_fn_34):
    """ Generates ommatidia for a specific zone """

    zone_ommatidia = []

    # Zone-specific growth directions and boundaries
    sign_a = 1 if zone in [1, 2] else -1
    sign_e = 1 if zone in [1, 4] else -1
    boundary_fn = az_fn_12 if zone in [1, 2] else az_fn_34

    e = 0.0
    is_odd_row = False

    # Grow outwards from equator until pole is reached
    while abs(e) <= 90:
        # Stagger rows
        a = (ioa_h_to_azimuth_delta(IOA_H_MID, e) / 2.0) * sign_a if is_odd_row else 0.0

        # Condition for inner loop to continue growth
        def is_in_bounds(current_a, current_e):
            if abs(current_e) > 88.9: return False
            az_boundary = boundary_fn(current_e)
            if zone in [1, 2]:  # Grow right
                return current_a < min(az_boundary, 270)
            else:  # Grow left
                return current_a > max(az_boundary, -90)

        # Generate a single row for this zone
        while is_in_bounds(a, e):
            zone_ommatidia.append((a, e))
            delta_a = ioa_h_to_azimuth_delta(get_ioa_h_scalar(a, e), e)
            if delta_a < 1e-6: break  # Prevent infinite loop
            a += delta_a * sign_a

        # Move to the next elevation using the CORRECT step size
        e += sign_e * get_ioa_v(e) / 2.0
        is_odd_row = not is_odd_row

    return zone_ommatidia


def generate_eye_model(az_fn_12, az_fn_34):
    """ Combining the four zones into a full eye """

    # Generate each of the four zones independently
    om_z1 = _generate_zone(1, az_fn_12, az_fn_34)
    om_z2 = _generate_zone(2, az_fn_12, az_fn_34)
    om_z3 = _generate_zone(3, az_fn_12, az_fn_34)
    om_z4 = _generate_zone(4, az_fn_12, az_fn_34)

    all_ommatidia_list = om_z1 + om_z2 + om_z3 + om_z4

    ommatidia_array = np.array(all_ommatidia_list)

    # Remove duplicates
    ommatidia_array = np.unique(ommatidia_array.round(decimals=3), axis=0)
    return ommatidia_array

##

def main():

    start_time = time.time()

    az_fn_12, az_fn_34, raw_dfs = load_azimuth_data(interp='akima')
    if az_fn_12 is None: return

    ommatidia = generate_eye_model(az_fn_12, az_fn_34)
    end_time = time.time()

    print(f"Generated {len(ommatidia)} unique ommatidia in {end_time - start_time:.4f} seconds.")

    output_file = "ommatidia_sturzl_2010.csv"
    np.savetxt(output_file, ommatidia, delimiter=",", header="azimuth,elevation", fmt="%.6f", comments='')
    print(f"Saved ommatidia data to '{output_file}'")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot generated ommatidia
    plt.scatter(ommatidia[:, 0], ommatidia[:, 1], color='darkblue', alpha=0.4, s=2)

    # Plot interpolated boundaries
    elevations = np.linspace(-90, 90, 500)
    plt.plot(az_fn_12(elevations), elevations, color='orangered', lw=2, label="Interpolated boundary (1-2)")
    plt.plot(az_fn_34(elevations), elevations, color='limegreen', lw=2, label="Interpolated boundary (3-4)")

    # Plot the original experimental data points
    df_12, df_34 = raw_dfs
    plt.plot(df_12['azimuth_max_1_2'], df_12['elevation_1_2'], 'o', color='darkred', markersize=5,
             label="Raw data (1-2)")
    plt.plot(df_34['azimuth_max_3_4'], df_34['elevation_3_4'], 'o', color='darkgreen', markersize=5,
             label="Raw data (3-4)")

    plt.xlim(-90, 270)
    plt.ylim(-90, 90)
    plt.xlabel("Azimuth α (degrees)")
    plt.ylabel("Elevation ε (degrees)")
    plt.title(f"Viewing Directions of {len(ommatidia)} Procedurally Generated Bee Ommatidia (Stürzl et al. 2010 model)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()