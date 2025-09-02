import numpy as np

def get_rotation_matrix(angle_rad, axis):
    """
    Calculates a rotation matrix for a given angle and axis
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    one_minus_cos_a = 1 - cos_a

    return np.array([
        [cos_a + one_minus_cos_a * x ** 2, one_minus_cos_a * x * y - sin_a * z, one_minus_cos_a * x * z + sin_a * y],
        [one_minus_cos_a * y * x + sin_a * z, cos_a + one_minus_cos_a * y ** 2, one_minus_cos_a * y * z - sin_a * x],
        [one_minus_cos_a * z * x - sin_a * y, one_minus_cos_a * z * y + sin_a * x, cos_a + one_minus_cos_a * z ** 2]
    ])


def stereographic_to_spherical(x, y):
    """
    Converts 2D stereographic coordinates to 3D spherical coordinates (long, lat)
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    colatitude = 2 * np.arctan(rho / 2.0)
    latitude = np.pi / 2.0 - colatitude
    longitude = theta
    return longitude, latitude


def spherical_to_cartesian(longitude, latitude):
    """
    Converts spherical coordinates (long, lat) to 3D cartesian vectors
    """
    colatitude = np.pi / 2.0 - latitude
    x = np.sin(colatitude) * np.cos(longitude)
    y = np.sin(colatitude) * np.sin(longitude)
    z = np.cos(colatitude)
    return np.vstack([x, y, z]).T


##

PLOT = True

print("Generating Drosophila Eye Model (from Buchner 1971 data)")

ommatidia_positions = np.genfromtxt('stuff/biological_data/buchner1971_xy.csv', delimiter=',')[1:, :]

h_long, h_lat = stereographic_to_spherical(ommatidia_positions[:, 0], ommatidia_positions[:, 1])

# Transformation matrix to go from Heisenberg's coordinate system to standard
m_forward = get_rotation_matrix(-np.pi / 2, (1, 0, 0))
scale_matrix = np.diag([1, 1, -1])
m_forward = m_forward @ scale_matrix
m_reverse = np.linalg.inv(m_forward)

# Transform to cartesian vectors, apply rotation, then back to spherical to get final angles
heisenberg_coords = spherical_to_cartesian(h_long, h_lat)
left_eye_dirs = heisenberg_coords @ m_reverse.T

# Create the right eye by mirroring the X-coordinate
right_eye_dirs = left_eye_dirs.copy()
right_eye_dirs[:, 1] *= -1
print("Created right eye by mirroring the left eye.")

# Define plausible origins for the eyes
eye_radius = 0.00035        # 0.35 mm radius
eye_separation = 0.0003     # 0.3 mm separation

# The original data is for the left eye
left_eye_origins = left_eye_dirs * eye_radius - np.array([eye_separation / 2, 0, 0])
right_eye_origins = right_eye_dirs * eye_radius + np.array([eye_separation / 2, 0, 0])

both_eyes_origs = np.concatenate((right_eye_origins, left_eye_origins))
both_eyes_dirs = np.concatenate((right_eye_dirs, left_eye_dirs))

# Save as npz
np.savez_compressed(
    "drosophila_eye.npz",
    directions=both_eyes_origs,
    origins=both_eyes_dirs
    # acceptance_angles_rad could be added here
)

if PLOT:
    import matplotlib.pyplot as plt
    from PIL import Image

    IMAGE_FILENAME = 'stuff/biological_data/drosophila_buchner1971.bmp'

    image = Image.open(IMAGE_FILENAME)
    image_array = np.array(image)

    # These values are straight from Andrew Straw's trace_buchner_1971.py code
    aspect_ratio = image.size[0] / image.size[1]
    center = (0.7247, 0.5023)
    im_scale = (2 / 0.4268, 2 / 0.4248)

    im_extent = (im_scale[0] * (aspect_ratio - center[0]),
                 im_scale[0] * (0 - center[0]),
                 im_scale[1] * (0 - center[1]),
                 im_scale[1] * (1 - center[1]))

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(image, origin='upper', extent=im_extent, aspect='equal', cmap='gray')

    ax.scatter(ommatidia_positions[:, 0], ommatidia_positions[:, 1], s=40, alpha=0.8, facecolors='none', edgecolors='red',
               label='Digitized')

    ax.set_title("Verification of Digitized Drosophila Eye Data", fontsize=16)
    ax.set_xlabel("Stereographic X Coordinate", fontsize=12)
    ax.set_ylabel("Stereographic Y Coordinate", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlim(im_extent[0], im_extent[1])
    ax.set_ylim(im_extent[2], im_extent[3])

    plt.show()

print("\nDrosophila eye models generated successfully!")
