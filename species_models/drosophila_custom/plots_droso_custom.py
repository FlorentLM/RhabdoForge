import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


def plot_buchner_3d(ommatidia_3d, stars_3d, origin_3d, axes_3d, title="Drosophila eye - Buchner, 1971"):
    """
    3D version of the Buchner data with.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*ommatidia_3d.T, c='grey', s=10, alpha=0.5, label='Ommatidia')
    ax.scatter(*stars_3d.T, c='red', s=150, marker='*', label='Forward', depthshade=False)
    ax.scatter(*origin_3d, c='black', s=50, marker='X', label='Origin', depthshade=False)

    # Plot axes
    colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
    for axis_id, points in axes_3d.items():
        ax.plot(*points.T, color=colors[axis_id], linewidth=2.5,
                label=f'Axis {axis_id[-1].upper()}')

    ax.set_title(title, fontsize=16)

    # Set up axes properly
    all_points = np.vstack([ommatidia_3d, stars_3d])
    min_coords, max_coords = all_points.min(axis=0), all_points.max(axis=0)
    center, max_range = (max_coords + min_coords) / 2, (max_coords - min_coords).max() / 2

    ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
    ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
    ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_parametric_model(ommatidia_ellipsoid, lattice_3d, stars_3d, origin_3d,
                          axes_3d, ommatidia_2d, lattice_2d, density_func,
                          nb_ommatidia, use_2d_density, grid_angle_deg, density_mult):
    """
    Comprehensive 4-panel plot for parametric model generation.
    """
    fig = plt.figure(figsize=(16, 12))

    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(*ommatidia_ellipsoid.T, c='grey', s=20, alpha=0.3, label='Real ommatidia')
    ax1.scatter(*lattice_3d.T, c='green', s=15, alpha=0.5, label='Generated ommatidia')
    ax1.scatter(*stars_3d.T, c='red', s=150, marker='*', label='Forward', depthshade=False)
    ax1.scatter(*origin_3d, c='black', s=50, marker='X', label='Lattice origin', depthshade=False)

    colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
    for axis_id, points in axes_3d.items():
        ax1.plot(*points.T, color=colors[axis_id], linewidth=2.5,
                 label=f'Axis {axis_id[-1].upper()}')

    ax1.set_title("Parametric Eye model", fontsize=14)
    ax1.legend(loc='upper right', fontsize=8)

    all_points = np.vstack([ommatidia_ellipsoid, stars_3d])
    min_coords, max_coords = all_points.min(axis=0), all_points.max(axis=0)
    center, max_range = (max_coords + min_coords) / 2, (max_coords - min_coords).max() / 2

    ax1.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
    ax1.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
    ax1.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

    ax1.set_xlim(center[0] - max_range, center[0] + max_range)
    ax1.set_ylim(center[1] - max_range, center[1] + max_range)
    ax1.set_zlim(center[2] - max_range, center[2] + max_range)
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=30, azim=45)

    # Density field visualization
    ax2 = fig.add_subplot(222)

    x_range = np.linspace(ommatidia_2d[:, 0].min() - 0.1,
                          ommatidia_2d[:, 0].max() + 0.1, 50)
    y_range = np.linspace(ommatidia_2d[:, 1].min() - 0.1,
                          ommatidia_2d[:, 1].max() + 0.1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    Z = density_func(grid_points).reshape(X.shape)

    # Plot density field
    im = ax2.contourf(X, Y, Z, levels=20, cmap='viridis_r', alpha=0.7)
    ax2.scatter(*ommatidia_2d.T, c='red', s=2, alpha=0.3, label='Real data')
    ax2.set_title('Density Field', fontsize=14)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (stereo plane)')
    ax2.set_ylabel('Y (stereo plane)')
    cbar = plt.colorbar(im, ax=ax2, label='Relative spacing')
    cbar.ax.tick_params(labelsize=8)

    # 2D lattice comparison
    ax3 = fig.add_subplot(223)
    ax3.scatter(*ommatidia_2d.T, c='grey', s=8, alpha=0.5, label='Real ommatidia')
    ax3.scatter(*lattice_2d.T, c='green', s=5, alpha=0.6, label='Generated lattice')
    ax3.set_title('Stereographic plane', fontsize=14)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X (stereo plane)')
    ax3.set_ylabel('Y (stereo plane)')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Density profile plot
    ax4 = fig.add_subplot(224)

    # radial distances and local densities
    radii = np.sqrt(ommatidia_2d[:, 0] ** 2 + ommatidia_2d[:, 1] ** 2)
    tree = cKDTree(ommatidia_2d)
    distances, _ = tree.query(ommatidia_2d, k=7)
    local_spacings = distances[:, 1:].mean(axis=1)

    # Plot actual vs fitted density
    ax4.scatter(radii, local_spacings, c='blue', s=10, alpha=0.3, label='Actual spacing')

    # Plot fitted profile at various angles
    test_radii = np.linspace(0, radii.max(), 100)
    for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
        test_points = np.column_stack([
            test_radii * np.cos(angle),
            test_radii * np.sin(angle)
        ])
        fitted_density = density_func(test_points) * local_spacings.mean()
        ax4.plot(test_radii, fitted_density, alpha=0.7,
                 label=f'Fitted at θ={np.rad2deg(angle):.0f}°')

    ax4.set_xlabel('Distance from origin')
    ax4.set_ylabel('Local spacing')
    ax4.set_title('Spacing vs Distance', fontsize=14)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Drosophila Eye - Parametric model, {nb_ommatidia} ommatidia\n" +
                 f"Method: {'2D RBF' if use_2d_density else 'Angular-Radial'}, " +
                 f"Angle: {grid_angle_deg}°, Density mult: {density_mult}",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()