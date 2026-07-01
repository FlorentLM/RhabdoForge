import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import EllipseCollection
import matplotlib.patches as patches


def fig7_eye_zones(ommatidia_dirs, interp_fn_12, interp_fn_34, raw_12, raw_34):
    """
    2D equirectangular projection of ommatidia directions, overlaid with the 4 zones.

    Reproduction of Fig. 7 from Stürzl et al., 2010.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(ommatidia_dirs[:, 0], ommatidia_dirs[:, 1], color='darkblue', alpha=0.3, s=2, label='Ommatidia')

    # Boundaries
    elevations = np.linspace(-90, 90, 500)
    plt.plot(interp_fn_12(elevations), elevations, color='limegreen', lw=2, label='Eye boundaries (interp.)')
    plt.plot(interp_fn_34(elevations), elevations, color='limegreen', lw=2)

    plt.plot(raw_12[:, 1], raw_12[:, 0], 'o', color='darkgreen', markersize=5, label='Boundary datapoints')
    plt.plot(raw_34[:, 1], raw_34[:, 0], 'o', color='darkgreen', markersize=5)

    plt.xlim(-90, 270)
    plt.ylim(-90, 90)

    ax.axhline(0, color='black', linewidth=1.5, alpha=0.7)
    ax.axvline(0, color='black', linewidth=1.5, alpha=0.7)

    ax.text(125, 50, '1', fontsize=30, color='gray', ha='center', va='center', weight='bold')
    ax.text(125, -50, '2', fontsize=30, color='gray', ha='center', va='center', weight='bold')
    ax.text(-50, -50, '3', fontsize=30, color='gray', ha='center', va='center', weight='bold')
    ax.text(-50, 50, '4', fontsize=30, color='gray', ha='center', va='center', weight='bold')

    ax.set_xlim(-100, 270)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('Azimuth α (degrees)')
    ax.set_ylabel('Elevation ε (degrees)')
    ax.set_title('Equirectangular projection of ommatidia directions')

    plt.tight_layout()
    plt.show()


def fig8_ortho_projection(ommatidia_dirs):
    """
    Orthographic projection of ommatidia directions.

    Reproduction of Fig. 8 from Stürzl et al., 2010.
    """
    from species_models.bee_Sturzl2010.run_sturzl import spherical_to_cartesian_sturzl # TODO: This circular import is annoying

    az = ommatidia_dirs[:, 0]
    el = ommatidia_dirs[:, 1]
    dirs = spherical_to_cartesian_sturzl(az, el, degrees=True)
    x, y, z = dirs

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    scatter_kwargs = {'s': 2, 'c': '#5C60FF', 'alpha': 1.0, 'linewidths': 0}

    # -90 <= alpha <= 90 (front/right view)
    mask1 = (az >= -90) & (az <= 90)
    axes[0].scatter(y[mask1], z[mask1], **scatter_kwargs)
    axes[0].set_xlabel('y')
    axes[0].set_ylabel('z')
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_xticks([-1.0, 0.0, 1.0])
    axes[0].set_yticks([-1.0, 0.0, 1.0])
    axes[0].set_aspect('equal')

    # 0 <= alpha <= 180 (right/back view)
    mask2 = (az >= 0) & (az <= 180)
    axes[1].scatter(x[mask2], z[mask2], **scatter_kwargs)
    axes[1].set_xlabel('x')
    axes[1].set_xlim(1.1, -1.1)
    axes[1].set_xticks([-1.0, 0.0, 1.0])
    axes[1].set_aspect('equal')
    axes[1].tick_params(labelleft=False)

    # 90 <= alpha <= 270 (back view)
    mask3 = (az >= 90) & (az <= 270)
    axes[2].scatter(y[mask3], z[mask3], **scatter_kwargs)
    axes[2].set_xlabel('y')
    axes[2].set_xlim(1.1, -1.1)
    axes[2].set_xticks([-1.0, 0.0, 1.0])
    axes[2].set_aspect('equal')
    axes[2].tick_params(labelleft=False)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    plt.suptitle('Orthographic projections of viewing directions')
    plt.tight_layout()
    plt.show()


def fig10_receptive_fields(directions, delta_rhos):
    """
    Receptive fields.

    Reproduction of Fig. 10 from Stürzl et al., 2010.
    """
    azimuths = directions[:, 0]
    elevations = directions[:, 1]

    # width stretched by 1/cos(elevation) because equirectangular projection
    cos_elev = np.cos(np.radians(elevations))
    cos_elev = np.maximum(cos_elev, 0.001)

    widths = delta_rhos / cos_elev
    heights = delta_rhos

    def add_ellipses(ax, x, y, w, h):
        ec = EllipseCollection(
            widths=w,
            heights=h,
            angles=0,
            units='x',
            offsets=np.column_stack((x, y)),
            transOffset=ax.transData,
            facecolors='none',
            edgecolors='blue',
            linewidths=0.8
        )
        ax.add_collection(ec)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax_main = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    add_ellipses(ax_main, azimuths, elevations, widths, heights)
    ax_main.scatter(azimuths, elevations, color='blue', marker='.', s=1)
    ax_main.set_xlim(-90, 270)
    ax_main.set_ylim(-95, 95)
    ax_main.set_xlabel(r'azimuth $\alpha$ [DEG]')
    ax_main.set_ylabel(r'elevation $\epsilon$ [DEG]')
    ax_main.set_title('(a) Viewing directions and receptive fields')
    ax_main.grid(True, linestyle='--', alpha=0.5)

    # Frontal region
    zoom_b_x = [-10, 10]
    zoom_b_y = [-7, 7]

    # Acute region
    zoom_c_x = [35, 55]
    zoom_c_y = [-7, 7]

    rect_b = patches.Rectangle((zoom_b_x[0], zoom_b_y[0]),
                               zoom_b_x[1] - zoom_b_x[0], zoom_b_y[1] - zoom_b_y[0],
                               linewidth=2, edgecolor='red', facecolor='none')
    rect_c = patches.Rectangle((zoom_c_x[0], zoom_c_y[0]),
                               zoom_c_x[1] - zoom_c_x[0], zoom_c_y[1] - zoom_c_y[0],
                               linewidth=2, edgecolor='red', facecolor='none')
    ax_main.add_patch(rect_b)
    ax_main.add_patch(rect_c)

    # Frontal zoom
    mask_b = (azimuths >= zoom_b_x[0] - 2) & (azimuths <= zoom_b_x[1] + 2) & \
             (elevations >= zoom_b_y[0] - 2) & (elevations <= zoom_b_y[1] + 2)

    add_ellipses(ax_b, azimuths[mask_b], elevations[mask_b], widths[mask_b], heights[mask_b])

    ax_b.scatter(azimuths[mask_b], elevations[mask_b], color='blue', marker='.', s=1)
    ax_b.set_xlim(zoom_b_x)
    ax_b.set_ylim(zoom_b_y)
    ax_b.set_title(r'(b) Frontal region ($\alpha \approx 0^\circ$)')
    ax_b.grid(True, linestyle='--', alpha=0.5)

    # Acute zoom
    mask_c = (azimuths >= zoom_c_x[0] - 2) & (azimuths <= zoom_c_x[1] + 2) & \
             (elevations >= zoom_c_y[0] - 2) & (elevations <= zoom_c_y[1] + 2)

    add_ellipses(ax_c, azimuths[mask_c], elevations[mask_c], widths[mask_c], heights[mask_c])

    ax_c.scatter(azimuths[mask_c], elevations[mask_c], color='blue', marker='.', s=1)
    ax_c.set_xlim(zoom_c_x)
    ax_c.set_ylim(zoom_c_y)
    ax_c.set_title(r'(c) Acute region ($\alpha \approx 45^\circ$)')
    ax_c.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
