import numpy as np

from insectvision.geometry.spherical import sphere_to_stereo
from insectvision.lattice_fitting.generator import LatticeGenerator, FittingParameters, EyeMeasurements
from insectvision.lattice_fitting.plots import plot_lattice, plot_lattice_diagnostics
from morphological_scaffolds.drosophila.run import reconstruct_buchner_data


SVG = 'morphological_scaffolds/drosophila/data/buchner1971_redigitised.svg'

DENSITY_SCALE     = 1.0         # > 1 packs more ommatidia
DENSITY_SMOOTHING = 0.1         # RBF smoothing of the spacing field
AXES_SMOOTHING    = 0.50        # RBF smoothing of the hexatic-axis field
MIN_HEX_ORDER     = 0.20        # |psi6| below which a point is dropped from the axis fit
GHOST_SOURCE      = 'hull'      # 'lattice' | 'hull' | 'edge' | 'none'
ALIGN             = True        # align init grid to the source cloud (False = purely parametric)
BYPASS_INIT = False             # If True, starts from the source data directly

if __name__ == '__main__':

    # Measure the source eye: directions -> stereographic plane -> fields
    raw_dirs = reconstruct_buchner_data(SVG)
    pts2d, *_ = sphere_to_stereo(raw_dirs)

    measurements = EyeMeasurements.from_points(
        points2d=pts2d,
        density_smoothing=DENSITY_SMOOTHING,
        axes_smoothing=AXES_SMOOTHING,
        min_hex_order=MIN_HEX_ORDER,
    )

    points_count = len(measurements.source_points)

    print(f'Source: N={points_count} '
          f'mean_spacing={measurements.mean_spacing:.4f}  '
          f'lattice_angles={np.rad2deg(measurements.lattice_angles_rad).round(1)}°')

    # Generate a lattice from those fields
    params = FittingParameters(density_scale=DENSITY_SCALE, ghost_source=GHOST_SOURCE, bypass_init=BYPASS_INIT)
    gen = LatticeGenerator(measurements, params)

    lattice = gen.run(align=ALIGN, verbose=True)

    print(f'Generated: N={len(lattice)} (target {int(round(points_count * DENSITY_SCALE))})')

    # Look at it
    plot_lattice(lattice, measurements, density_scale=DENSITY_SCALE)
    plot_lattice_diagnostics(gen.stages)