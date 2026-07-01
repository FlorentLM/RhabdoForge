import numpy as np

from insectvision.geometry.spherical import sphere_to_stereo
from insectvision.lattice_fitting.profile import EyeMeasurements
from insectvision.lattice_fitting.generator import LatticeGenerator, FittingParameters
from insectvision.lattice_fitting.plots import plot_lattice, lattice_diagnostics
from species_models.drosophila_custom.run import reconstruct_buchner_data


SVG = 'species_models/drosophila_custom/drosophila_Buchner_1971_redigitized.svg'

DENSITY_SCALE     = 1.0     # > 1 packs more ommatidia
DENSITY_SMOOTHING = 0.05    # RBF smoothing of the spacing field
AXES_SMOOTHING    = 0.30    # RBF smoothing of the hexatic-axis field
MIN_HEX_ORDER     = 0.20    # |psi6| below which a point is dropped from the axis fit
GHOST_SOURCE      = 'edge'  # 'hull' | 'edge' | 'none'
ALIGN             = True    # align init grid to the source cloud (False = purely parametric)


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

    print(f'Source:  N={measurements.n_source}  '
          f'mean_spacing={measurements.mean_spacing:.4f}  '
          f'lattice_angles={np.rad2deg(measurements.lattice_angles).round(1)} deg')

    # Generate a lattice from those fields
    params = FittingParameters(density_scale=DENSITY_SCALE, ghost_source=GHOST_SOURCE)
    gen = LatticeGenerator(measurements, params)

    lattice = gen.run(align=ALIGN, verbose=True)

    print(f'Generated:  N={len(lattice)}  '
          f'(target {int(round(measurements.n_source * DENSITY_SCALE))})')

    # Look at it
    plot_lattice(lattice, measurements, density_scale=DENSITY_SCALE)
    lattice_diagnostics(gen.stages)