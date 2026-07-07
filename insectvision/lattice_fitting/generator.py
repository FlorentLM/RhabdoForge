"""
Generation: build a lattice from a measured EyeProfile

  1- Warp: Hex grid clipped to the domain, warped to the target density field
  2- Spring relax: Orientation-following spring relaxation
  3- Adjust: Smooth area-correcting transport so density has the last word
  4- Finalise: Trim then clean the boundary (cull / fill notches / settle)
"""
from dataclasses import dataclass
from typing import Optional, Callable, Sequence
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from insectvision.geometry.fields import interpolate_spacing_field, interpolate_hexatic_field
from insectvision.geometry.linalg import rotation_matrix2d
from insectvision.geometry.polygons import resample_contour, Polygon2D
from insectvision.geometry.neighbours import delaunay_neighbours, ball_spacing
from insectvision.geometry.lattice import (
    first_ring_gap, create_hexagonal_grid, base_bond_dirs, compute_lattice_basis, trace_lattice_rows, bearings_to_angles
)
from insectvision.lattice_fitting.relaxation import align_grid, density_warp, spring_relaxation, density_correct


# Stage 1: density warp

def _warp_init(
        domain: 'Polygon2D',
        target_spacing_fn: Callable,
        target_count: int,
        rot0: ArrayLike,
        extent: float,
        buffer: float,
        lattice_angles: float | Sequence[float],
        align_points: Optional[ArrayLike] = None,
        warp_exponent: float = 1.0
    ) -> np.ndarray:
    """
    Hex grid clipped to the domain and warped to the target density field.
    """

    rot0 = np.asarray(rot0, dtype=np.float64)
    lattice_angles = np.asarray(lattice_angles, dtype=np.float64)

    factor = abs(np.linalg.det(compute_lattice_basis(1.0, lattice_angles)))

    current_spacing = np.sqrt(domain.area / (target_count * factor))
    for _ in range(6):
        tentative_grid = create_hexagonal_grid(spacing=current_spacing, angles=lattice_angles, extent=extent) @ rot0.T
        tentative_lattice = tentative_grid[domain.inside(tentative_grid, buffer=buffer)]

        tentative_lattice = density_warp(tentative_lattice, target_spacing_fn,
                                         reference_spacing=current_spacing, exponent=warp_exponent)

        count_inside = domain.inside(tentative_lattice, buffer=0.0).sum()
        if count_inside == 0:
            current_spacing *= 0.7
            continue

        ratio = target_count / count_inside
        current_spacing *= 1.0 / np.sqrt(ratio)
        if abs(ratio - 1.0) < 0.03:
            break

    selected_grid = create_hexagonal_grid(spacing=current_spacing, angles=lattice_angles, extent=extent) @ rot0.T
    if align_points is not None:
        selected_grid = align_grid(selected_grid, align_points)

    selected_lattice = selected_grid[domain.inside(selected_grid, buffer=buffer)]
    selected_warped_lattice = density_warp(selected_lattice, target_spacing_fn,
                                           reference_spacing=current_spacing, exponent=warp_exponent)

    return selected_warped_lattice


# Stage 4: boundary finalisation

def _cull_junk(
        points2d: ArrayLike,
        domain: 'Polygon2D',
        target_spacing_fn: Callable,
        avg_spacing: float,
        boundary_gap_deg: float = 110.0,
        straggler_ratio: float = 1.5,
        outside_factor: float = 0.5,
        merge_factor: float = 0.5,
        verbose: bool = False
    ) -> np.ndarray:
    """
    Remove out of boundary stray points and merge near-duplicates.

    A point is junk if it is both a boundary site (open first ring) and either of:
        - under-coordinated (deg <= 2)
        - too sparse (local/target spacing > straggler_ratio),
        - Clearly outside the contour (signed hull distance > outside_factor * avg_spacing)

    Near-duplicates (closer than merge_factor * avg_spacing) are collapsed to one point
    """

    pts = np.copy(points2d).astype(np.float64)

    neighbours = delaunay_neighbours(pts, max_length_factor=1.8)

    gap = first_ring_gap(pts, neighbours)
    deg = np.array([len(nb) for nb in neighbours])

    s_ratio = ball_spacing(query_points=pts, k=6) / np.maximum(target_spacing_fn(pts).ravel(), 1e-12)
    d_hull = domain.signed_distance(pts)

    is_boundary = gap > np.deg2rad(boundary_gap_deg)
    is_junk = is_boundary & ((deg <= 2) | (s_ratio > straggler_ratio) | (d_hull > outside_factor * avg_spacing))
    pts = pts[~is_junk]
    if verbose:
        print(f'  finalize: culled {int(is_junk.sum())} boundary stragglers')

    pairs = cKDTree(pts).query_pairs(r=merge_factor * avg_spacing)
    if pairs:
        drop = set()
        for i, j in sorted(pairs):   # deterministic -> drop higher index
            if i not in drop and j not in drop:
                drop.add(j)

        keep = np.ones(len(pts), dtype=bool)
        keep[list(drop)] = False
        pts = pts[keep]
        if verbose:
            print(f'  finalize: merged {len(drop)} near-duplicate pairs')

    return pts


def _finalize_lattice(
        points2d: ArrayLike,
        domain: 'Polygon2D',
        target_spacing_fn: Callable,
        avg_spacing: float,
        theta_fn: Callable,
        bond_dirs: ArrayLike,
        params: Optional['FittingParameters'] = None,
        verbose: bool = False
    ) -> np.ndarray:
    """
    Clean the boundary of a relaxed lattice:
        1- Cull boundary stragglers + merge near-duplicates
        2- Add candidate ring points along the smooth contour at the local target spacing, then only keep those that fill a gap
        3- Settle with a short spring relaxation pass bounded by mirrored ghost points (skipped if ghost_source == 'none')
    """

    p = params or FittingParameters()

    pts = _cull_junk(
        points2d=points2d,
        domain=domain,
        target_spacing_fn=target_spacing_fn,
        avg_spacing=avg_spacing,
        boundary_gap_deg=p.boundary_gap_deg,
        straggler_ratio=p.straggler_ratio,
        outside_factor=p.outside_factor,
        merge_factor=p.merge_factor,
        verbose=verbose
    )

    # Ring points are gap fillers: keep a seed only if no real point is already near
    ring = resample_contour(domain.boundary, target_spacing_fn)
    d_to_pts, _ = cKDTree(pts).query(ring)

    fill = ring[d_to_pts > p.fill_factor * target_spacing_fn(ring).ravel()]
    combined = np.vstack([pts, fill]) if len(fill) else pts

    if p.ghost_source == 'none':
        if verbose:
            print(f'  finalize: filled {len(fill)} boundary gaps, settle skipped (ghosts off)')
        return combined

    if verbose:
        print(f'  finalize: filled {len(fill)} boundary gaps, settling {p.settle_iters} '
              f'spring iters (free boundary, ghost_source={p.ghost_source!r})')

    # Free-boundary settle: ghost points provide outward pressure
    return spring_relaxation(
        points2d=combined,
        spacing_fn=target_spacing_fn,
        theta_fn=theta_fn,
        bond_dirs=bond_dirs,
        max_iter=p.settle_iters,
        dt=p.dt,
        retriangulate_every=p.retriangulate_every,
        verbose=verbose,
        domain=domain,
        ghost_depth=p.ghost_depth_factor * avg_spacing,
        ghost_source=p.ghost_source
    )

##

# Fitting dataclasses

@dataclass
class EyeMeasurements:
    """
    Measurements from a source 2D cloud (used to feed generation).
    """
    domain: 'Polygon2D'
    spacing_fn: Callable
    theta_fn: Callable
    lattice_angles_rad: np.ndarray
    mean_spacing: float
    source_points: Optional[np.ndarray] = None

    @classmethod
    def from_points(cls,
            points2d: ArrayLike,
            density_smoothing: float = 0.1,
            axes_smoothing: float = 0.4,
            min_hex_order: float = 0.2,
            max_length_factor:float = 1.8,
            smooth_domain: bool = True
        ) -> 'EyeMeasurements':

        points2d = np.asarray(points2d, dtype=float)

        neighbours = delaunay_neighbours(points2d=points2d, max_length_factor=max_length_factor)

        # Create continuous fields
        spacing_fn, mean_spacing = interpolate_spacing_field(
            points2d=points2d,
            neighbours=neighbours,
            smoothing=density_smoothing,
            return_mean_spacing=True
        )

        theta_fn = interpolate_hexatic_field(
            points2d=points2d,
            neighbours=neighbours,
            smoothing=axes_smoothing,
            min_order=min_hex_order,
            return_confidence=False     # TODO: actually why not? could be useful
        )

        # Trace the rows to get the 3 primary bearings
        _, _, bearings = trace_lattice_rows(points2d=points2d, neighbours=neighbours, theta_fn=theta_fn)

        # Convert to internal angles
        lattice_angles = bearings_to_angles(bearings)

        domain = Polygon2D.from_points(points2d=points2d, smooth=smooth_domain)

        return cls(domain, spacing_fn, theta_fn, lattice_angles, mean_spacing, points2d)


@dataclass
class FittingParameters:
    """Parameters for lattice generation."""

    density_scale: float = 1.0          # > 1 packs more ommatidia

    # Stage 1: Density warp
    buffer_factor: float = 2.0          # grid kept at this * spacing outside the domain (pre-relaxation)
    warp_exponent: float = 1.0          # > 1 pushes the initial radial density contrast harder
    spring_margin_factor: float = 1.0   # Outer scaffold kept at this * spacing (before relaxing)

    # Stage 2: Spring relaxation (curved axes + emergent defects)
    spring_iters: int = 120
    dt: float = 0.1
    retriangulate_every: int = 5

    # Ghosts points (for boundary handling, shared by stage 2 and final settle)
    ghost_source: str = 'lattice'       # 'lattice' | 'hull' | 'edge' | 'none'
    ghost_depth_factor: float = 1.5     # mirror points within this * spacing (from the boundary)

    # Stage 3: Density correction
    density_correct_iters: int = 3      # Poisson transport passes (0 = disabled)

    # Stage 4: Boundary finalisation
    finalize: bool = True
    keep_tol_factor: float = 0.35        # Coarse distance-trim before finalisation
    boundary_gap_deg: float = 110.0     # first-ring gap above which a point counts as boundary
    straggler_ratio: float = 1.2        # Cull boundary points sparser than this * target
    outside_factor: float = 0.5         # Cull boundary points this * spacing outside the hull
    merge_factor: float = 0.5           # Merge point pairs closer than this * spacing
    fill_factor: float = 0.75            # Add a point seed if no point is within this * spacing
    settle_iters: int = 40              # Spring relaxation iters for the final settle

    def __post_init__(self):
        self.ghost_source = 'none' if not self.ghost_source else str(self.ghost_source).lower()
        if self.ghost_source not in ('lattice', 'hull', 'edge', 'none'):
            raise ValueError(f"ghost_source must be 'lattice', 'hull', 'edge' or 'none', got {self.ghost_source!r}")


class LatticeGenerator:
    """
    Generate a lattice from a EyeMeasurements.

    Construct with a profile and (optionally) an FittingParameters, then call run()
        profile = EyeMeasurements.from_points(points2d)
        gen = LatticeGenerator(profile, FittingParameters(density_scale=1.1))
        lattice = gen.run()
        # gen.stages -> {'init', 'relaxed', 'trimmed', 'final', ...}
    """

    def __init__(self,
            measurements: 'EyeMeasurements',
            params: Optional['FittingParameters'] = None
        ):

        self.measurements: 'EyeMeasurements' = measurements
        self.params: 'FittingParameters' = params or FittingParameters()
        self.stages: dict = {}

    def run(self, align: bool = True, verbose: bool = True) -> np.ndarray:

        # Determine spacing of the wanted lattice (applies density scale)
        spacing = self.measurements.mean_spacing / np.sqrt(self.params.density_scale)
        target_spacing_fn = lambda q: self.measurements.spacing_fn(q) / np.sqrt(self.params.density_scale)

        if self.measurements.source_points is not None:
            target_point_count = int(round(len(self.measurements.source_points) * self.params.density_scale))
        else:
            factor = abs(np.linalg.det(compute_lattice_basis(1.0, self.measurements.lattice_angles_rad)))
            target_point_count = int(round(self.measurements.domain.area / (factor * spacing ** 2)))

        buffer = self.params.buffer_factor * spacing
        extent = float(np.max(np.abs(self.measurements.domain.boundary)) + 5 * spacing)

        theta0 = float(self.measurements.theta_fn(self.measurements.domain.boundary.mean(axis=0, keepdims=True))[0])
        rot0 = rotation_matrix2d(theta0)
        bond_dirs = base_bond_dirs(self.measurements.lattice_angles_rad)

        align_points = self.measurements.source_points if (align and self.measurements.source_points is not None) else None

        # Density match
        lattice = _warp_init(
            domain=self.measurements.domain,
            target_spacing_fn=target_spacing_fn,
            target_count=target_point_count,
            rot0=rot0,
            extent=extent,
            buffer=buffer,
            lattice_angles=self.measurements.lattice_angles_rad,
            align_points=align_points,
            warp_exponent=self.params.warp_exponent
        )

        # outer scaffold trimmed before relaxing
        inside = self.measurements.domain.signed_distance(lattice) < self.params.spring_margin_factor * spacing
        init = lattice[inside]

        # Curved axes + emergent defects
        relaxed = spring_relaxation(
            points2d=init,
            spacing_fn=target_spacing_fn,
            theta_fn=self.measurements.theta_fn,
            bond_dirs=bond_dirs,
            max_iter=self.params.spring_iters,
            dt=self.params.dt,
            retriangulate_every=self.params.retriangulate_every,
            verbose=verbose,
            domain=self.measurements.domain,
            ghost_depth=self.params.ghost_depth_factor * spacing,
            ghost_source=self.params.ghost_source
        )

        # Density gets the last word: smooth area-correcting transport
        if self.params.density_correct_iters > 0:
            relaxed = density_correct(
                points2d=relaxed,
                target_spacing_fn=target_spacing_fn,
                domain=self.measurements.domain,
                n_iter=self.params.density_correct_iters,
                verbose=verbose
            )

        # Final trim and then clean the boundary
        inside = self.measurements.domain.signed_distance(relaxed) < self.params.keep_tol_factor * spacing
        trimmed = relaxed[inside]

        if self.params.finalize:
            final = _finalize_lattice(
                points2d=trimmed,
                domain=self.measurements.domain,
                target_spacing_fn=target_spacing_fn,
                avg_spacing=spacing,
                theta_fn=self.measurements.theta_fn,
                bond_dirs=bond_dirs,
                params=self.params,
                verbose=verbose
            )
        else:
            final = trimmed

        self.stages = {
            'init': init,
            'relaxed': relaxed,
            'trimmed': trimmed,
            'final': final,
            'target_spacing_fn': target_spacing_fn,
            'gen_spacing': spacing,
            'measurements': self.measurements,
            'parameters': self.params,
        }

        return final