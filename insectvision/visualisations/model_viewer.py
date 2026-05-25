import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

from insectvision.compound_eyes import CompoundEyeModel
from insectvision.compound_eyes.kernel import drosophila_kernel
from insectvision.compound_eyes.orientation import BundlesAligner
from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD
from insectvision.utils.math import project_to_stereo

RECEPTOR_PALETTE = [
    '#ffad13',  # R1
    '#FF1C25',  # R2
    '#880015',  # R3
    '#8000ff',  # R4
    '#008000',  # R5
    '#0000ff',  # R6
    '#aaa712',  # R7/8
]

CHIRALITY_NEG_COLOR = '#B95D21'
CHIRALITY_POS_COLOR = '#FF9800'

SURFACE_COLOR = 'white'
SURFACE_OPAC = 0.18
EQUATOR_COLOR = '#3b6dff'
SAGITTAL_COLOR = '#888888'
PLANE_OPAC = 0.06
FLOW_COLOR = '#9b3ddc'

# Templates for glyph rendering (built once, shared across panels)

_ARROW_TEMPLATE = None
_PHASOR_TEMPLATE = None
_DISC_TEMPLATE = None


##

# Helper functions

def receptor_tip_offsets(ra) -> np.ndarray:
    N, R = ra.lens_count, ra.receptors_per_lens
    d = ra.lenses.direction
    rec_dirs = ra.rcpt_dynamic_data['direction'].reshape(N, R, 3)
    axial = np.sum(rec_dirs * d[:, None, :], axis=2, keepdims=True)
    return -(rec_dirs - axial * d[:, None, :])


def radii_from_lattice(ra) -> np.ndarray:
    """Per-lens display radius = mean distance to the lens's immediate lattice neighbours."""

    radii = np.zeros(ra.lens_count, dtype=np.float32)
    for eye in ra.eyes:
        if len(eye) == 0:
            continue
        result = eye.neighbours(lens_indices=eye.lens_indices, k=6, immediate_only=True)
        if result.distances.size == 0:
            continue
        radii[eye.lens_indices] = result.distances.mean(axis=1)
    radii[radii < 1e-9] = 0.001
    return radii


def make_eye_mesh(ra) -> pv.PolyData:
    """
    Lens-indexed mesh: vertex i = lens i.
    """
    positions = ra.lenses.position
    if positions.shape[0] == 0:
        return pv.PolyData()

    all_faces = []
    for eye in ra.eyes:
        if len(eye) < 3:
            continue

        global_idx = np.asarray(eye.lens_indices, dtype=np.int_)
        eye_pos = positions[global_idx]

        eye_dirs = eye_pos / np.linalg.norm(eye_pos, axis=1, keepdims=True)
        try:
            pts_2d, _, _, _ = project_to_stereo(eye_dirs)
            tri = Delaunay(pts_2d)
            simplices = tri.simplices
        except Exception:
            continue

        # Filter out spiky gap-filling triangles
        p0 = eye_pos[simplices[:, 0]]
        p1 = eye_pos[simplices[:, 1]]
        p2 = eye_pos[simplices[:, 2]]

        # Get 3D lengths of all triangle edges
        l01 = np.linalg.norm(p0 - p1, axis=1)
        l12 = np.linalg.norm(p1 - p2, axis=1)
        l20 = np.linalg.norm(p2 - p0, axis=1)

        # Find maximum edge length for each triangle and estimate median edge length
        max_len = np.max(np.column_stack([l01, l12, l20]), axis=1)
        typical_edge = np.median(np.min(np.column_stack([l01, l12, l20]), axis=1))

        # Drop triangles whose longest edge is significantly large
        valid_mask = max_len < (typical_edge * 2.0)
        valid_simplices = simplices[valid_mask]

        if len(valid_simplices) > 0:
            all_faces.append(global_idx[valid_simplices])

    if not all_faces:
        return pv.PolyData(positions.astype(np.float32))

    faces = np.vstack(all_faces)
    flat = np.empty((faces.shape[0], 4), dtype=np.int_)
    flat[:, 0] = 3
    flat[:, 1:] = faces
    return pv.PolyData(positions.astype(np.float32), faces=flat.flatten())


def eye_boundary_line(lens_data_mesh: pv.PolyData) -> pv.PolyData:
    """
    Extracts the exact outer boundary of the valid triangulated eye mesh and smoothes for a more organic edge.
    """
    if lens_data_mesh.n_cells == 0 or lens_data_mesh.n_points == 0:
        return pv.PolyData()

    edges = lens_data_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
        feature_edges=False
    )

    # relaxation smoothing to the extracted polyline vertices
    if edges.n_points > 0:
        edges = edges.smooth(n_iter=100, relaxation_factor=0.05)

    return edges


##

def _arrow_template() -> pv.PolyData:
    global _ARROW_TEMPLATE
    if _ARROW_TEMPLATE is None:
        _ARROW_TEMPLATE = pv.Arrow(tip_radius=0.08, shaft_radius=0.03, tip_length=0.25)
    return _ARROW_TEMPLATE


def _phasor_template() -> pv.PolyData:
    global _PHASOR_TEMPLATE
    if _PHASOR_TEMPLATE is None:
        _PHASOR_TEMPLATE = pv.Line(pointa=(-0.5, 0, 0), pointb=(0.5, 0, 0))
    return _PHASOR_TEMPLATE


def _disc_template() -> pv.PolyData:
    """
    Unit disc lying in YZ plane (normal = +X). Glyph filter aligns the
    template's +X with the orient vector, so this places each disc with its
    normal along the lens optical axis (tangent to the eye surface).
    """
    global _DISC_TEMPLATE
    if _DISC_TEMPLATE is None:
        _DISC_TEMPLATE = pv.Disc(inner=0, outer=1.0, c_res=18, normal=(1, 0, 0))
    return _DISC_TEMPLATE


##

class EyeViewer:
    """
    Multi-panel 3D viewer for a CompoundEyeModel.
    """

    def __init__(
        self,
        model: CompoundEyeModel,
        aligner: BundlesAligner = None,
        optic_flow_world=None,
        sparsity: float = 0.0,
    ):

        self.model = model
        self.kernel = model.kernel
        self.N = model.lens_count
        self.R = model.receptors_per_lens
        self.sparsity = float(sparsity)

        self.right = np.asarray(WORLD_RIGHT, dtype=np.float32)
        self.up = np.asarray(WORLD_UP, dtype=np.float32)
        self.fwd = np.asarray(WORLD_FORWARD, dtype=np.float32)

        if aligner is None:
            flow = optic_flow_world if optic_flow_world is not None else -self.fwd
            aligner = BundlesAligner(flow_direction=flow)
        self.aligner_smooth = aligner
        self.aligner_raw = BundlesAligner(
            flow_direction=aligner.flow_direction,
            diagonal_strength=aligner.diagonal_strength,
            diagonal_angle_deg=getattr(aligner, 'diagonal_angle_deg', 45.0),
            alignment_smoothing_iterations=0,
            saccade_smoothing_iterations=0,
            falloff=aligner.falloff,
            strength=aligner.strength,
        )
        self.optic_flow_world = aligner.flow_direction.astype(np.float32)

        self.result_smooth = self.aligner_smooth.compute(self.model)
        self.result_raw = self.aligner_raw.compute(self.model)

        # Cache geometry
        self.p = model.lenses.position
        self.d = model.lenses.direction
        self.r_sphere = float(np.mean(np.linalg.norm(self.p, axis=1)))
        self.arrow_len = self.r_sphere * 0.08

        # Per-eye Delaunay mesh (vertex i = lens i)
        self.lens_data_mesh = make_eye_mesh(self.model)
        self.eye_boundary_lines = eye_boundary_line(self.lens_data_mesh)

        # Per-lens chirality
        self.chirality = self.model.lenses.chirality.astype(np.float32)

        # Per-lens lattice spacing
        self.disc_radii = radii_from_lattice(self.model) * 0.4

        # Heatmap scalars
        self.collinearity = self._compute_collinearity()
        self.smoothness = self._compute_smoothness()

        if self.R > 1 and getattr(self.model, '_cartridges_wired', False):
            c_field = np.zeros(self.N, dtype=np.float32)
            c_field[self.model.donation_conflicts] += 1.0
            c_field[self.model.receiving_conflicts] += 2.0
            self.conflicts_field = c_field
        else:
            self.conflicts_field = None

        # Plot bookkeeping
        self.plotter = None
        self.actors_alignment_smooth = []
        self.actors_alignment_raw = []
        self.actors_saccade_smooth = []
        self.actors_saccade_raw = []
        self.actors_bundles = []
        self.actors_conflicts = []
        self.actors_debugger = []

        self.state_alignment_smoothed = True
        self.state_saccade_smoothed = True

        # Big panel state: 0 = bundles, 1 = debugger, 2 = conflicts heatmap
        self.state_bigpanel = 1
        self._BIGPANEL_LABELS = ['Bundles', 'Wiring debugger', 'Conflicts']

        self._debugger_subplot = None  # filled in show()

    ##

    def show(self):
        self.plotter = pv.Plotter(
            shape=(2, 4),
            groups=[(1, slice(2, 4))],
            window_size=list((2400, 1100)),
            border=True,
        )
        self.plotter.set_background('white')

        self._debugger_subplot = (1, 2)

        # Panel layout
        panel_setups = [
            ((0, 0), "Optic Flow", self._add_optic_flow_panel),
            ((0, 1), "Alignment Axes  [A]", self._add_alignment_panel),
            ((0, 2), "Major Axes", self._add_major_axis_panel),
            ((0, 3), "Saccade Axes  [S]", self._add_saccade_panel),
            ((1, 0), "Flow / Alignment collinearity",
             self._add_collinearity_panel),
            ((1, 1), "Saccade smoothing consistency",
             self._add_smoothness_panel),
            ((1, 2), "Bundles / Wiring / Conflicts  [B]  [N]",
             self._add_bigpanel),
        ]

        for (row, col), title, builder in panel_setups:
            self.plotter.subplot(row, col)
            self._common_scene(title)
            builder()

        # visibility states + linking
        self._apply_alignment_visibility()
        self._apply_saccade_visibility()
        self._apply_bigpanel_visibility()

        self.plotter.link_views()
        self._setup_keybindings()
        self._setup_camera()
        self.plotter.show()

    ##

    # Per-panel fluff (planes, flow indicator, title, axes)

    def _common_scene(self, title: str) -> None:
        self.plotter.add_text(title, font_size=9, color='black')

        # Equatorial and sagittal ref planes (head frame)
        plane_size = self.r_sphere * 2.6
        eq = pv.Plane(center=(0, 0, 0), direction=self.up,
                      i_size=plane_size, j_size=plane_size)
        sag = pv.Plane(center=(0, 0, 0), direction=self.right,
                       i_size=plane_size, j_size=plane_size)
        self.plotter.add_mesh(eq, color=EQUATOR_COLOR, opacity=PLANE_OPAC)
        self.plotter.add_mesh(sag, color=SAGITTAL_COLOR, opacity=PLANE_OPAC)

        # Optic flow direction (big arrow at the head front)
        flow_arrow = pv.Arrow(
            start=(-1.8 * self.r_sphere * self.optic_flow_world).tolist(),
            direction=self.optic_flow_world.tolist(),
            scale=self.r_sphere * 0.6,
        )
        self.plotter.add_mesh(flow_arrow, color=FLOW_COLOR, lighting=False, opacity=0.85)

        self.plotter.add_axes(interactive=False)

    def _add_eye_surface(
        self,
        scalars: np.ndarray = None,
        cmap=None,
        clim=None,
        sbar_title: str = None,
        faint: bool = False,
    ) -> None:
        """
        Render the eye as a surface mesh, optionally coloured by a scalar field

        - scalars not None: continuous heatmap (smooth_shading + Gouraud)
        - scalars is None, faint=True: faint white background surface
        - scalars is None, faint=False: opaque white surface
        """
        if self.lens_data_mesh.n_cells == 0:
            return

        m = self.lens_data_mesh.copy()
        if scalars is not None:
            m.point_data['_scalar'] = scalars.astype(np.float32)
            kwargs = dict(
                scalars='_scalar', cmap=cmap,
                show_scalar_bar=(sbar_title is not None),
                smooth_shading=True, ambient=0.3, diffuse=0.7,
            )
            if clim is not None:
                kwargs['clim'] = list(clim)
            if sbar_title is not None:
                kwargs['scalar_bar_args'] = {
                    'title': sbar_title, 'n_labels': 3,
                    'position_x': 0.78, 'position_y': 0.05,
                    'width': 0.18, 'height': 0.06, 'color': 'black',
                }
            self.plotter.add_mesh(m, **kwargs)
        else:
            self.plotter.add_mesh(
                m, color=SURFACE_COLOR,
                opacity=(SURFACE_OPAC if faint else 1.0),
                show_edges=False, smooth_shading=True, lighting=True,
            )

        # Boundary polyline overlay
        if self.eye_boundary_lines.n_cells > 0:
            self.plotter.add_mesh(
                self.eye_boundary_lines,
                color='black', line_width=1.5, opacity=0.55,
                lighting=False,
            )

    ##

    # Individual panel content

    def _glyph_arrows(self, mesh: pv.PolyData, orient_name: str) -> pv.PolyData:
        return mesh.glyph(
            geom=_arrow_template(), orient=orient_name,
            factor=self.arrow_len, scale=False,
            tolerance=self.sparsity if self.sparsity > 0 else None,
        )

    def _glyph_phasors(self, mesh: pv.PolyData, orient_name: str) -> pv.PolyData:
        return mesh.glyph(
            geom=_phasor_template(), orient=orient_name,
            factor=self.arrow_len, scale=False,
            tolerance=self.sparsity if self.sparsity > 0 else None,
        )

    def _add_optic_flow_panel(self) -> None:
        if self.lens_data_mesh.n_cells == 0:
            return
        dots = self.d @ self.optic_flow_world
        v_proj = self.optic_flow_world[None, :] - dots[:, None] * self.d
        norms = np.linalg.norm(v_proj, axis=1, keepdims=True)
        v_proj = np.divide(v_proj, norms.clip(min=1e-8))

        m = self.lens_data_mesh.copy()
        m.point_data['OpticFlow'] = v_proj.astype(np.float32)
        arrows = self._glyph_arrows(m, 'OpticFlow')
        self.plotter.add_mesh(arrows, color='#da70d6',
                              ambient=0.4, diffuse=0.7, smooth_shading=True)

    def _add_alignment_panel(self) -> None:
        if self.lens_data_mesh.n_cells == 0 or self.R <= 1:
            return

        m_smooth = self.lens_data_mesh.copy()
        m_smooth.point_data['AlignmentSmooth'] = self.result_smooth.alignment_phasor.astype(np.float32)
        g_smooth = self._glyph_phasors(m_smooth, 'AlignmentSmooth')
        a_smooth = self.plotter.add_mesh(g_smooth, color='green', line_width=2)
        self.actors_alignment_smooth.append(a_smooth)

        m_raw = self.lens_data_mesh.copy()
        m_raw.point_data['AlignmentRaw'] = self.result_raw.alignment_phasor.astype(np.float32)
        g_raw = self._glyph_phasors(m_raw, 'AlignmentRaw')
        a_raw = self.plotter.add_mesh(g_raw, color='#8c8c00', line_width=2)
        self.actors_alignment_raw.append(a_raw)

    def _add_major_axis_panel(self) -> None:
        if self.lens_data_mesh.n_cells == 0 or self.R <= 1:
            return

        major = self.result_smooth.major_axis.astype(np.float32)

        pos_mask = self.chirality > 0
        neg_mask = self.chirality < 0

        for mask, color in [(neg_mask, CHIRALITY_NEG_COLOR),
                            (pos_mask, CHIRALITY_POS_COLOR)]:
            if not np.any(mask):
                continue
            pd = pv.PolyData(self.p[mask].astype(np.float32))
            pd.point_data['MajorAxis'] = major[mask]
            arrows = pd.glyph(geom=_arrow_template(), orient='MajorAxis',
                              factor=self.arrow_len, scale=False)
            self.plotter.add_mesh(arrows, color=color,
                                  ambient=0.4, diffuse=0.7, smooth_shading=True)

    def _add_saccade_panel(self) -> None:
        if self.lens_data_mesh.n_cells == 0:
            return

        m_smooth = self.lens_data_mesh.copy()
        m_smooth.point_data['SaccadeSmooth'] = self.result_smooth.saccade_phasor.astype(np.float32)
        g_smooth = self._glyph_arrows(m_smooth, 'SaccadeSmooth')
        a_smooth = self.plotter.add_mesh(g_smooth, color='red',
                                         ambient=0.4, diffuse=0.7, smooth_shading=True)
        self.actors_saccade_smooth.append(a_smooth)

        m_raw = self.lens_data_mesh.copy()
        m_raw.point_data['SaccadeRaw'] = self.result_raw.saccade_phasor.astype(np.float32)
        g_raw = self._glyph_phasors(m_raw, 'SaccadeRaw')
        a_raw = self.plotter.add_mesh(g_raw, color='#FF94BD', line_width=2)
        self.actors_saccade_raw.append(a_raw)

    def _add_collinearity_panel(self) -> None:
        """Per-lens |raw flow . combed alignment phasor|, as a coloured surface."""
        if self.collinearity is None or self.R <= 1:
            self.plotter.add_text("(needs R > 1)", position='lower_left',
                                  font_size=8, color='gray')
            return
        self._add_eye_surface(
            scalars=self.collinearity, cmap='inferno', clim=[0.0, 1.0],
            sbar_title='Collinearity',
        )

    def _add_smoothness_panel(self) -> None:
        """Per-lens |raw saccade . smoothed saccade|, as a coloured surface."""
        if self.smoothness is None:
            self.plotter.add_text("(needs R > 1)", position='lower_left',
                                  font_size=8, color='gray')
            return
        self._add_eye_surface(
            scalars=self.smoothness, cmap='inferno', clim=[0.9, 1.0],
            sbar_title='Smoothness',
        )

    def _add_bigpanel(self) -> None:
        """Bottom-right panel (wide)."""

        self._add_eye_surface(faint=True)

        # Mode 1: rhabdomere-tip bundles overview
        if self.R > 1:
            offsets = receptor_tip_offsets(self.model)
            max_off = float(np.max(np.linalg.norm(offsets, axis=2)))
            if max_off > 1e-8:
                tip_scale = (self.r_sphere * 0.012) / max_off
                tip_positions = self.p[:, None, :] + offsets * tip_scale
                is_mirrored = self.model.lenses.chirality < 0
                i1, i2 = self.kernel.main_axis_indices

                groups = {
                    'gold': [], 'red': [], 'white': [],
                    'teal': [], 'royalblue': [],
                }
                for r in range(self.R):
                    pts_r = tip_positions[:, r, :]
                    if r == self.kernel.center_index:
                        groups['gold'].append(pts_r)
                    elif r == i1:
                        groups['red'].append(pts_r)
                    elif r == i2:
                        groups['white'].append(pts_r)
                    else:
                        groups['teal'].append(pts_r[~is_mirrored])
                        groups['royalblue'].append(pts_r[is_mirrored])

                for color, pts_list in groups.items():
                    arrays = [a for a in pts_list if len(a) > 0]
                    if not arrays:
                        continue
                    stacked = np.vstack(arrays)
                    act = self.plotter.add_points(
                        stacked, color=color, point_size=8,
                        render_points_as_spheres=True,
                    )
                    self.actors_bundles.append(act)

        # Mode 2: conflicts heatmap
        if self.conflicts_field is not None:
            eps = float(np.median(self.disc_radii)) * 0.05
            positions = self.p + eps * self.d
            pd = pv.PolyData(positions.astype(np.float32))
            pd.point_data['vectors'] = self.d.astype(np.float32)
            pd.point_data['radius'] = self.disc_radii.astype(np.float32)
            pd.point_data['conflicts'] = self.conflicts_field
            discs = pd.glyph(geom=_disc_template(), orient='vectors',
                             scale='radius', factor=1.1)

            # Discrete colormap:
            # 0: Light gray (OK), 1: Yellow (Donation), 2: Cyan (Receiving), 3: Red (Both)
            cmap_colors = ['#e0e0e0', '#f1c40f', '#00bcd4', '#e74c3c']

            act = self.plotter.add_mesh(
                discs, scalars='conflicts',
                cmap=cmap_colors, clim=[-0.5, 3.5],
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Conflicts (0:OK 1:Don 2:Rcv 3:Both)', 'n_labels': 5,
                    'position_x': 0.70, 'position_y': 0.05,
                    'width': 0.28, 'height': 0.06, 'color': 'black',
                },
                ambient=0.4, diffuse=0.7,
            )
            self.actors_conflicts.append(act)

        # Mode 3: wiring debugger
        if getattr(self.model, '_cartridges_wired', False) and self.R > 1:
            self._redraw_debugger()

    ##

    # Wiring debugger

    def _redraw_debugger(self, target_idx: int = None) -> None:
        """Clear previous debugger actors and draw a new (random) cartridge."""
        # TODO: These annoying dots still show up :(

        if self._debugger_subplot is not None:
            self.plotter.subplot(*self._debugger_subplot)

        # Clear previous
        for act in self.actors_debugger:
            self.plotter.remove_actor(act)
        self.actors_debugger.clear()

        if not getattr(self.model, '_cartridges_wired', False) or self.R <= 1:
            return

        if target_idx is None:
            target_idx = int(np.random.randint(0, self.N))

        omm = self.model.lenses[target_idx]
        eye = omm.eye
        target_pos = omm.position

        # Find spatial neighbours within this lens's eye
        k_nb = min(20, len(eye))
        result = eye.neighbours(points=target_pos[None, :], k=k_nb)
        neighbours = result.indices[0]
        dists = result.distances[0]

        # Cartridge partners for this lens
        partners = self.model.cartridges[target_idx].sources
        partner_to_type = {
            int(lens_idx): r
            for r, lens_idx in enumerate(partners)
            if int(lens_idx) != target_idx
        }

        # Define rings by distance (skip target itself)
        nb_dists = dists[1:]
        nb_idx = neighbours[1:]
        if len(nb_dists) > 0:
            closest = nb_dists[0]
            ring1_mask = nb_dists <= closest * 1.5
            ring1 = nb_idx[ring1_mask]
            rest = nb_idx[~ring1_mask]
            ring2 = rest[:12]
            outer = rest[12:]
            ioa_estimate = float(np.mean(nb_dists[ring1_mask]))
        else:
            ring1 = ring2 = outer = np.array([], dtype=np.intp)
            ioa_estimate = 0.01

        disc_rad = ioa_estimate * 0.4

        ring_groups = [
            ([target_idx], '#FFD700', 0.7),    # home
            (ring1.tolist(), '#000000', 0.5),  # ring 1
            (ring2.tolist(), '#000000', 0.2),  # ring 2
            (outer.tolist(), '#000000', 0.04), # outer
        ]

        # Coloured disc per neighbour, partners get a receptor-coloured disc
        for idx_list, default_color, default_alpha in ring_groups:
            for idx in idx_list:
                idx = int(idx)
                disc = pv.Disc(center=self.p[idx], normal=self.d[idx],
                               inner=0, outer=disc_rad, c_res=20)
                if idx in partner_to_type:
                    color = RECEPTOR_PALETTE[partner_to_type[idx] % len(RECEPTOR_PALETTE)]
                    alpha = 0.75
                    width = 5.0
                else:
                    color = default_color
                    alpha = default_alpha
                    width = 1.0

                act = self.plotter.add_mesh(
                    disc, color=color, opacity=alpha,
                    line_width=width, edge_color='black',
                )
                self.actors_debugger.append(act)

        labels_pos, labels = [], []
        for r, lens_idx in enumerate(partners):
            if int(lens_idx) != target_idx:
                labels.append(f"R{r + 1}")
                labels_pos.append(self.p[int(lens_idx)])
        if labels:
            act = self.plotter.add_point_labels(
                labels_pos, labels, font_size=14, text_color='black',
                shape_color='white', shape_opacity=0.6,
            )
            self.actors_debugger.append(act)

        # Coloured receptor dots on the home ommatidium
        offsets = receptor_tip_offsets(self.model)[target_idx]
        max_off = float(np.max(np.linalg.norm(offsets, axis=1)))
        if max_off > 1e-8:
            dot_scale = disc_rad * 0.7 / max_off
            for r in range(self.R):
                color = RECEPTOR_PALETTE[r % len(RECEPTOR_PALETTE)]
                if np.linalg.norm(offsets[r]) > 1e-10:
                    dot_pos = target_pos + offsets[r] * dot_scale
                else:
                    dot_pos = target_pos
                act = self.plotter.add_points(
                    np.array([dot_pos]), color=color, point_size=12,
                    render_points_as_spheres=True, opacity=0.95,
                    smooth_shading=True,
                )
                self.actors_debugger.append(act)

        for a in self.actors_debugger:
            a.SetVisibility(self.state_bigpanel == 1)

    ##

    # Visibility helpers

    def _apply_alignment_visibility(self) -> None:
        for a in self.actors_alignment_smooth:
            a.SetVisibility(self.state_alignment_smoothed)
        for a in self.actors_alignment_raw:
            a.SetVisibility(not self.state_alignment_smoothed)

    def _apply_saccade_visibility(self) -> None:
        for a in self.actors_saccade_smooth:
            a.SetVisibility(self.state_saccade_smoothed)
        for a in self.actors_saccade_raw:
            a.SetVisibility(not self.state_saccade_smoothed)

    def _apply_bigpanel_visibility(self) -> None:
        s = self.state_bigpanel
        for a in self.actors_bundles:   a.SetVisibility(s == 0)
        for a in self.actors_debugger:  a.SetVisibility(s == 1)
        for a in self.actors_conflicts: a.SetVisibility(s == 2)

    # Heatmap scalars (cached at init)

    def _compute_collinearity(self) -> np.ndarray | None:
        """|raw flow projected to lens tangent . combed alignment phasor|."""
        if self.R <= 1:
            return None
        dots = self.d @ self.optic_flow_world
        raw_flow = self.optic_flow_world[None, :] - dots[:, None] * self.d
        norms = np.linalg.norm(raw_flow, axis=1, keepdims=True).clip(min=1e-8)
        raw_unit = raw_flow / norms
        return np.abs(np.einsum('ij,ij->i',
                                raw_unit,
                                self.result_smooth.alignment_phasor)).astype(np.float32)

    def _compute_smoothness(self) -> np.ndarray | None:
        """|raw saccade phasor . smoothed saccade phasor| (1.0 = unchanged)."""
        if self.R <= 1:
            return None
        return np.abs(np.einsum('ij,ij->i',
                                self.result_raw.saccade_phasor,
                                self.result_smooth.saccade_phasor)).astype(np.float32)

    ##

    # Key binds

    def _setup_keybindings(self) -> None:
        def toggle_alignment():
            self.state_alignment_smoothed = not self.state_alignment_smoothed
            self._apply_alignment_visibility()
            self._update_alignment_hint()
            self.plotter.render()

        def toggle_saccade():
            self.state_saccade_smoothed = not self.state_saccade_smoothed
            self._apply_saccade_visibility()
            self._update_saccade_hint()
            self.plotter.render()

        def cycle_bigpanel():
            self.state_bigpanel = (self.state_bigpanel + 1) % 3
            label = self._BIGPANEL_LABELS[self.state_bigpanel]
            print(f"[B] showing: {label}")
            self._apply_bigpanel_visibility()
            self._update_bigpanel_hint()
            self.plotter.render()

        def cycle_debugger():
            if self.state_bigpanel != 1:
                self.state_bigpanel = 1
                self._apply_bigpanel_visibility()
                self._update_bigpanel_hint()
            self._redraw_debugger()
            self.plotter.render()

        self.plotter.add_key_event('a', toggle_alignment)
        self.plotter.add_key_event('s', toggle_saccade)
        self.plotter.add_key_event('b', cycle_bigpanel)
        self.plotter.add_key_event('n', cycle_debugger)

        self._update_alignment_hint()
        self._update_saccade_hint()
        self._update_bigpanel_hint()

    # Hint helpers

    def _set_hint(self, subplot, name: str, text: str) -> None:
        self.plotter.subplot(*subplot)
        self.plotter.add_text(
            text, position='lower_left', font_size=8,
            color='black', font='courier', name=name,
        )

    def _update_alignment_hint(self) -> None:
        state = 'smoothed' if self.state_alignment_smoothed else 'raw'
        self._set_hint((0, 1), 'hint_a', f'[A] alignment: {state}')

    def _update_saccade_hint(self) -> None:
        state = 'smoothed' if self.state_saccade_smoothed else 'raw'
        self._set_hint((0, 3), 'hint_s', f'[S] saccade: {state}')

    def _update_bigpanel_hint(self) -> None:
        label = self._BIGPANEL_LABELS[self.state_bigpanel]
        self._set_hint(
            (1, 2), 'hint_bn',
            f'[B] showing: {label}\n[N] next random cartridge',
        )

    ##

    # Camera

    def _setup_camera(self) -> None:
        cam_dir = self.fwd + self.right + self.up
        cam_pos = cam_dir / np.linalg.norm(cam_dir) * (self.r_sphere * 6.5)
        for (row, col) in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]:
            self.plotter.subplot(row, col)
            self.plotter.camera_position = [cam_pos.tolist(), (0, 0, 0), self.up.tolist()]


##

if __name__ == "__main__":

    head_ptich = np.deg2rad(10.1)
    optic_flow = np.array([0.0, np.sin(head_ptich), np.cos(head_ptich)])

    kernel = drosophila_kernel()

    aligner = BundlesAligner(
        flow_direction=optic_flow,
        diagonal_strength=1.0,
        diagonal_angle_deg=45.0,
        alignment_smoothing_iterations=4,
        saccade_smoothing_iterations=15,
    )

    ra = CompoundEyeModel.from_file(
        'species_models/drosophila_custom.npz',
        kernel=kernel, orientation=aligner,
    )

    # ra = CompoundEyeModel.from_sphere(
    #     n=1600, kernel=kernel, orientation=aligner,
    # )

    EyeViewer(ra, aligner=aligner).show()