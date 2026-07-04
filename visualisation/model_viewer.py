import os
from datetime import datetime
from itertools import cycle
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from PIL import Image

from insectvision.compound_eyes import Model
from insectvision.compound_eyes.rhabdomeres import RHAB_COLOURS
from insectvision.compound_eyes.helpers.alignment import BundlesAligner
from insectvision.geometry.linalg import tangent_frames
from insectvision.utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, WORLD_BACKWARD
from insectvision.geometry.spherical import sphere_to_stereo
from insectvision.utils import norm_l2

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

def receptor_tip_offsets(model) -> np.ndarray:
    d = model.directions
    rec_dirs = model.buffer['curr_direction']
    axial = np.sum(rec_dirs * d[:, None, :], axis=2, keepdims=True)
    return -(rec_dirs - axial * d[:, None, :])


def radii_from_lattice(model) -> np.ndarray:
    """Per-lens display radius = mean distance to the lens's immediate lattice neighbours."""

    radii = np.zeros(model.shape[0], dtype=np.float32)
    for eye in model.eyes:
        if len(eye) == 0:
            continue
        result = eye.neighbours(query=eye.indices, k=6, immediate_only=True)
        if result.distances.size == 0:
            continue
        radii[eye.indices] = result.distances.mean(axis=1)
    radii[radii < 1e-9] = 0.001
    return radii


def make_ommatidia_lattice(model, delaunay_mesh: pv.PolyData) -> pv.PolyData:
    """
    Creates the dual Voronoi-like lattice mesh from the Delaunay triangulation.
    Each vertex in the Delaunay mesh (lens center) becomes a polygonal facet.
    """
    if delaunay_mesh.n_cells == 0 or delaunay_mesh.n_points == 0:
        return pv.PolyData()

    # The existing make_eye_mesh guarantees pure triangles -> reshape to (N, 4) and strip the '3' prefix
    faces = delaunay_mesh.faces.reshape(-1, 4)[:, 1:4]

    # Identify boundary vertices (avoid drawing malformed open facets)
    edge_counts = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_counts[e] = edge_counts.get(e, 0) + 1

    boundary_vertices = set()
    for e, count in edge_counts.items():
        if count == 1:  # an edge belonging to only 1 triangle == a boundary edge
            boundary_vertices.add(e[0])
            boundary_vertices.add(e[1])

    # Triangle centers become the corners of the hexagonal lattice
    pts = delaunay_mesh.points
    cell_centers = np.mean(pts[faces], axis=1)

    # Map each ommatidium (Delaunay vertex) to the triangles that contain it
    point_to_cells = {i: [] for i in range(delaunay_mesh.n_points)}
    for cell_idx, face in enumerate(faces):
        for pt_idx in face:
            point_to_cells[pt_idx].append(cell_idx)

    # Build the polygons
    dual_faces = []
    valid_ommatidia_indices = []
    right_vecs, up_vecs = tangent_frames(model.directions)

    for pt_idx, connected_cells in point_to_cells.items():

        # For now skipping 'is_edge' lenses... # TODO: figure out a way to correctly build edge lenses
        if model.is_edge[pt_idx] or pt_idx in boundary_vertices or len(connected_cells) < 3:
            continue

        # Get centers of surrounding triangles
        centers = cell_centers[connected_cells]
        center_vecs = centers - pts[pt_idx]

        # Project to tangent plane
        x = np.sum(center_vecs * right_vecs[pt_idx], axis=1)
        y = np.sum(center_vecs * up_vecs[pt_idx], axis=1)
        angles = np.arctan2(y, x)

        # Sort cell indices radially
        sorted_order = np.argsort(angles)
        sorted_cells = np.array(connected_cells)[sorted_order]

        # PyVista face list format: [N, v0, v1, ..., vN-1]
        dual_faces.append(len(sorted_cells))
        dual_faces.extend(sorted_cells)
        valid_ommatidia_indices.append(pt_idx)

    if not dual_faces:
        return pv.PolyData()

    lattice = pv.PolyData(cell_centers.astype(np.float32), faces=np.array(dual_faces, dtype=np.int_))

    # Store the original lens index so we can map heatmaps correctly
    lattice.cell_data['ommatidia_index'] = np.array(valid_ommatidia_indices, dtype=np.int_)
    return lattice


def make_delaunay_mesh(model) -> pv.PolyData:
    """
    Lens-indexed mesh: vertex i = lens i.
    """
    positions = model.positions
    if positions.shape[0] == 0:
        return pv.PolyData()

    all_faces = []
    for eye in model.eyes:
        if len(eye) < 3:
            continue

        global_idx = np.asarray(eye.indices, dtype=np.int_)
        eye_pos = positions[global_idx]

        eye_dirs = norm_l2(eye.directions)
        try:
            pts_2d, _, _, _ = sphere_to_stereo(eye_dirs)
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
        valid_mask = max_len < (typical_edge * 4.0)
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


##

class EyeViewer:
    """
    Multi-panel 3D viewer for a CompoundEyeModel.
    """

    def __init__(
        self,
        model: Model,
        aligner: BundlesAligner = None,
        optic_flow_world=None,
        sparsity: float = 0.0,
        debug_IDs=None
    ):

        self.model = model
        self.bundle = model.bundle
        self.N, self.R = model.shape
        self.sparsity = float(sparsity)

        self.right = np.asarray(WORLD_RIGHT, dtype=np.float32)
        self.up = np.asarray(WORLD_UP, dtype=np.float32)
        self.fwd = np.asarray(WORLD_FORWARD, dtype=np.float32)

        self._debug_ids = cycle(list(debug_IDs)) if debug_IDs else None

        if aligner is None:
            flow = optic_flow_world if optic_flow_world is not None else -self.fwd
            aligner = BundlesAligner(flow_direction=flow)
        self.aligner_smooth = aligner
        self.aligner_raw = BundlesAligner(
            flow_direction=aligner.flow_direction,
            diagonal_strength=aligner.diagonal_strength,
            diagonal_angle_deg=getattr(aligner, 'diagonal_angle_deg', 45.0),
            alignment_smoothing_iter=0,
            saccade_smoothing_iter=0,
            falloff=aligner.falloff,
            strength=aligner.strength,
        )
        self.optic_flow_world = aligner.flow_direction.astype(np.float32)

        self.result_smooth = self.aligner_smooth.compute(self.model)
        self.result_raw = self.aligner_raw.compute(self.model)

        # Cache geometry
        self.p = model.positions
        self.d = model.directions
        self.r_sphere = float(np.mean(np.linalg.norm(self.p, axis=1)))
        self.arrow_len = self.r_sphere * 0.08

        # Per-eye Delaunay mesh (vertex i = lens i)
        self.delaunay_mesh = make_delaunay_mesh(self.model)

        # Generate the ommatidia lattice
        self.lattice_mesh = make_ommatidia_lattice(self.model, self.delaunay_mesh)

        # Per-lens chirality
        self.chirality = self.model.chirality.astype(np.float32)

        # Per-lens lattice spacing
        self.disc_radii = radii_from_lattice(self.model) * 0.4

        # Heatmap scalars
        self.collinearity = self._compute_collinearity()
        self.smoothness = self._compute_smoothness()

        if self.R > 1 and self.model.neural_superposition:
            c_field = np.zeros(self.N, dtype=np.float32)

            # Unwired (voluntary edge drop / slack)
            unw = self.model.has_selfwires

            # True conflicts override unwired status (2, 3, 4)
            don = self.model.donation_conflicts
            rcv = self.model.receiving_conflicts

            c_field[unw] = 1.0
            c_field[don] = 2.0
            c_field[rcv] = 3.0
            c_field[don & rcv] = 4.0

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
        self.actors_binocular = []
        self.actors_edge = []
        self.actors_neighbours = []
        self.state_heatmap = 0  # 0: Collinearity, 1: Smoothness
        self.actors_collinearity = []
        self.actors_smoothness = []
        self.actors_ioa = []

        self.state_alignment_smoothed = True
        self.state_saccade_smoothed = True

        # Big panel state: 0 = bundles, 1 = debugger, 2 = conflicts heatmap
        self.state_bigpanel = 1
        self._BIGPANEL_LABELS = ['Bundles', 'Wiring debugger', 'Conflicts', 'Binocularity', 'Edges', 'Neighbours']

        self._debugger_subplot = None  # filled in show()

    ##

    def show(self):

        self.plotter = pv.Plotter(
            shape=(2, 4),
            groups=[(1, slice(2, 4))],
            window_size=list((2400, 1100)),
            border=True,
        )

        try:
            pv.set_new_attribute(self.plotter, 'pickpoint', None)   # needed to prevent errors on Windows and macOS...
            # ...in a try-except because Linux errors on allow_new_attributes() call...
        except AttributeError:
            pass

        self.plotter.set_background('white')

        self._debugger_subplot = (1, 2)

        # Panel layout
        panel_setups = [
            ((0, 0), "Optic Flow", self._add_optic_flow_panel),
            ((0, 1), "Alignment Axes  [A]", self._add_alignment_panel),
            ((0, 2), "Major Axes", self._add_major_axis_panel),
            ((0, 3), "Saccade Axes  [S]", self._add_saccade_panel),
            ((1, 0), "Alignment Heatmaps  [H]", self._add_heatmaps),
            ((1, 1), "IOA", self._add_ioa_panel),
            ((1, 2), "Bundles / Wiring / Conflicts  [B]  [N]", self._add_bigpanel),
        ]

        for (row, col), title, builder in panel_setups:
            self.plotter.subplot(row, col)
            self._common_scene(title)
            builder()

        # visibility states + linking
        self._apply_alignment_visibility()
        self._apply_saccade_visibility()
        self._apply_heatmap_visibility()
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

        if self.lattice_mesh.n_cells == 0:
            return

        m = self.lattice_mesh.copy()

        # lattice mesh represents lenses as faces (cells) instead of vertices (points)
        if scalars is not None:
            valid_indices = m.cell_data['ommatidia_index']
            m.cell_data['_scalar'] = scalars[valid_indices].astype(np.float32)
            kwargs = dict(
                scalars='_scalar', cmap=cmap,
                show_scalar_bar=(sbar_title is not None),
                show_edges=True, line_width=1.5, edge_color='#2c3e50',  # hexagon outlines
                ambient=0.3, diffuse=0.7,
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
                show_edges=True, line_width=1.0, edge_color='#666666',  # subtle grey lattice wireframe
                lighting=True,
            )

        # Extract boundary of the hexagons
        edges = m.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False
        )
        if edges.n_points > 0:
            self.plotter.add_mesh(
                edges, color='black', line_width=2.5, opacity=0.85, lighting=False
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
        if self.delaunay_mesh.n_cells == 0:
            return
        dots = self.d @ self.optic_flow_world
        v_proj = self.optic_flow_world[None, :] - dots[:, None] * self.d
        norms = np.linalg.norm(v_proj, axis=1, keepdims=True)
        v_proj = np.divide(v_proj, norms.clip(min=1e-8))

        m = self.delaunay_mesh.copy()
        m.point_data['OpticFlow'] = v_proj.astype(np.float32)
        arrows = self._glyph_arrows(m, 'OpticFlow')
        self.plotter.add_mesh(arrows, color='#da70d6',
                              ambient=0.4, diffuse=0.7, smooth_shading=True)

    def _add_alignment_panel(self) -> None:
        if self.delaunay_mesh.n_cells == 0 or self.R <= 1:
            return

        m_smooth = self.delaunay_mesh.copy()
        m_smooth.point_data['AlignmentSmooth'] = self.model.orientation_field
        g_smooth = self._glyph_phasors(m_smooth, 'AlignmentSmooth')
        a_smooth = self.plotter.add_mesh(g_smooth, color='green', line_width=2)
        self.actors_alignment_smooth.append(a_smooth)

        m_raw = self.delaunay_mesh.copy()
        m_raw.point_data['AlignmentRaw'] = self.result_raw.alignment_phasor.astype(np.float32)
        g_raw = self._glyph_phasors(m_raw, 'AlignmentRaw')
        a_raw = self.plotter.add_mesh(g_raw, color='#8c8c00', line_width=2)
        self.actors_alignment_raw.append(a_raw)

    def _add_major_axis_panel(self) -> None:
        if self.delaunay_mesh.n_cells == 0 or self.R <= 1:
            return

        for mask, color in [
            (self.chirality < 0, CHIRALITY_NEG_COLOR),
            (self.chirality > 0, CHIRALITY_POS_COLOR)
        ]:
            if not np.any(mask):
                continue

            pd = pv.PolyData(self.p[mask].astype(np.float32))
            pd.point_data['MajorAxis'] = -self.model.main_axis_field[mask]

            arrows = pd.glyph(geom=_arrow_template(), orient='MajorAxis', factor=self.arrow_len, scale=False)
            self.plotter.add_mesh(arrows, color=color, ambient=0.4, diffuse=0.7, smooth_shading=True)

    def _add_saccade_panel(self) -> None:
        if self.delaunay_mesh.n_cells == 0:
            return

        m_smooth = self.delaunay_mesh.copy()
        m_smooth.point_data['SaccadeSmooth'] = self.model.saccade_field
        g_smooth = self._glyph_arrows(m_smooth, 'SaccadeSmooth')
        a_smooth = self.plotter.add_mesh(g_smooth, color='red',
                                         ambient=0.4, diffuse=0.7, smooth_shading=True)
        self.actors_saccade_smooth.append(a_smooth)

        m_raw = self.delaunay_mesh.copy()
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

    def _add_heatmaps(self) -> None:

        if self.R <= 1 or self.lattice_mesh.n_cells == 0:
            return

        valid_indices = self.lattice_mesh.cell_data['ommatidia_index']

        # Collinearity mesh
        m_col = self.lattice_mesh.copy()
        m_col.cell_data['val'] = self.collinearity[valid_indices]
        act_col = self.plotter.add_mesh(
            m_col, scalars='val', cmap='inferno', clim=[0, 1],
            show_edges=True, edge_color='#666666', line_width=1.0,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Optic flow', 'position_x': 0.78, 'width': 0.18},
            ambient=0.3, diffuse=0.7
        )
        self.actors_collinearity.append(act_col)

        # Smoothness mesh
        m_sm = self.lattice_mesh.copy()
        m_sm.cell_data['val'] = self.smoothness[valid_indices]
        act_sm = self.plotter.add_mesh(
            m_sm, scalars='val', cmap='viridis', clim=[0.9, 1.0],
            show_edges=True, edge_color='#666666', line_width=1.0,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Saccades', 'position_x': 0.78, 'width': 0.18},
            ambient=0.3, diffuse=0.7
        )
        self.actors_smoothness.append(act_sm)

    def _add_ioa_panel(self) -> None:
        """Display the local Interommatidial Angle (sampling density)."""

        ioa_deg = np.degrees(self.model.interommatidial_angles[:, 0]) # minor IOA only

        self._add_eye_surface(
            scalars=ioa_deg, cmap='plasma_r',
            sbar_title='IOA (deg)',
        )

    def _add_binocular_view(self) -> None:
        """Render Left (Red), Right (Blue), and Binocular (Purple) zones."""

        # 0: Other, 1: Left, 2: Right, 3: Binocular
        side_field = np.zeros(self.N, dtype=np.float32)

        for eye in self.model.eyes:
            val = 1.0 if eye.side == 'left' else 2.0 if eye.side == 'right' else 0.0
            side_field[eye.indices] = val
        try:
            side_field[self.model.is_binocular] = 3.0
        except: pass

        m = self.lattice_mesh.copy()
        valid_indices = m.cell_data['ommatidia_index']
        m.cell_data['zones'] = side_field[valid_indices]

        zone_colors = ['#e0e0e0', '#ff4b4b', '#4b70ff', '#9b3ddc']

        act = self.plotter.add_mesh(
            m, scalars='zones', cmap=zone_colors, clim=[-0.5, 3.5],
            show_edges=True, edge_color='#666666',
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Zones (Red:L, Blue:R, Purple:Binoc)', 'n_labels': 5,
                'position_x': 0.70, 'position_y': 0.05, 'width': 0.28, 'height': 0.06, 'color': 'black',
            },
            ambient=0.4, diffuse=0.7,
        )
        self.actors_binocular.append(act)

    def _add_edginess_view(self) -> None:
        """
        Visualises edge detection logic.
        """

        m = self.lattice_mesh.copy()
        valid_indices = m.cell_data['ommatidia_index']
        m.cell_data['edge_mask'] = self.model.is_edge[valid_indices].astype(np.float32)

        edge_colors = ['#e0e0e0', '#ff00ff']
        act = self.plotter.add_mesh(
            m, scalars='edge_mask', cmap=edge_colors, clim=[-0.5, 1.5],
            show_edges=True, edge_color='#666666',
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Edges', 'n_labels': 0,
                'position_x': 0.70, 'position_y': 0.05, 'width': 0.28, 'height': 0.06, 'color': 'black',
            },
            ambient=0.4, diffuse=0.7,
        )
        self.actors_edge.append(act)

    def _add_neighb_count_view(self) -> None:
        """
        Visualises the 'neighbour_count' metadata field.
        """

        if self.lattice_mesh.n_cells == 0:
            return

        m = self.lattice_mesh.copy()
        counts = self.model.buffer['neighbour_count'][:, 0].astype(np.float32)
        valid_indices = m.cell_data['ommatidia_index']
        m.cell_data['counts'] = counts[valid_indices]

        count_colors = ['black', 'black', 'black', '#e74c3c', '#e67e22', '#f1c40f', '#ecf0f1', '#3498db', '#414181']

        act = self.plotter.add_mesh(
            m, scalars='counts', cmap=count_colors, clim=[-0.5, 8.5],
            show_edges=True, edge_color='#666666',
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Neighbours counts', 'n_labels': 9,
                'position_x': 0.70, 'position_y': 0.05, 'width': 0.28, 'height': 0.06, 'color': 'black',
            },
            ambient=0.4, diffuse=0.7, interpolate_before_map=False
        )
        self.actors_neighbours.append(act)

    def _add_bigpanel(self) -> None:
        """Bottom-right panel (wide)."""

        self._add_eye_surface(faint=True)

        # Mode 1: rhabdomere-tip bundles overview
        if self.R > 1:
            offsets = receptor_tip_offsets(self.model)
            max_off = float(np.max(np.linalg.norm(offsets, axis=2)))

            if max_off < 1e-5:
                # Symmetrical or perfectly fused bundle, no offset to scale
                tip_positions = np.broadcast_to(self.p[:, None, :], (self.N, self.R, 3))
                is_mirrored = np.zeros(self.N, dtype=bool)
            else:
                # Standard open rhabdom, scale up the pattern for visibility
                tip_scale = (self.r_sphere * 0.012) / max_off
                tip_positions = self.p[:, None, :] + offsets * tip_scale
                is_mirrored = self.model.chirality < 0

            i1, i2 = self.bundle.main_axis_indices

            groups = {
                'gold': [], 'red': [], 'white': [],
                'teal': [], 'royalblue': [],
            }
            for r in range(self.R):
                pts_r = tip_positions[:, r, :]
                if r == self.bundle.center_index:
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
        if self.conflicts_field is not None and self.lattice_mesh.n_cells > 0:
            m = self.lattice_mesh.copy()
            valid_indices = m.cell_data['ommatidia_index']
            m.cell_data['conflicts'] = self.conflicts_field[valid_indices].astype(np.float32)

            # Discrete colormap:
            # 0: Light gray (OK), 1: Blue (Unwired), 2: Yellow (Donation), 3: Cyan (Receiving), 4: Red (Both)
            cmap_colors = ['#e0e0e0', '#4287f5', '#f1c40f', '#00bcd4', '#e74c3c']

            act = self.plotter.add_mesh(
                m, scalars='conflicts',
                cmap=cmap_colors, clim=[-0.5, 4.5],
                show_edges=True, edge_color='#666666', line_width=1.0,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Conflicts (Gray:OK Blue:Drop Yel:Don Cy:Rcv Red:Both)', 'n_labels': 0,
                    'position_x': 0.65, 'position_y': 0.05,
                    'width': 0.33, 'height': 0.06, 'color': 'black',
                },
                ambient=0.4, diffuse=0.7,
            )
            self.actors_conflicts.append(act)

        # Mode 3: wiring debugger
        if getattr(self.model, '_cartridges_wired', False) and self.R > 1:
            self._redraw_debugger()

        # Mode 4: Binocular zone
        self._add_binocular_view()

        # Mode 5: Edges
        self._add_edginess_view()

        # Mode 6: Neighbours counts
        self._add_neighb_count_view()

    ##

    # Wiring debugger

    def _redraw_debugger(self) -> None:

        if self._debugger_subplot is not None:
            self.plotter.subplot(*self._debugger_subplot)

        for act in self.actors_debugger:
            self.plotter.remove_actor(act)
        self.actors_debugger.clear()

        if not self.model.neural_superposition or self.R <= 1:
            return

        target_idx = next(self._debug_ids) if self._debug_ids else np.random.randint(0, self.N)
        print(f'Target idx: {target_idx}')

        # Find the eye view containing the target ommatidium
        target_eye_id = int(self.model.eye_index[target_idx])
        target_eye = next(e for e in self.model.eyes if e.eye_index == target_eye_id)

        k_nb = min(40, len(target_eye))
        result = target_eye.neighbours(positions=self.p[target_idx][None, :], k=k_nb)
        neighb_indices = result.indices[0]

        partners = self.model.cartridge_map[target_idx]

        ioa_estimate = float(np.mean(result.distances[0][1:7])) if k_nb > 1 else 0.01
        d_rad = ioa_estimate * 0.45

        # Create a fast O(1) lookup: model ommatidium index -> mesh cell index
        valid_indices = self.lattice_mesh.cell_data['ommatidia_index']
        omm_to_cell = {omm: cell_id for cell_id, omm in enumerate(valid_indices)}

        # Background lenses
        bg_mask = ~np.isin(neighb_indices, partners) & (neighb_indices != target_idx)
        bg_cells = [omm_to_cell[idx] for idx in neighb_indices[bg_mask] if idx in omm_to_cell]

        if bg_cells:
            # Extract just those polygons to display faintly
            m_bg = self.lattice_mesh.extract_cells(bg_cells)
            self.actors_debugger.append(self.plotter.add_mesh(
                m_bg, color='black', opacity=0.1, show_edges=True, edge_color='gray'
            ))

        # Cartridge members
        member_cells = []
        member_scalars = []
        label_pts, label_txt = [], []

        for r_type, l_idx in enumerate(partners):
            l_idx = int(l_idx)
            if l_idx < 0: continue

            if l_idx in omm_to_cell:
                member_cells.append(omm_to_cell[l_idx])
                member_scalars.append(r_type)

            if l_idx != target_idx:
                label_pts.append(self.p[l_idx])
                label_txt.append(f"R{r_type + 1}")

                # Wiring lines (kept as is)
                line = pv.Line(self.p[target_idx], self.p[l_idx])
                self.actors_debugger.append(
                    self.plotter.add_mesh(line, color=RHAB_COLOURS[r_type], line_width=2, opacity=0.4))

        if member_cells:
            # Extract active polygons to color heavily
            m_mem = self.lattice_mesh.extract_cells(member_cells)
            m_mem.cell_data['r_type'] = np.array(member_scalars)
            self.actors_debugger.append(self.plotter.add_mesh(
                m_mem, scalars='r_type', cmap=RHAB_COLOURS,
                clim=[0, len(RHAB_COLOURS) - 1], show_scalar_bar=False,
                ambient=0.3, diffuse=0.8, show_edges=True, line_width=1.5, edge_color='black', opacity=0.8
            ))

        # Local rhabdomere bundle
        offsets = receptor_tip_offsets(self.model)[target_idx]
        tip_scale = (self.r_sphere * 0.05) / np.max(np.linalg.norm(offsets, axis=1)) * 0.2
        bundle_origin = self.p[target_idx] + (self.d[target_idx] * d_rad * 0.1)

        for r_type in range(self.R):
            tip_pos = bundle_origin + offsets[r_type] * tip_scale
            self.actors_debugger.append(self.plotter.add_mesh(
                pv.Sphere(radius=d_rad * 0.075, center=tip_pos),
                color=RHAB_COLOURS[r_type], lighting=True
            ))

        # Labels
        if label_pts:
            self.actors_debugger.append(self.plotter.add_point_labels(
                label_pts, label_txt, font_size=15, show_points=False,
                shape_color='white', shape_opacity=0.8
            ))

        self._apply_bigpanel_visibility()

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
        for a in self.actors_binocular: a.SetVisibility(s == 3)
        for a in self.actors_edge:      a.SetVisibility(s == 4)
        for a in self.actors_neighbours: a.SetVisibility(s == 5)

        for title, bar in self.plotter.scalar_bars.items():
            if 'Conflicts' in title:
                bar.SetVisibility(s == 2)
            elif 'Zones' in title:
                bar.SetVisibility(s == 3)
            elif 'Edges' in title:
                bar.SetVisibility(s == 4)
            elif 'Neighbours' in title:
                bar.SetVisibility(s == 5)

    def _apply_heatmap_visibility(self) -> None:
        h = self.state_heatmap
        for a in self.actors_collinearity:
            a.SetVisibility(h == 0)
        for a in self.actors_smoothness:
            a.SetVisibility(h == 1)

        for title, bar in self.plotter.scalar_bars.items():
            if 'Collinearity' in title:
                bar.SetVisibility(h == 0)
            if 'Smoothness' in title:
                bar.SetVisibility(h == 1)

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

        def toggle_heatmap():
            self.state_heatmap = (self.state_heatmap + 1) % 2
            self._apply_heatmap_visibility()
            self._update_heatmap_hint()
            self.plotter.render()

        def cycle_bigpanel():
            self.state_bigpanel = (self.state_bigpanel + 1) % len(self._BIGPANEL_LABELS)
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

        self.plotter.add_key_event('h', toggle_heatmap)
        self.plotter.add_key_event('a', toggle_alignment)
        self.plotter.add_key_event('s', toggle_saccade)
        self.plotter.add_key_event('b', cycle_bigpanel)
        self.plotter.add_key_event('n', cycle_debugger)
        self.plotter.add_key_event('Insert', self._dump_snapshots)

        self._update_alignment_hint()
        self._update_saccade_hint()
        self._update_heatmap_hint()
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

    def _update_heatmap_hint(self) -> None:
        mode = 'Collinearity' if self.state_heatmap == 0 else 'Smoothness'
        self._set_hint((1, 0), 'hint_h', f'[H] heatmap: {mode}')

    def _update_bigpanel_hint(self) -> None:
        label = self._BIGPANEL_LABELS[self.state_bigpanel]
        self._set_hint(
            (1, 2), 'hint_bn',
            f'[B] showing: {label}\n[N] next random cartridge',
        )

    def _dump_snapshots(self):
        """Captures a high-res screenshot and saves each panel as a separate PNG."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"eye_viewer_snaps_{timestamp}"
        os.makedirs(folder, exist_ok=True)

        print(f"Exporting snapshots to {folder}...")

        img_array = self.plotter.screenshot(None, return_img=True, scale=3)
        full_img = Image.fromarray(img_array)

        # grid dimensions
        rows, cols = 2, 4
        w, h = full_img.size
        cw, rh = w // cols, h // rows

        # Define panels to extract
        panels = [
            (0, 0, 1, "01_optic_flow"),
            (0, 1, 1, "02_alignment_axes"),
            (0, 2, 1, "03_major_axes"),
            (0, 3, 1, "04_saccade_axes"),
            (1, 0, 1, "05_heatmaps"),
            (1, 1, 1, "06_ioa"),
            (1, 2, 2, "07_big_panel"),  # merged panel is 2 columns wide
        ]

        for r, c, span, name in panels:
            left = c * cw
            top = r * rh
            right = (c + span) * cw
            bottom = (r + 1) * rh

            panel_img = full_img.crop((left, top, right, bottom))
            path = os.path.join(folder, f"{name}.png")
            panel_img.save(path)
            print(f"  > Saved {path}")

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
    from insectvision.compound_eyes.rhabdomeres import drosophila_bundle, honeybee_bundle

    # model = Model.from_sphere(
    #     n=1600, bundle=drosophila_bundle(), orientation=aligner, neural_superposition=True
    # )

    # ----------------------------------------------------------------------

    droso_head_ptich = np.deg2rad(10.1)     # drosophila head pitch in flight

    aligner = BundlesAligner(
        equatorial_discontinuity=True,  # important for drosophila rhabdomere bundles alignment
        flow_direction=np.array([0.0, np.sin(droso_head_ptich), np.cos(droso_head_ptich)]),   # optic flow in flight
        diagonal_strength=1.0,
        diagonal_angle_deg=45.0,
        alignment_smoothing_iter=4,
        saccade_smoothing_iter=5,
    )

    model = Model.from_file(
        'assets/drosophila_scaffold.npz',
        bundle=drosophila_bundle(),
        orientation=aligner,
        neural_superposition=True,    # superposition eyes
    )

    # ----------------------------------------------------------------------

    # aligner = BundlesAligner(
    #     equatorial_discontinuity=False,  # no equatorial discontinuity in the bee
    #     flow_direction=WORLD_BACKWARD,   # optic flow in flight
    #     diagonal_strength=1.0,
    #     diagonal_angle_deg=45.0,
    #     alignment_smoothing_iterations=4,
    #     saccade_smoothing_iterations=5,
    # )
    #
    # model = Model.from_file(
    #     'assets/honeybee_scaffold_s10.npz',
    #     bundle=honeybee_bundle(),
    #     orientation=aligner,
    #     neural_superposition=False,     # Apposition eyes
    #     lattice_beta=0.9                # The Stürzl procedural lattice has zones artifacts, lowering the beta helps
    # )

    # ----------------------------------------------------------------------

    model.refine_superposition(smooth_iters=2, relax=0.5, adjust_scale=True)

    viewer = EyeViewer(model, aligner=aligner)
    viewer.show()

