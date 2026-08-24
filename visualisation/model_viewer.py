import os
from datetime import datetime
from itertools import cycle
from typing import Optional, Sequence, Tuple
import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from PIL import Image
from scipy.spatial import Delaunay

from rhabdoforge.types import WORLD_BACKWARD, WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, RHAB_COLOURS
from rhabdoforge.utils import norm_l2
from rhabdoforge.geometry.linalg import tangent_frames
from rhabdoforge.geometry.spherical import sphere_to_stereo
from rhabdoforge.compound_eyes import Model
from rhabdoforge.compound_eyes.helpers.alignment import BundlesAligner


# TODO: Maybe move some of these colours to the constants / types file
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


##

# TODO: Move the meshing functions to the mesh utilities and use the mesh in the renderer?

def make_hex_mesh(
        normals: np.ndarray,
        delaunay_points: np.ndarray,
        delaunay_faces: np.ndarray,
        delaunay_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if len(delaunay_points) == 0 or len(delaunay_faces) == 0:
        return np.array([]), np.array([]), np.array([])

    faces = delaunay_faces[:, 1:4]
    pts = delaunay_points
    N_mesh = len(pts)

    # Tangent frames for all real points
    real_mask = (delaunay_indices != -1)
    real_normals = normals[delaunay_indices[real_mask]]
    right_vecs, up_vecs = tangent_frames(real_normals)

    # Expand to full mesh (zeros for ghosts)
    right_full = np.zeros((N_mesh, 3), dtype=np.float32)
    up_full = np.zeros((N_mesh, 3), dtype=np.float32)
    right_full[real_mask] = right_vecs
    up_full[real_mask] = up_vecs

    # Triangle centres (Voronoi vertices)
    cell_centers = np.mean(pts[faces], axis=1)

    # Build point‑to‑cell adjacency via sorting
    point_indices = faces.flatten()
    cell_indices = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(point_indices)
    sorted_points = point_indices[order]
    sorted_cells = cell_indices[order]

    # Unique points and their ranges in the sorted arrays
    unique_points, first_idx, counts = np.unique(
        sorted_points, return_index=True, return_counts=True
    )
    real_unique = unique_points[real_mask[unique_points]]
    # Map real point index to its position in the unique array
    real_pos = np.searchsorted(unique_points, real_unique)

    # Build dual faces for each real point
    dual_faces = []
    valid_global_indices = []

    for pos, pt_idx in zip(real_pos, real_unique):
        start = first_idx[pos]
        end = start + counts[pos]
        connected_cells = sorted_cells[start:end]

        if len(connected_cells) < 3:
            continue

        centers = cell_centers[connected_cells]
        vecs = centers - pts[pt_idx]

        # Project onto tangent plane of this point
        x = np.sum(vecs * right_full[pt_idx], axis=1)
        y = np.sum(vecs * up_full[pt_idx], axis=1)
        angles = np.arctan2(y, x)
        sorted_cells_local = connected_cells[np.argsort(angles)]

        dual_faces.append(len(sorted_cells_local))
        dual_faces.extend(sorted_cells_local)
        valid_global_indices.append(delaunay_indices[pt_idx])

    points = cell_centers.astype(np.float32)
    faces = np.array(dual_faces, dtype=np.int_)
    indices = np.array(valid_global_indices, dtype=np.int_)

    return points, faces, indices


def make_delaunay_mesh(model: 'Model', ghost_rows: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-eye Delaunay mesh from each eye's real + virtual points.
    """
    if model.positions.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    all_points, all_global_indices, all_faces = [], [], []
    offset = 0

    for eye in model.eyes:
        if len(eye) < 3:
            continue

        indices = eye.indices
        real_pos = eye.positions
        real_dirs = eye.directions

        if ghost_rows > 0:
            eye.build_ghosts(n_rows=ghost_rows)  # honour the requested ring count
            gpos, gdirs = eye.ghost_positions, eye.ghost_directions
        else:
            gpos = np.zeros((0, 3))
            gdirs = np.zeros((0, 3))

        # TODO: use combined_cloud helper
        # TODO: Maybe run a quick spring relax ????
        if len(gpos):
            combined_pos = np.vstack([real_pos, gpos]).astype(np.float32)
            combined_dirs = np.vstack([real_dirs, gdirs])
            mapping = np.concatenate([indices, -np.ones(len(gpos), dtype=np.int_)])
        else:
            combined_pos = real_pos.astype(np.float32)
            combined_dirs = real_dirs
            mapping = indices

        # Delaunay in the locked stereo plane of the (real + ghost) directions
        pts_2d, *_ = sphere_to_stereo(norm_l2(combined_dirs))
        tri = Delaunay(pts_2d)

        # Prune oversized boundary triangles (keeps the hull tight to the ghost ring)
        p2d_f = pts_2d[tri.simplices]
        l2d_max = np.max(np.linalg.norm(p2d_f - np.roll(p2d_f, 1, axis=1), axis=2), axis=1)
        valid_faces = tri.simplices[l2d_max < (np.median(l2d_max) * 4.0)]

        all_points.append(combined_pos)
        all_global_indices.append(mapping)
        all_faces.append(valid_faces + offset)
        offset += len(combined_pos)

    if not all_points:
        return np.array([]), np.array([]), np.array([])

    all_points = np.vstack(all_points)
    all_indices = np.concatenate(all_global_indices)
    all_faces_stacked = np.vstack(all_faces)

    faces_pv = np.empty((len(all_faces_stacked), 4), dtype=np.int_)
    faces_pv[:, 0] = 3
    faces_pv[:, 1:] = all_faces_stacked

    return all_points, faces_pv, all_indices


def _arrow_template() -> pv.PolyData:
    global _ARROW_TEMPLATE
    if _ARROW_TEMPLATE is None:
        _ARROW_TEMPLATE = pv.Arrow(tip_radius=0.08, shaft_radius=0.03, tip_length=0.25)
    return _ARROW_TEMPLATE


##

class EyeViewer:
    """
    Multi-panel 3D viewer for a CompoundEyeModel.
    """

    def __init__(
        self,
        model: Model,
        aligner: Optional[BundlesAligner] = None,
        optic_flow_world: Optional[ArrayLike] = None,
        debug_IDs: Optional[Sequence[int]] = None
    ):

        self.model = model
        self.bundle = model.bundle
        self.N, self.R = model.shape

        self._debug_ids = cycle(list(debug_IDs)) if debug_IDs else None

        if aligner is None:
            flow = optic_flow_world if optic_flow_world is not None else WORLD_BACKWARD
            aligner = BundlesAligner(ref_direction=flow)

        self.aligner = aligner

        self.aligner_raw = BundlesAligner(
            ref_direction=self.aligner.ref_direction,
            combing_strength=self.aligner.combing_strength,
            combing_angle_deg=self.aligner.combing_angle_deg,
            combing_falloff=self.aligner.combing_falloff,
            alignment_smoothing_iter=0,
            saccade_smoothing_iter=0,
            equatorial_discontinuity=self.aligner.equatorial_discontinuity,
            flip_polarity=self.aligner.flip_polarity,
        )

        self.optic_flow_world = self.aligner.ref_direction.astype(np.float32)
        self.result_raw = self.aligner_raw.compute(self.model, verbose=False)

        # Cache geometry
        self.p = model.positions
        self.d = model.directions
        self.r_sphere = float(np.mean(np.linalg.norm(self.p, axis=1)))
        self.arrow_len = self.r_sphere * 0.08

        # Build Delaunay mesh (real + ghosts)
        delaunay_points, delaunay_faces, delaunay_indices = make_delaunay_mesh(self.model, ghost_rows=3)

        if np.any(delaunay_points) and np.any(delaunay_faces):
            self.delaunay_mesh_full = pv.PolyData(delaunay_points, faces=delaunay_faces.flatten())
        else:
            self.delaunay_mesh_full = pv.PolyData()

        # Delaunay mesh containing only real points (for glyphs, scalars, etc.)
        self._real_mask = delaunay_indices != -1
        self.delaunay_mesh_real = pv.PolyData(delaunay_points[self._real_mask])

        # Build hexagonal lattice from the combined Delaunay (real + ghosts)
        hexmesh_points, hexmesh_faces, hexmesh_indices = make_hex_mesh(self.model.directions, delaunay_points, delaunay_faces, delaunay_indices)

        if np.any(delaunay_points) and np.any(delaunay_faces):
            self.lattice_mesh = pv.PolyData(hexmesh_points, faces=hexmesh_faces)
        else:
            self.lattice_mesh = pv.PolyData()
        self.lattice_mesh.cell_data['ommatidia_index'] = hexmesh_indices

        # Per-lens chirality
        self.chirality = self.model.chirality.astype(np.float32)

        self.saccade_fields = {
            (True, True): self.model.saccade_field,             # applied (A1, S1)
            (False, False): self.result_raw.saccade_phasor,     # neither (A0, S0)
            (True, False): self.aligner._saccade_from_major(self.model, self.model.major_axis_field, skip_smooth=True),
            (False, True): self.aligner._saccade_from_major(self.model, self.result_raw.major_axis, skip_smooth=False),
        }

        # Heatmap scalars
        self.collinearity = self._compute_collinearity()
        self.smoothness = self._compute_smoothness()

        if self.R > 1 and self.model.neural_superposition:
            c_field = np.zeros(self.N, dtype=np.float32)

            # Unwired (voluntary edge drop / slack)
            unw = self.model.unwired_slots.any(axis=-1)

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
        self.actors_meshing_debug = []
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
        self.actors_alignment_smooth = []
        self.actors_alignment_raw = []
        self.actors_major_smooth = []
        self.actors_major_raw = []
        self.actors_saccade = {}
        self.actors_decorations = []

        self.state_alignment_smoothed = True
        self.state_saccade_smoothed = True
        self.state_meshing_debug = False

        self.state_refined = False
        self._unrefined_state = None
        self._refined_state = None

        # Big panel state: 0 = bundles, 1 = debugger, 2 = conflicts heatmap
        self.state_bigpanel = 1
        self._BIGPANEL_LABELS = [
            'Bundles',
            'Wiring debugger',
            'Conflicts',
            'Binocularity',
            'Edges',
            'Neighbours',
        ]
        self._debugger_subplot = None  # filled in show()

    def _add_debug_meshing(self, subplot=None):
        """Temporarily display ghost points and Delaunay edges."""

        pts = self.delaunay_mesh_real.points
        pts_w_ghosts = self.delaunay_mesh_full.points
        faces = self.delaunay_mesh_full.faces.reshape(-1, 4)[:, 1:4]

        # Real points (blue) and ghost points (red)
        ghost_mask = ~self._real_mask

        if subplot is not None:
            self.plotter.subplot(*subplot)

        # Plot real points
        real_points_actors = self.plotter.add_points(pts, color='blue', point_size=10,
                                                     render_points_as_spheres=True)
        self.actors_meshing_debug.append(real_points_actors)

        # Plot ghost points
        if np.any(ghost_mask):
            ghost_points_actors = self.plotter.add_points(pts_w_ghosts[ghost_mask], color='red', point_size=10,
                                                          render_points_as_spheres=True)
            self.actors_meshing_debug.append(ghost_points_actors)

        # Draw Delaunay edges (only those connecting real–ghost or real–real)
        edges = set()
        for tri in faces:
            for i in range(3):
                e = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                edges.add(e)
        edge_lines = np.stack(list(edges))

        if np.any(edge_lines):
            edge_lines_actors = self.plotter.add_lines(pts_w_ghosts[edge_lines].reshape(-1, 3), color='black', width=1)
            self.actors_meshing_debug.append(edge_lines_actors)

    # Per-panel fluff (planes, flow indicator, title, axes)

    def _common_scene(self, title: str) -> None:
        self.plotter.add_text(title, font_size=9, color='black')

        # Equatorial and sagittal ref planes (head frame)
        plane_size = self.r_sphere * 2.6
        eq = pv.Plane(center=(0, 0, 0), direction=WORLD_UP, i_size=plane_size, j_size=plane_size)
        sag = pv.Plane(center=(0, 0, 0), direction=WORLD_RIGHT, i_size=plane_size, j_size=plane_size)

        a1 = self.plotter.add_mesh(eq, color=EQUATOR_COLOR, opacity=PLANE_OPAC)
        a2 = self.plotter.add_mesh(sag, color=SAGITTAL_COLOR, opacity=PLANE_OPAC)

        # Optic flow direction (big arrow at the head front)
        flow_arrow = pv.Arrow(
            start=(-1.8 * self.r_sphere * self.optic_flow_world).tolist(),
            direction=self.optic_flow_world.tolist(),
            scale=self.r_sphere * 0.6,
        )
        a3 = self.plotter.add_mesh(flow_arrow, color=FLOW_COLOR, lighting=False, opacity=0.85)

        self.actors_decorations.extend([a1, a2, a3])

        self.plotter.add_axes(interactive=False)

    def _add_eye_surface(
        self,
        scalars: Optional[np.ndarray] = None,
        cmap=None,
        clim=None,
        sbar_title: Optional[str] = None,
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
        return mesh.glyph(geom=_arrow_template(), orient=orient_name, factor=self.arrow_len, scale=False)

    def _glyph_phasors(self, mesh: pv.PolyData, orient_name: str) -> pv.PolyData:

        global _PHASOR_TEMPLATE
        if _PHASOR_TEMPLATE is None:
            _PHASOR_TEMPLATE = pv.Line(pointa=(-0.5, 0, 0), pointb=(0.5, 0, 0))

        return mesh.glyph(geom=_PHASOR_TEMPLATE, orient=orient_name, factor=self.arrow_len, scale=False)

    def _add_optic_flow_panel(self) -> None:
        if self.delaunay_mesh_real.n_cells == 0:
            return
        dots = self.d @ self.optic_flow_world
        v_proj = self.optic_flow_world[None, :] - dots[:, None] * self.d
        norms = np.linalg.norm(v_proj, axis=1, keepdims=True)
        v_proj = np.divide(v_proj, norms.clip(min=1e-8))

        m = self.delaunay_mesh_real.copy()
        m.point_data['OpticFlow'] = v_proj.astype(np.float32)
        arrows = self._glyph_arrows(m, 'OpticFlow')
        self.plotter.add_mesh(arrows, color='#da70d6',
                              ambient=0.4, diffuse=0.7, smooth_shading=True)

    def _add_alignment_panel(self) -> None:
        if self.delaunay_mesh_real.n_cells == 0 or self.R <= 1:
            return

        m_smooth = self.delaunay_mesh_real.copy()
        m_smooth.point_data['AlignmentSmooth'] = self.model.reference_field
        g_smooth = self._glyph_phasors(m_smooth, 'AlignmentSmooth')
        a_smooth = self.plotter.add_mesh(g_smooth, color='green', line_width=2)
        self.actors_alignment_smooth.append(a_smooth)

        m_raw = self.delaunay_mesh_real.copy()
        m_raw.point_data['AlignmentRaw'] = self.result_raw.flow_line.astype(np.float32)
        g_raw = self._glyph_phasors(m_raw, 'AlignmentRaw')
        a_raw = self.plotter.add_mesh(g_raw, color='#8c8c00', line_width=2)
        self.actors_alignment_raw.append(a_raw)

    def _add_major_axis_panel(self) -> None:
        if self.delaunay_mesh_real.n_cells == 0 or self.R <= 1:
            return
        for field, actor_list in [
            (self.model.major_axis_field, self.actors_major_smooth),
            (self.result_raw.major_axis, self.actors_major_raw),
        ]:
            for mask, color in [
                (self.chirality < 0, CHIRALITY_NEG_COLOR),
                (self.chirality > 0, CHIRALITY_POS_COLOR),
            ]:
                if not np.any(mask):
                    continue
                pd = pv.PolyData(self.p[mask].astype(np.float32))
                pd.point_data['MajorAxis'] = -field[mask]
                arrows = pd.glyph(geom=_arrow_template(), orient='MajorAxis', factor=self.arrow_len, scale=False)
                a = self.plotter.add_mesh(arrows, color=color, ambient=0.4, diffuse=0.7, smooth_shading=True)
                actor_list.append(a)

    def _add_saccade_panel(self) -> None:

        if self.delaunay_mesh_real.n_cells == 0:
            return

        for (a_on, s_on), field in self.saccade_fields.items():
            m = self.delaunay_mesh_real.copy()
            key = 'Saccade_%d%d' % (int(a_on), int(s_on))
            m.point_data[key] = field.astype(np.float32)
            if s_on:
                g = self._glyph_arrows(m, key)
                actor = self.plotter.add_mesh(g, color='red', ambient=0.4, diffuse=0.7, smooth_shading=True)
            else:
                g = self._glyph_phasors(m, key)
                actor = self.plotter.add_mesh(g, color='#FF94BD', line_width=2)
            self.actors_saccade[(a_on, s_on)] = actor

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

    def _redraw_bundles(self) -> None:
        if self._debugger_subplot is not None:
            self.plotter.subplot(*self._debugger_subplot)

        for act in self.actors_bundles:
            self.plotter.remove_actor(act)
        self.actors_bundles.clear()

        # Mode 1: rhabdomere-tip bundles overview
        if self.R > 1:
            tips_rel = self.model.rhabdomeres.relative_positions.reshape(self.N, self.R, 3) * 1.3
            pts_all = self.model.positions[:, None, :] + tips_rel

            is_mirrored = self.model.chirality < 0
            groups = {'gold': [], 'red': [], 'white': [], 'teal': [], 'royalblue': []}

            for r in range(self.R):
                pts_r = pts_all[:, r, :]

                if r == self.bundle.center_index:
                    groups['gold'].append(pts_r)
                else:
                    groups['teal'].append(pts_r[~is_mirrored])
                    groups['royalblue'].append(pts_r[is_mirrored])

            for color, pts_list in groups.items():
                arrays = [a for a in pts_list if len(a) > 0]
                if not arrays:
                    continue
                act = self.plotter.add_points(np.vstack(arrays), color=color, point_size=4, render_points_as_spheres=True)
                act.SetVisibility(self.state_bigpanel == 0)
                self.actors_bundles.append(act)

    def _add_bigpanel(self) -> None:
        """Bottom-right panel (wide)."""

        self._add_eye_surface(faint=True)
        self._add_debug_meshing()

        # Mode 1: rhabdomere-tip bundles overview
        self._redraw_bundles()

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
        if self.model.neural_superposition and self.R > 1:
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
        """
        Visualizes the neural superposition wiring for a single cartridge.
        Uses CartridgeView to find donor rhabdomeres and their world-space positions.
        """
        if self._debugger_subplot is not None:
            self.plotter.subplot(*self._debugger_subplot)

        # Clear previous debugger actors
        for act in self.actors_debugger:
            self.plotter.remove_actor(act)
        self.actors_debugger.clear()

        if not self.model.neural_superposition or self.R <= 1:
            return

        target_idx = next(self._debug_ids) if self._debug_ids else np.random.randint(0, self.N)

        # Target ommatidium for the tip origins
        target_omm = self.model.ommatidia[target_idx]
        cartridge_tips = target_omm.rhabdomeres.positions

        # Wiring map
        donor_omm_indices = self.model.cartridge_map[target_idx]

        target_eye = self.model.eyes[int(self.model.eye_index[target_idx])]

        k_nb = min(42, len(target_eye))
        nb_res = target_eye.neighbours(query=[target_idx], k=k_nb)
        neighb_indices = nb_res.indices[0]

        valid_indices = self.lattice_mesh.cell_data['ommatidia_index']
        omm_to_cell = {omm: cell_id for cell_id, omm in enumerate(valid_indices)}

        # Render background lenses (faint)
        bg_mask = ~np.isin(neighb_indices, donor_omm_indices) & (neighb_indices != target_idx)
        bg_cells = [omm_to_cell[idx] for idx in neighb_indices[bg_mask] if idx in omm_to_cell]
        if bg_cells:
            m_bg = self.lattice_mesh.extract_cells(bg_cells)
            self.actors_debugger.append(self.plotter.add_mesh(
                m_bg, color='black', opacity=0.05, show_edges=True, edge_color='gray'
            ))

        # Render donor lenses and wiring

        label_pts, label_txt = [], []

        for r_slot, donor_idx in enumerate(donor_omm_indices):
            donor_idx = int(donor_idx)
            if donor_idx < 0:
                continue  # unwired slots

            # Draw wiring line
            target_ommatidium = self.model.ommatidia[target_idx]
            tip_pos = target_ommatidium.rhabdomeres.positions[r_slot]
            donor_pos = self.p[donor_idx]

            line = pv.Line(tip_pos, donor_pos)
            self.actors_debugger.append(
                self.plotter.add_mesh(line, color=RHAB_COLOURS[r_slot], line_width=3, opacity=0.6)
            )
            label_pts.append(donor_pos)
            label_txt.append(f"R{r_slot + 1}")

            # Colour donor neighbours
            if donor_idx in omm_to_cell:
                cell_id = omm_to_cell[donor_idx]
                m_single = self.lattice_mesh.extract_cells([cell_id])
                donor = self.plotter.add_mesh(
                    m_single,
                    color=RHAB_COLOURS[r_slot],
                    show_edges=True,
                    line_width=2,
                    edge_color='black',
                    opacity=0.7,
                    ambient=0.3,
                    diffuse=0.8
                )
                self.actors_debugger.append(donor)

        # Render rhabdomere tips
        tip_radius = np.median(nb_res.distances) * 0.015
        for r_slot in range(self.R):
            if donor_omm_indices[r_slot] == -1:
                col = 'white'
            else:
                col = RHAB_COLOURS[r_slot]

            tip_sphere = pv.Sphere(radius=tip_radius, center=cartridge_tips[r_slot])
            self.actors_debugger.append(self.plotter.add_mesh(
                tip_sphere, color=col, lighting=True
            ))

        # Labels and target marker
        if label_pts:
            self.actors_debugger.append(self.plotter.add_point_labels(
                label_pts, label_txt, font_size=14, show_points=False,
                shape_color='white', shape_opacity=0.8
            ))

        self._apply_bigpanel_visibility()

    ##

    # Visibility helpers

    def _apply_meshing_debug_visibility(self) -> None:
        for a in self.actors_meshing_debug:
            a.SetVisibility(self.state_meshing_debug)

    def _apply_alignment_visibility(self) -> None:
        for a in self.actors_alignment_smooth:
            a.SetVisibility(self.state_alignment_smoothed)
        for a in self.actors_alignment_raw:
            a.SetVisibility(not self.state_alignment_smoothed)
        for a in self.actors_major_smooth:
            a.SetVisibility(self.state_alignment_smoothed)
        for a in self.actors_major_raw:
            a.SetVisibility(not self.state_alignment_smoothed)

    def _apply_saccade_visibility(self) -> None:
        active = (self.state_alignment_smoothed, self.state_saccade_smoothed)
        for key, actor in self.actors_saccade.items():
            actor.SetVisibility(key == active)

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

    def _compute_collinearity(self) -> np.ndarray:

        # Project flow into (right, up) tangent coords of every ommatidium
        dot_r = self.model.right @ self.optic_flow_world
        dot_u = self.model.up @ self.optic_flow_world

        # Local unit flow vector in tangent plane
        mag = np.hypot(dot_r, dot_u).clip(min=1e-8)
        local_flow = np.stack([dot_r / mag, dot_u / mag], axis=-1)

        # compare to alignment field (already in local tangent coords)
        alignment_local = self.model.reference_field_local
        return np.abs(np.einsum('ij,ij->i', local_flow, alignment_local))

    def _compute_smoothness(self) -> np.ndarray | None:
        """|raw saccade phasor . smoothed saccade phasor| (1.0 = unchanged)."""
        if self.R <= 1:
            return None
        s = np.einsum('ij,ij->i', self.result_raw.saccade_phasor, self.model.saccade_field)
        return np.abs(s).astype(np.float32)

    ##

    # Key binds

    def _setup_keybindings(self) -> None:
        def toggle_debug():
            self.state_meshing_debug = not self.state_meshing_debug
            self._apply_meshing_debug_visibility()
            self.plotter.render()

        def toggle_alignment():
            self.state_alignment_smoothed = not self.state_alignment_smoothed
            self._apply_alignment_visibility()
            self._apply_saccade_visibility()  # saccade rides on the major axis
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

        def toggle_refinement():
            if not self.model.neural_superposition or self.R <= 1:
                return

            if self._unrefined_state is None:
                print("Computing bundle refinement...")
                self._unrefined_state = {
                    'rest_offsets': self.model.buffer['rest_offset'].copy(),
                    'chi': self.model.chi.copy(),
                    'saccade_dxdy': self.model.buffer['saccade_dxdy'].copy(),
                    'curr_direction': self.model.buffer['curr_direction'].copy(),
                }
                self.model.refine_superposition(smooth_iters=3, adjust_scale=True, adjust_anisotropy=True, rewire=True)
                self._refined_state = {
                    'rest_offsets': self.model.buffer['rest_offset'].copy(),
                    'chi': self.model.chi.copy(),
                    'saccade_dxdy': self.model.buffer['saccade_dxdy'].copy(),
                    'curr_direction': self.model.buffer['curr_direction'].copy(),
                }
                self.state_refined = True
            else:
                self.state_refined = not self.state_refined

            state = self._refined_state if self.state_refined else self._unrefined_state
            self.model.buffer['rest_offset'] = state['rest_offsets']
            self.model.chi = state['chi']
            self.model.buffer['saccade_dxdy'] = state['saccade_dxdy']
            self.model.buffer['curr_direction'] = state['curr_direction']

            self._redraw_bundles()
            self._redraw_debugger()
            self._update_bigpanel_hint()
            self.plotter.render()
            print(f"Refinement state: {'ON' if self.state_refined else 'OFF'}")

        self.plotter.add_key_event('h', toggle_heatmap)
        self.plotter.add_key_event('r', toggle_refinement)
        self.plotter.add_key_event('a', toggle_alignment)
        self.plotter.add_key_event('s', toggle_saccade)
        self.plotter.add_key_event('d', toggle_debug)
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
        s_state = 'smoothed' if self.state_saccade_smoothed else 'raw'
        a_state = 'smoothed' if self.state_alignment_smoothed else 'raw'
        self._set_hint((0, 1), 'hint_a', f'[A] Alignment: {a_state}')
        self._set_hint((0, 2), 'hint_a', f'[A] Major axis from {a_state} alignment')
        self._set_hint((0, 3), 'hint_s', f'[S] Saccade: {s_state} (from {a_state} alignment)')

    def _update_saccade_hint(self) -> None:
        s_state = 'smoothed' if self.state_saccade_smoothed else 'raw'
        a_state = 'smoothed' if self.state_alignment_smoothed else 'raw'
        self._set_hint((0, 3), 'hint_s', f'[S] Saccade: {s_state} (from {a_state} alignment)')

    def _update_heatmap_hint(self) -> None:
        mode = 'Collinearity' if self.state_heatmap == 0 else 'Smoothness'
        self._set_hint((1, 0), 'hint_h', f'[H] heatmap: {mode}')

    def _update_bigpanel_hint(self) -> None:
        label = self._BIGPANEL_LABELS[self.state_bigpanel]
        ref_str = 'ON' if self.state_refined else 'OFF'
        self._set_hint(
            (1, 2), 'hint_bn',
            f'[B] showing: {label}\n[N] next random cartridge\n[R] toggle refinement: {ref_str}',
        )

    def _dump_snapshots(self):
        """Captures a high-res screenshot and saves each panel as a separate PNG."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"eye_viewer_snaps_{timestamp}"
        os.makedirs(folder, exist_ok=True)

        for actor in self.actors_decorations:
            actor.SetVisibility(False)

        print(f"Exporting snapshots to {folder}...")

        img_array = self.plotter.screenshot(
            None,
            return_img=True,
            scale=3,
            transparent_background=True
        )
        full_img = Image.fromarray(img_array)

        for actor in self.actors_decorations:
            actor.SetVisibility(True)

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
            (1, 2, 2, "07_big_panel"),
        ]

        for r, c, span, name in panels:
            offset_x = (span - 1) * (cw / 2)
            left = int(c * cw + offset_x)
            top = int(r * rh)
            right = int(left + cw)
            bottom = int(top + rh)

            panel_img = full_img.crop((left, top, right, bottom))
            path = os.path.join(folder, f"{name}.png")
            panel_img.save(path)
            print(f"  > Saved {path} (Size: {panel_img.size})")

        self.plotter.render()  # refresh to show decorations again

    def _setup_camera(self) -> None:
        cam_dir = WORLD_FORWARD + WORLD_RIGHT + WORLD_UP
        cam_pos = cam_dir / np.linalg.norm(cam_dir) * (self.r_sphere * 6.5)

        for (row, col) in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]:
            self.plotter.subplot(row, col)
            self.plotter.camera_position = [list(cam_pos), (0, 0, 0), list(WORLD_UP)]

    def show(self):

        self.plotter = pv.Plotter(
            shape=(2, 4),
            groups=[(1, slice(2, 4))],
            window_size=[2400, 1200],  # Changed from 1100 to 1200 (4:2 ratio)
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
        self._apply_meshing_debug_visibility()
        self._apply_alignment_visibility()
        self._apply_saccade_visibility()
        self._apply_heatmap_visibility()
        self._apply_bigpanel_visibility()

        self.plotter.link_views()
        self._setup_keybindings()
        self._setup_camera()
        self.plotter.show()

##

if __name__ == "__main__":
    from rhabdoforge.compound_eyes.rhabdomeres import drosophila_bundle, honeybee_bundle

    # ----------------------------------------------------------------------

    droso_head_ptich = np.deg2rad(10.1)     # drosophila head pitch in flight

    aligner = BundlesAligner(
        ref_direction=np.array([0.0, np.sin(droso_head_ptich), np.cos(droso_head_ptich)]),   # optic flow in flight
        combing_strength=1.0,
        combing_angle_deg=45.0,
        combing_falloff=0.7,
        alignment_smoothing_iter=5,
        saccade_smoothing_iter=5,
        flip_polarity=False,
        flip_saccade_polarity=True,
        equatorial_discontinuity=True,  # important for drosophila rhabdomere bundles alignment
    )

    # model = Model.from_sphere(
    #     n=1600,
    #     eye_radius=200.0,    # 200 µm
    #     bundle=drosophila_bundle(),
    #     orientation=aligner,
    #     neural_superposition=True
    # )

    model = Model.from_file(
        'assets/drosophila_scaffold.npz',
        bundle=drosophila_bundle(),
        orientation=aligner,
        neural_superposition=True,    # superposition eyes
    )
    # model.refine_superposition(smooth_iters=3, adjust_scale=True, adjust_anisotropy=True, rewire=True)

    # ----------------------------------------------------------------------

    # aligner = BundlesAligner(
    #     equatorial_discontinuity=False,  # no equatorial discontinuity in the bee
    #     ref_direction=WORLD_BACKWARD,   # optic flow in flight
    #     combing_strength=1.0,
    #     combing_angle_deg=45.0,
    #     alignment_smoothing_iter=4,
    #     saccade_smoothing_iter=5,
    # )
    #
    # model = Model.from_file(
    #     'assets/honeybee_scaffold_s10.npz',
    #     bundle=honeybee_bundle(),
    #     orientation=aligner,
    #     neural_superposition=False,     # Apposition eyes
    # )

    # ----------------------------------------------------------------------

    viewer = EyeViewer(model, aligner=aligner)
    viewer.show()

