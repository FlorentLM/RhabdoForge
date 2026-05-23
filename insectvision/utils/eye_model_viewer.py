import numpy as np
import pyvista as pv
import vtk

from insectvision.compound_eyes.receptor_array import ReceptorArray
from insectvision.compound_eyes.kernel import drosophila_kernel
from insectvision.compound_eyes.orientation import BundleOrientationField
from insectvision.compound_eyes.datatypes import get_metadata_field
from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD


RECEPTOR_PALETTE = [
    '#ff4034',  # R1
    '#44e453',  # R2
    '#4581ff',  # R3
    '#ffad13',  # R4
    '#c61ee9',  # R5
    '#42d4f4',  # R6
    '#FFD700',  # R7/8
]

LENS_DOME_COLORS = ['#E6B7B1', '#B8BEE6']   # left, right


def cartridge_map_dense(ra) -> np.ndarray:
    """
    Build the (N, R) cartridge map: result[L, r] is the lens index that
    donates its R-r receptor to cartridge L.

    ReceptorArray stores wiring as forward links (each receptor knows
    which cartridge it feeds via rcpt_static_data['cartridge_src']), this
    reconstructs the inverse view used by the cartridge-map visualisation.

    If two source receptors of the same type feed the same cartridge
    (a receiving conflict) only the last-written source survives here
    """
    N, R = ra.lens_count, ra.receptors_per_lens
    result = np.tile(np.arange(N, dtype=np.intp)[:, None], (1, R))
    if not ra._cartridges_wired:
        return result

    cartridge_src = ra.rcpt_static_data['cartridge_src']
    cart_lens = (cartridge_src // R).astype(np.intp)
    types = get_metadata_field(ra.rcpt_static_data['metadata'], 'rcpt_type').astype(np.intp)
    sources = (np.arange(N * R, dtype=np.intp) // R)

    result[cart_lens, types] = sources
    return result


def compute_cartridge_conflicts(ra) -> dict:
    """
    Detect receiving conflicts in the wired cartridges.

    A clean neural-superposition cartridge gets exactly one receptor of each
    peripheral type (plus the home ommatidium's own central R7/8). If two
    source receptors of type r both have their 'cartridge_src' pointing to
    the same cartridge L, that's a receiving conflict for L.

    # TODO: Donation conflicts!!!!!
    """
    N, R = ra.lens_count, ra.receptors_per_lens
    center = ra.kernel.center_index

    cart_lens = (ra.rcpt_static_data['cartridge_src'] // R).astype(np.intp)
    types = get_metadata_field(ra.rcpt_static_data['metadata'], 'rcpt_type').astype(np.intp)

    counts = np.zeros((N, R), dtype=np.intp)
    np.add.at(counts, (cart_lens, types), 1)

    incoming = np.zeros(N, dtype=np.intp)
    per_rhab = np.zeros(R, dtype=np.intp)
    for r in range(R):
        if r == center:
            continue
        excess = np.maximum(counts[:, r] - 1, 0)
        incoming += excess
        per_rhab[r] = int(excess.sum())

    return {
        'incoming': incoming,
        'per_rhab': per_rhab,
        'counts': counts,
        'total': int(per_rhab.sum()),
    }


def lens_radii_from_lattice(ra) -> np.ndarray:
    """Per-lens display radius = mean distance to the lens's immediate lattice neighbours."""
    radii = np.zeros(ra.lens_count, dtype=np.float32)
    for eye in ra.eyes:
        if len(eye) == 0:
            continue
        result = eye.neighbours(
            lens_indices=eye.lens_indices,
            k=6,
            immediate_only=True,
        )
        if result.distances.size == 0:
            continue
        radii[eye.lens_indices] = result.distances.mean(axis=1)
    radii[radii < 1e-9] = 0.001
    return radii


def receptor_tip_offsets(ra) -> np.ndarray:
    """
    (N, R, 3) world-space tangent offsets of each rhabdomere tip from its
    parent lens centre.

    Reconstructed from receptor directions: each direction points from the
    rhabdomere tip through the nodal point, so subtracting the axial
    component and negating gives the tip offset in the tangent plane.
    """
    N, R = ra.lens_count, ra.receptors_per_lens
    d = ra.lenses.directions
    rec_dirs = ra.rcpt_dynamic_data['direction'].reshape(N, R, 3)
    axial = np.sum(rec_dirs * d[:, None, :], axis=2, keepdims=True)
    return -(rec_dirs - axial * d[:, None, :])


_LENS_TEMPLATE = None


def _lens_template() -> pv.PolyData:
    """Squished hemisphere used as the per-lens glyph."""
    global _LENS_TEMPLATE
    if _LENS_TEMPLATE is None:
        t = pv.Sphere(radius=1.0, phi_resolution=20, theta_resolution=20, end_phi=90)
        t.scale([1.0, 1.0, 0.5], inplace=True)  # flatten the dome
        t.rotate_y(90, inplace=True)            # top points along +X
        t.translate([-0.1, 0, 0], inplace=True) # recess into head
        _LENS_TEMPLATE = t
    return _LENS_TEMPLATE


def build_lens_glyphs(ra, scalars=None) -> pv.PolyData:
    """Position and orient the lens dome at every lens, with optional scalar field."""
    p = ra.lenses.positions
    d = ra.lenses.directions
    radii = lens_radii_from_lattice(ra)

    pd = pv.PolyData(p)
    pd.point_data['radius'] = radii
    pd.point_data['vectors'] = d

    if scalars is None:
        scalars = (p @ np.asarray(WORLD_RIGHT, dtype=np.float32) < 0).astype(float)
    pd.point_data['active_scalars'] = scalars

    return pd.glyph(geom=_lens_template(), scale='radius', orient='vectors', factor=1.0)


class EyeViewer:
    """
    Interactive 3D viewer for a ReceptorArray.
    """

    def __init__(
        self,
        ra: ReceptorArray,
        optic_flow_world=None,
        window_size=(1200, 900),
    ):

        self.ra = ra
        self.kernel = ra.kernel
        self.N = ra.lens_count
        self.R = ra.receptors_per_lens

        self.right = np.asarray(WORLD_RIGHT, dtype=np.float32)
        self.up = np.asarray(WORLD_UP, dtype=np.float32)
        self.fwd = np.asarray(WORLD_FORWARD, dtype=np.float32)

        flow = optic_flow_world if optic_flow_world is not None else -self.fwd
        self.optic_flow_world = np.asarray(flow, dtype=np.float32)

        # Cache geometry
        self.p = ra.lenses.positions
        self.d = ra.lenses.directions
        self.is_left_eye = (self.p @ self.right) < 0

        self.r_sphere = float(np.mean(np.linalg.norm(self.p, axis=1)))
        self.arrow_len = self.r_sphere * 0.15

        # plot setup
        self.plotter = pv.Plotter(window_size=list(window_size))
        self.plotter.set_background('white')

        # Actor groups for visibility toggling
        self.actors = {
            'lenses':          [],
            'local_flow':      [],
            'saccades':        [],
            'flow_axis':       [],
            'main_axis':       [],
            'cartridge_field': [],
            'conflicts':       [],
            'bundles':         [],
        }
        self._clip_actors = []                # everything clipped by the slicer plane
        self._cart_neighbourhood_actors = []  # cleared between [N] presses
        self._slicer_callback = None

    # main entry point

    def show(self) -> None:
        self._add_equator_plane()
        self._add_lenses_glyphs()
        if self.R > 1 and self.ra._cartridges_wired:
            self._add_cartridge_conflicts()
        self._add_local_flow()
        self._add_saccades()
        if self.R > 1:
            self._add_main_axis()
            self._add_flow_axis()
            self._add_bundles()
            if self.ra._cartridges_wired:
                self._add_cartridge_field()
        self._add_world_references()
        self._setup_slicer()
        self._setup_keybindings()
        self._setup_camera()
        self.plotter.show()

    # Scene layers

    def _add_equator_plane(self) -> None:
        eq_rad = self.r_sphere * 1.3
        plane = pv.Plane(
            center=(0, 0, 0), direction=self.up,
            i_size=eq_rad * 2, j_size=eq_rad * 2,
        )
        self.plotter.add_mesh(plane, color='black', opacity=0.05, show_edges=True)

    def _add_lenses_glyphs(self) -> None:
        mesh = build_lens_glyphs(self.ra)
        actor = self.plotter.add_mesh(
            mesh,
            scalars='active_scalars',
            cmap=LENS_DOME_COLORS,
            show_scalar_bar=False,
            opacity=1.0,
            smooth_shading=True,
            ambient=0.3,
            show_edges=False,
            backface_culling=True,
        )
        actor.SetVisibility(False)
        self._clip_actors.append(actor)
        self.actors['lenses'].append(actor)

    def _add_cartridge_conflicts(self) -> None:
        conflicts = compute_cartridge_conflicts(self.ra)
        incoming = conflicts['incoming'].astype(float)

        mesh = build_lens_glyphs(self.ra, scalars=incoming)
        actor = self.plotter.add_mesh(
            mesh,
            scalars='active_scalars',
            cmap='RdYlGn_r',
            clim=[0, max(1, float(incoming.max()))],
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Incoming conflicts', 'position_x': 0.85},
            smooth_shading=True,
            ambient=0.3,
            show_edges=False,
            backface_culling=True,
        )
        actor.SetVisibility(False)
        self._clip_actors.append(actor)
        self.actors['conflicts'].append(actor)

    def _add_local_flow(self) -> None:
        dots = np.sum(self.optic_flow_world * self.d, axis=1)
        v_proj = self.optic_flow_world - dots[:, None] * self.d
        norms = np.linalg.norm(v_proj, axis=1)
        valid = norms > 1e-6
        v_proj[valid] /= norms[valid, None]

        if not np.any(valid):
            return

        pd = pv.PolyData(self.p[valid])
        pd['vectors'] = v_proj[valid] * self.arrow_len
        arrows = pd.glyph(orient='vectors', scale='vectors', factor=1.0)

        actor = self.plotter.add_mesh(
            arrows, color='#da70d6', opacity=0.9,
            ambient=0.5, diffuse=0.8, smooth_shading=True,
        )
        actor.SetVisibility(False)
        self._clip_actors.append(actor)
        self.actors['local_flow'].append(actor)

    def _add_saccades(self) -> None:
        s_world = self.ra.saccade_field()
        for mask, color in [(self.is_left_eye, 'red'), (~self.is_left_eye, 'blue')]:
            if not np.any(mask):
                continue
            pd = pv.PolyData(self.p[mask])
            pd['vectors'] = s_world[mask] * self.arrow_len
            arrows = pd.glyph(orient='vectors', scale='vectors', factor=1.0)
            actor = self.plotter.add_mesh(
                arrows, color=color, ambient=0.5, diffuse=0.8, smooth_shading=True,
            )
            actor.SetVisibility(False)
            self._clip_actors.append(actor)
            self.actors['saccades'].append(actor)

    def _add_main_axis(self) -> None:
        """Bundle main axis (R3 -> centre) reconstructed from receptor geometry."""
        offsets = receptor_tip_offsets(self.ra)
        center = self.kernel.center_index
        i1, _ = self.kernel.main_axis_indices
        if i1 == center:
            return

        main_world = offsets[:, i1, :] - offsets[:, center, :]
        norms = np.linalg.norm(main_world, axis=1, keepdims=True)
        np.divide(main_world, norms, out=main_world, where=norms > 1e-11)

        v_offset = main_world * self.arrow_len * 0.4
        pd = pv.PolyData(self.p - v_offset)
        pd['vectors'] = main_world * self.arrow_len * 0.8
        arrows = pd.glyph(orient='vectors', scale='vectors', factor=1.0)

        actor = self.plotter.add_mesh(
            arrows, color='#F6C735', opacity=0.9,
            ambient=0.5, diffuse=0.8, smooth_shading=True,
        )
        actor.SetVisibility(False)
        self._clip_actors.append(actor)
        self.actors['main_axis'].append(actor)

    def _add_flow_axis(self) -> None:
        """Bundle alignment / flow axis as an undirected line through each lens."""
        axis_world = self._kernel_axis_to_world(self.kernel.flow_axis_rad)
        v_half = axis_world * self.arrow_len * 0.4

        pts = np.empty((self.N * 2, 3), dtype=np.float32)
        pts[0::2] = self.p - v_half
        pts[1::2] = self.p + v_half

        lines = np.empty((self.N, 3), dtype=np.int_)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(0, self.N * 2, 2)
        lines[:, 2] = np.arange(1, self.N * 2, 2)

        pd = pv.PolyData(pts, lines=lines)
        actor = self.plotter.add_mesh(pd, color='green', line_width=3.0, opacity=0.9)
        actor.SetVisibility(False)
        self._clip_actors.append(actor)
        self.actors['flow_axis'].append(actor)

    def _add_cartridge_field(self) -> None:
        """Per-cartridge line from R3-source to R6-source, coloured by conflict status."""
        i1, i2 = self.kernel.main_axis_indices
        if i1 == i2:
            return

        cart_map = cartridge_map_dense(self.ra)
        v_cart = self.p[cart_map[:, i2]] - self.p[cart_map[:, i1]]
        dots = np.sum(v_cart * self.d, axis=1)
        v_proj = v_cart - dots[:, None] * self.d
        norms = np.linalg.norm(v_proj, axis=1, keepdims=True)
        np.divide(v_proj, norms, out=v_proj, where=norms > 1e-8)

        # Orient consistently against the saccade field
        s_world = self.ra.saccade_field()
        v_proj *= np.sign(np.sum(v_proj * s_world, axis=1))[:, None]

        # Highlight cartridges with any peripheral self-reference (= missing
        # contributor for that receptor type). Receiving conflicts have their
        # own dedicated heatmap on the [V] toggle.
        center = self.kernel.center_index
        periph_cols = np.array([r for r in range(self.R) if r != center])
        self_ref = cart_map[:, periph_cols] == np.arange(self.N)[:, None]
        is_incomplete = np.any(self_ref, axis=1)

        for mask, color in [(~is_incomplete, '#00FF00'), (is_incomplete, '#FF4500')]:
            if not np.any(mask):
                continue
            pd = pv.PolyData(self.p[mask])
            pd['vectors'] = v_proj[mask] * self.arrow_len * 0.9
            arrows = pd.glyph(orient='vectors', scale='vectors')
            actor = self.plotter.add_mesh(
                arrows, color=color, opacity=0.9,
                ambient=0.5, diffuse=0.8, smooth_shading=True,
            )
            actor.SetVisibility(False)
            self._clip_actors.append(actor)
            self.actors['cartridge_field'].append(actor)

    def _add_bundles(self) -> None:
        """Coloured rhabdomere tips."""

        offsets = receptor_tip_offsets(self.ra)
        max_offset = float(np.max(np.linalg.norm(offsets, axis=2)))
        tip_scale = (self.r_sphere * 0.01) / max(max_offset, 1e-8)
        tip_positions = self.p[:, None, :] + offsets * tip_scale

        is_mirrored = self.ra.lenses.chiralities < 0
        i1, i2 = self.kernel.main_axis_indices

        groups = {'gold': [], 'white': [], 'red': [], 'teal': [], 'royalblue': []}
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
            actor = self.plotter.add_points(
                stacked, color=color, point_size=12,
                render_points_as_spheres=True,
            )
            # actor.SetVisibility(False)
            self._clip_actors.append(actor)
            self.actors['bundles'].append(actor)

    def _add_world_references(self) -> None:
        g_vec = self.optic_flow_world / np.linalg.norm(self.optic_flow_world)
        start = self.fwd * (self.r_sphere * 1.5)
        pd = pv.PolyData(np.array([start]))
        pd['vectors'] = np.array([g_vec]) * (self.r_sphere * 0.8)
        self.plotter.add_mesh(
            pd.glyph(orient='vectors', scale='vectors'), color='#da70d6',
        )

        label_dist = self.r_sphere * 1.6
        anchors = [
            ( self.fwd * label_dist, "Front",   'gray'),
            (-self.fwd * label_dist, "Back",    'gray'),
            ( self.up  * label_dist, "Dorsal",  'gray'),
            (-self.up  * label_dist, "Ventral", 'gray'),
            ( self.right * label_dist, "Right", 'blue'),
            (-self.right * label_dist, "Left",  'red'),
        ]
        for pt, txt, col in anchors:
            self.plotter.add_point_labels(
                np.array([pt]), [txt], text_color=col, point_size=0,
                font_size=20, shape_opacity=0.0,
            )

    # Camera-tracking slicer

    def _setup_slicer(self) -> None:
        clip_plane = vtk.vtkPlane()
        clip_plane.SetNormal(0, 0, -1)
        clip_plane.SetOrigin(0, 0, 0)
        for act in self._clip_actors:
            act.GetMapper().AddClippingPlane(clip_plane)

        origin = np.mean(self.p, axis=0).tolist()

        def callback(caller=None, event=None):
            cam = np.array(self.plotter.camera.position)
            normal = cam - np.array(origin)
            normal /= np.linalg.norm(normal)
            clip_plane.SetNormal(normal.tolist())

        self._slicer_callback = callback
        self.plotter.iren.interactor.AddObserver('InteractionEvent', callback)

    # Key bindings

    def _setup_keybindings(self) -> None:
        def toggle(group_name: str):
            for a in self.actors[group_name]:
                a.SetVisibility(not a.GetVisibility())
            self._slicer_callback()
            self.plotter.render()

        bindings = {
            'l': 'lenses',
            'o': 'local_flow',
            'm': 'saccades',
            'r': 'bundles',
            'a': 'main_axis',
            'b': 'flow_axis',
            'c': 'cartridge_field',
            'v': 'conflicts',
        }
        for key, group in bindings.items():
            self.plotter.add_key_event(key, lambda g=group: toggle(g))
        self.plotter.add_key_event('n', lambda: self.show_neighbourhood())

        info_text = (
            '[ L ] Lens lattice (surface)\n\n'
            '[ R ] Rhabdomeres\n\n'
            '[ O ] Optic flow vector field\n\n'
            '[ M ] Microsaccade vector field\n\n'
            '[ A ] Bundle main axis\n\n'
            '[ B ] Bundle alignment axis\n\n'
            '[ C ] Cartridge map\n\n'
            '[ V ] Cartridge conflicts heatmap\n\n'
            '[ N ] Random cartridge inspector'
        )
        self.plotter.add_text(
            info_text, position='lower_edge', font_size=6,
            color='black', font='courier', name='toggle_info',
        )

    # internal helpers

    def _setup_camera(self) -> None:
        self.plotter.add_axes()
        cam_dir = self.fwd + self.right + self.up
        cam_pos = cam_dir / np.linalg.norm(cam_dir) * (self.r_sphere * 10)
        self.plotter.camera_position = [cam_pos.tolist(), (0, 0, 0), self.up.tolist()]
        self._slicer_callback()

    def _kernel_axis_to_world(self, axis_rad: float) -> np.ndarray:
        chi = self.ra.lenses.bundle_orientations.astype(np.float32)
        chirality = self.ra.lenses.chiralities.astype(np.float32)
        lr = self.ra.lenses.right_axes
        lu = self.ra.lenses.up_axes

        ax = np.cos(axis_rad) * chirality
        ay = np.full(self.N, np.sin(axis_rad), dtype=np.float32)
        cos_y, sin_y = np.cos(chi), np.sin(chi)
        bx = ax * cos_y - ay * sin_y
        by = ax * sin_y + ay * cos_y

        v = bx[:, None] * lr + by[:, None] * lu
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return np.divide(v, norms, out=v, where=norms > 1e-8)

    # Cartridge neighbourhood inspector

    def show_neighbourhood(self, target_idx: int = None) -> None:
        # Clear previous neighbourhood actors
        for act in self._cart_neighbourhood_actors:
            self.plotter.remove_actor(act)
        self._cart_neighbourhood_actors.clear()

        if target_idx is None:
            target_idx = int(np.random.randint(0, self.N))

        omm = self.ra.lenses[target_idx]
        eye = omm.eye
        target_pos = omm.position

        # Find spatial neighbours within this lens's eye
        k_nb = min(20, len(eye))
        result = eye.neighbours(points=target_pos[None, :], k=k_nb)
        neighbours = result.indices[0]
        dists = result.distances[0]

        # Cartridge partners for this lens
        partners = cartridge_map_dense(self.ra)[target_idx]
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
                disc = pv.Disc(
                    center=self.p[idx], normal=self.d[idx],
                    inner=0, outer=disc_rad, c_res=20,
                )
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
                self._cart_neighbourhood_actors.append(act)

        # 'R1', ..., 'R7/8' labels
        labels_pos, labels = [], []
        for r, lens_idx in enumerate(partners):
            if int(lens_idx) != target_idx:
                labels.append(f"R{r + 1}")
                labels_pos.append(self.p[int(lens_idx)])

        if labels:
            act = self.plotter.add_point_labels(
                labels_pos, labels, font_size=18, text_color='black',
                shape_color='white', shape_opacity=0.6,
            )
            self._cart_neighbourhood_actors.append(act)

        # Coloured receptor dots on the home ommatidium
        if self.R > 1:
            offsets = receptor_tip_offsets(self.ra)[target_idx]
            max_off = float(np.max(np.linalg.norm(offsets, axis=1)))
            dot_scale = disc_rad * 0.7 / max(max_off, 1e-6)
            for r in range(self.R):
                color = RECEPTOR_PALETTE[r % len(RECEPTOR_PALETTE)]
                if np.linalg.norm(offsets[r]) > 1e-10:
                    dot_pos = target_pos + offsets[r] * dot_scale
                else:
                    dot_pos = target_pos  # central receptor at the lens centre
                act = self.plotter.add_points(
                    np.array([dot_pos]), color=color, point_size=14,
                    render_points_as_spheres=True, opacity=0.95,
                    smooth_shading=True,
                )
                self._cart_neighbourhood_actors.append(act)

        print(f"\nCartridge {target_idx}  partners: {partners.tolist()}")


##

if __name__ == "__main__":

    pitch_rad = np.deg2rad(10.1)
    optic_flow = np.array([0.0, np.sin(pitch_rad), np.cos(pitch_rad)])

    kernel = drosophila_kernel()

    orientation = BundleOrientationField(
        flow_direction=optic_flow,
        diagonal_strength=1.0,
        alignment_smoothing_iterations=2,
        saccade_smoothing_iterations=10,
    )

    ra = ReceptorArray.from_file(
        'species_models/drosophila_custom.npz',
        kernel=kernel, orientation=orientation,
    )
    # ra = ReceptorArray.from_sphere(
    #     n=1600,
    #     kernel=kernel,
    #     orientation=orientation,
    # )

    EyeViewer(ra, optic_flow_world=optic_flow).show()