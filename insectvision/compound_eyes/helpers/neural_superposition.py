import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, LinearConstraint, milp, Bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from insectvision.geometry.circular import wrap_angle
from insectvision.geometry.neighbours import smooth_field_partitioned

if TYPE_CHECKING:
    from insectvision.compound_eyes.model import Model

logger = logging.getLogger(__name__)

# Sentinel stored in cartridge_src for a slot is not wired
UNWIRED_SRC = np.uint32(0xFFFFFFFF)


def get_conflict_masks(cartridge_map: np.ndarray, peripheral_indices: np.ndarray) -> SimpleNamespace:
    """
    Per-ommatidium conflict / completeness masks.

        - receiving: sources one of its own peripheral slots from itself
        - donation: a donor feeding more than one slot (over-subscribed)
        - any: has any receiving | donation
        - unwired_slots: (N, |periph|) bool, peripheral slots with no donor
        - has_selfwires: any unwired peripheral slot (name kept for the view layer, it really means "has an unwired peripheral slot") # TODO: remove that one?
        - unwired_count: total unwired peripheral slots
    """
    N = cartridge_map.shape[0]
    own = np.arange(N)
    periph = np.asarray(peripheral_indices)

    receiving = np.zeros(N, dtype=bool)
    donation = np.zeros(N, dtype=bool)
    for r in periph:
        col = cartridge_map[:, r]
        receiving |= (col == own)
        valid = col >= 0
        if valid.any():
            donation |= np.bincount(col[valid], minlength=N) > 1

    unwired_slots = cartridge_map[:, periph] < 0
    return SimpleNamespace(
        receiving=receiving,
        donation=donation,
        any=receiving | donation,
        unwired_slots=unwired_slots,
        has_selfwires=np.any(unwired_slots, axis=1),
        unwired_count=int(unwired_slots.sum()),
    )


def get_noconflict_masks(N: int, R: int) -> SimpleNamespace:
    """Masks for unwired model (no superposition)."""

    return SimpleNamespace(
        receiving=np.zeros(N, dtype=bool),
        donation=np.zeros(N, dtype=bool),
        any=np.zeros(N, dtype=bool),
        unwired_slots=np.zeros((N, R), dtype=bool),
        has_selfwires=np.zeros(N, dtype=bool),
        unwired_count=0,
    )


def _too_similar(new_key, existing_keys, threshold=0.8):

    new_set = set(new_key)

    for existing in existing_keys:
        shared = len(new_set.intersection(set(existing)))
        if shared / len(new_set) >= threshold:
            return True

    return False


def _make_solver_context(model: 'Model', **p) -> 'SimpleNamespace':
    """
    Read-only template data + tuning params shared by the solver for every zone.
    """

    N, R = model.shape
    center = model.bundle.center_index
    periph = model.bundle.peripheral_indices

    rot = model.buffer['rest_offset']
    rot_dx, rot_dy = rot[..., 0], rot[..., 1]

    k_scale = float(np.mean(np.linalg.norm(
        model.bundle.offsets_um[periph] - model.bundle.offsets_um[center], axis=1))) or 1.0

    return SimpleNamespace(
        periph=periph,
        center=center, R=R,
        tpl_dx=(rot_dx - rot_dx[:, center:center + 1]) / k_scale,
        tpl_dy=(rot_dy - rot_dy[:, center:center + 1]) / k_scale,
        forward=model.buffer['forward'],
        right=model.buffer['right'],
        up=model.buffer['up'],
        ang_gate=np.radians(p['angular_dev']),
        **{k: p[k] for k in (
            'assign_radius', 'scale_dev', 'min_snap_matches',
            'k_search', 'top_k', 'identity_bias', 'unassigned_penalty',
            'allow_plasticity', 'plasticity_radius', 'time_limit_s',
            'assume_aligned'
        )}
    )


def _enumerate_candidates(zone, neighb, ctx):
    """
    Wiring stage 1, scoped to one zone. Returns List[List[candidate]] per lens.
    """

    n_e = len(zone)
    periph = ctx.periph
    candidates = [[] for _ in range(n_e)]

    for i_loc in range(n_e):
        i_glob = int(zone.indices[i_loc])
        tpl_i = ctx.tpl_dx[i_glob] + 1j * ctx.tpl_dy[i_glob]

        nb_dirs = ctx.forward[neighb.indices[i_loc]] - ctx.forward[i_glob]
        nb_uv = nb_dirs @ np.stack([ctx.right[i_glob], ctx.up[i_glob]], axis=1)

        # Normalise the neighbour cloud by the local first-ring metric
        imm = neighb.immediate[i_loc] & (neighb.indices[i_loc] != i_glob)
        ring = nb_uv[imm]
        if ring.shape[0] >= 4:
            C = (ring.T @ ring) / ring.shape[0]
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-12, None)
            # Cap per-axis stretch (3:1) so a one-sided boundary ring can't blow up
            evals = np.maximum(evals, evals.max() / 9.0)
            W = evecs @ np.diag(evals ** -0.5) @ evecs.T
            nb_w = nb_uv @ W.T
            scale = np.nanmedian(np.linalg.norm(nb_w[imm], axis=1))
        else:
            nb_w = nb_uv  # too few first-ring neighbours: isotropic
            scale = np.nanmedian(np.linalg.norm(ring, axis=1)) if ring.size else 1.0

        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        nb_i = (nb_w[:, 0] + 1j * nb_w[:, 1]) / scale

        valid_nb = (neighb.indices[i_loc] != i_glob) & neighb.same_chirality[i_loc]

        # Dedup candidates that are essentially the same

        ws_dedup = {(1.0, 0.0)}     # set can't contain complex numbers so stored as tuple of floats

        if not ctx.assume_aligned and (mag_ok := np.abs(tpl_i) > 1e-8).any():

            nb_valid_idx = np.flatnonzero(valid_nb)

            for i_a in np.flatnonzero(mag_ok):
                ws = nb_i[nb_valid_idx] / tpl_i[i_a]
                ok = (np.abs(np.angle(ws)) <= ctx.ang_gate) & (np.abs(np.abs(ws) - 1.0) <= ctx.scale_dev)

                for w in ws[ok]:
                    # Rounding for dedup
                    ws_dedup.add((float(np.round(w.real, 3)), float(np.round(w.imag, 2))))

        lens_cands, seen = [], set()

        for wr, wi in ws_dedup:
            w = wr + 1j * wi    # Rebuild the complex number from the tuple key

            d = np.where(valid_nb[None, :], np.abs((w * tpl_i[periph])[:, None] - nb_i[None, :]), np.inf)
            r_idx, c_idx = linear_sum_assignment(d)
            md = d[r_idx, c_idx]
            keep = (md < ctx.assign_radius) & np.isfinite(md)

            if int(keep.sum()) >= ctx.min_snap_matches:
                donors_g = neighb.indices[i_loc][c_idx[keep]].astype(int).tolist()
                slots_g = periph[r_idx[keep]].astype(int).tolist()

                topo = tuple(sorted(zip(slots_g, donors_g)))

                if topo not in seen and not _too_similar(topo, seen):
                    seen.add(topo)
                    lens_cands.append((ctx.identity_bias * float(np.abs(w - 1.0)),
                                       np.array(slots_g), np.array(donors_g), md[keep]))

        lens_cands.sort(key=lambda t: t[0])
        candidates[i_loc] = lens_cands[:ctx.top_k]

        if not lens_cands and ctx.allow_plasticity:

            d = np.where(valid_nb[None, :], np.abs(tpl_i[periph][:, None] - nb_i[None, :]), np.inf)
            r_idx, c_idx = linear_sum_assignment(d)
            md = d[r_idx, c_idx]
            keep = (md < ctx.plasticity_radius) & np.isfinite(md)

            if keep.any():
                lens_cands.append((ctx.unassigned_penalty * 0.5,
                                   periph[r_idx[keep]], neighb.indices[i_loc][c_idx[keep]], md[keep]))

        lens_cands.sort(key=lambda t: t[0])
        candidates[i_loc] = lens_cands[:ctx.top_k]

    return candidates


def _coupling_components(candidates, n_e):
    """
    Connected components of the lens graph: two lenses are coupled iff they share a
    candidate (donor, slot) pair. MILP is separable across these, so each is solved independently.
    """

    jr_to_lenses = {}

    for i_loc, cands in enumerate(candidates):
        for _, slots_g, donors_g, _ in cands:
            for r, j in zip(np.asarray(slots_g, dtype=int).tolist(),
                            np.asarray(donors_g, dtype=int).tolist()):
                jr_to_lenses.setdefault((j, r), []).append(i_loc)

    rows, cols = [], []
    for lenses in jr_to_lenses.values():
        l0 = lenses[0]
        for l in lenses[1:]:
            rows.append(l0)
            cols.append(l)

    if rows:
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_e, n_e))
    else:
        A = coo_matrix((n_e, n_e))

    _, labels = connected_components(A, directed=False)

    return [np.flatnonzero(labels == c) for c in range(labels.max() + 1)] if n_e else []


def _solve_lens_subset(lens_order, candidates, zone, ctx):
    """
    Build + solve the MILP for one component.
    Returns assignment triples, or None if solver failed.
    """

    periph = ctx.periph
    m = len(lens_order)

    costs, z_vars, x_vars, d_vars = [], [], [], []
    var_count = 0

    for i_loc in lens_order:
        cands = candidates[i_loc] or [(0.0, np.array([]), np.array([]), np.array([]))]

        i_z, i_x = [], []
        for w_cost, slots_g, donors_g, dists in cands:
            i_z.append(var_count); costs.append(w_cost); var_count += 1
            c_x = {}
            for r, j, dist in zip(slots_g, donors_g, dists):
                c_x[int(r)] = (var_count, int(j)); costs.append(dist); var_count += 1
            i_x.append(c_x)
        z_vars.append(i_z); x_vars.append(i_x)

        i_d = {}
        for r in periph:
            i_d[int(r)] = var_count; costs.append(ctx.unassigned_penalty); var_count += 1
        d_vars.append(i_d)

    if var_count == 0:
        return []

    rows_eq, cols_eq, data_eq, b_eq = [], [], [], []
    rows_ub, cols_ub, data_ub, b_ub = [], [], [], []
    eq_row = ub_row = 0

    for t in range(m):                    # Constraint 1: one candidate per lens
        for z_idx in z_vars[t]:
            rows_eq.append(eq_row); cols_eq.append(z_idx); data_eq.append(1.0)
        b_eq.append(1.0); eq_row += 1

    for t in range(m):                   # Constraint 2: each slot wired or dropped
        for r in periph:
            r = int(r)
            for c_x in x_vars[t]:
                if r in c_x:
                    rows_eq.append(eq_row); cols_eq.append(c_x[r][0]); data_eq.append(1.0)
            rows_eq.append(eq_row); cols_eq.append(d_vars[t][r]); data_eq.append(1.0)
            b_eq.append(1.0); eq_row += 1

    for t in range(m):                    # Constraint 3: x <= z
        for c_idx, c_x in enumerate(x_vars[t]):
            z_idx = z_vars[t][c_idx]
            for r, (x_idx, _) in c_x.items():
                rows_ub.append(ub_row); cols_ub.append(x_idx); data_ub.append(1.0)
                rows_ub.append(ub_row); cols_ub.append(z_idx); data_ub.append(-1.0)
                b_ub.append(0.0); ub_row += 1

    jr_to_x = {}                           # Constraint 4: each (donor, slot) used only once
    for t in range(m):
        for c_x in x_vars[t]:
            for r, (x_idx, j) in c_x.items():
                jr_to_x.setdefault((j, r), []).append(x_idx)
    for x_list in jr_to_x.values():
        for x_idx in x_list:
            rows_ub.append(ub_row); cols_ub.append(x_idx); data_ub.append(1.0)
        b_ub.append(1.0); ub_row += 1

    constraints = []
    if b_eq:
        A_eq = coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(len(b_eq), var_count))
        constraints.append(LinearConstraint(A_eq, lb=b_eq, ub=b_eq))
    if b_ub:
        A_ub = coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(len(b_ub), var_count))
        constraints.append(LinearConstraint(A_ub, lb=-np.inf, ub=b_ub))

    result = milp(
        c=costs,
        constraints=constraints,
        bounds=Bounds(0, 1),
        integrality=np.ones(var_count, dtype=int),
        options={'time_limit': ctx.time_limit_s, 'mip_rel_gap': 0.01, 'disp': False}
    )

    if not result.success:
        return None

    chosen = set(np.flatnonzero(result.x > 0.5))
    out = []
    for t, i_loc in enumerate(lens_order):
        i_glob = int(zone.indices[i_loc])
        for c_x in x_vars[t]:
            for r, (x_idx, j) in c_x.items():
                if x_idx in chosen:
                    out.append((i_glob, r, j))
    return out


def solve_zone(zone, ctx):
    """
    Wire one chirality zone. Returns (triples, n_failed_components).
    """

    n_e = len(zone)
    if n_e < 2:
        return [], 0

    neighb = zone.neighbours(query=zone.indices, k=min(ctx.k_search, n_e - 1))
    if not neighb:
        return [], 0

    candidates = _enumerate_candidates(zone, neighb, ctx)
    components = _coupling_components(candidates, n_e)

    triples, n_failed = [], 0
    for comp in components:
        res = _solve_lens_subset(comp, candidates, zone, ctx)
        if res is None:
            n_failed += 1
        else:
            triples.extend(res)
    return triples, n_failed


def wire_neural_superposition(
        model: 'Model',
        assign_radius: float = 0.5,
        angular_dev: float = 40.0,
        scale_dev: float = 0.75,
        min_snap_matches: int = 2,
        k_search: int = 20,
        top_k: int = 3,
        identity_bias: float = 0.05,
        unassigned_penalty: float = 10.0,
        allow_plasticity: bool = True,
        plasticity_radius: float = 2.0,
        time_limit_s: float = 60.0,
        assume_aligned: bool = False,
        n_jobs: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Neural-superposition wiring allowing for local distortions and missing connections.

    Solves a per-zone MILP that lets individual rhabdomere slots be left unwired
    (with a penalty) and uses 'plastic' connections in highly distorted lattice
    regions where a rigid transform would fail.

    Args:
        - assign_radius: Max distance (μm) for standard rhabdomere-to-lens assignment.
        - angular_dev: Max angular deviation (deg) allowed for bundle rotation.
        - scale_dev: Max scaling allowed (fraction) relative to the bundle size.
        - min_snap_matches: Minimum number of rhabdomeres that must align for a valid candidate.
        - k_search: Number of nearest neighbours to search for candidates.
        - top_k: Number of best-scoring candidates per lens to pass to the MILP solver.
        - identity_bias: Preference for the "default" orientation (0 rotation, 1 scale).
        - unassigned_penalty: Cost incurred for every receptor slot left unwired.
        - allow_plasticity: If True, allows 'best-effort' wiring in distorted regions.
        - plasticity_radius: Increased search radius (μm) used when in plastic mode.
        - time_limit_s: Hard timeout for the MILP solver.
        - n_jobs: Number of jobs to run in parallel, -1 for auto.
    """

    N, R = model.shape
    if R == 1:
        return np.full((N, 1), -1, dtype=np.intp)

    center = model.bundle.center_index
    cartridge_map = np.full((N, R), -1, dtype=np.intp)
    cartridge_map[:, center] = model.omm_indices

    solver_ctx = _make_solver_context(
        model=model, angular_dev=angular_dev, assign_radius=assign_radius, scale_dev=scale_dev,
        min_snap_matches=min_snap_matches, k_search=k_search, top_k=top_k, identity_bias=identity_bias,
        unassigned_penalty=unassigned_penalty, allow_plasticity=allow_plasticity,
        plasticity_radius=plasticity_radius, time_limit_s=time_limit_s, assume_aligned=assume_aligned
    )

    zones = [(eye.eye_index, sign, zv)
             for eye in model.eyes
             for sign, zv in eye.ommatidia_by_chirality().items()]

    if n_jobs == 1:
        results = [solve_zone(zv, solver_ctx) for _, _, zv in zones]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(solve_zone)(zv, solver_ctx) for _, _, zv in zones)

    for (eye_idx, sign, _), (triples, n_failed) in zip(zones, results):
        if n_failed:
            logger.warning(f"Eye {eye_idx} (chirality {sign:+d}): {n_failed} MILP "
                           f"component(s) failed; affected slots left unwired.")
        for i_glob, r, j in triples:
            cartridge_map[i_glob, r] = j

    cs_signed = cartridge_map * R + np.arange(R)
    unwired_mask = (cartridge_map < 0).astype(bool)
    cartridge_src = np.where(unwired_mask, UNWIRED_SRC, cs_signed).astype(np.uint32)

    return cartridge_src, unwired_mask


def refine_chi(
        model: 'Model',
        min_donors: int = 3,
        smooth_iters: int = 3,
        relax: float = 0.5,
        max_nudge_deg: float = 15.0,
        adjust_scale: bool = True
):
    """
    Nudges individual ommatidium bundle yaw (chi) and radial scale to better
    align the theoretical receptor lines of sight with the ommatidia
    selected as donors by the neural superposition wiring solver.
    """
    if not model.neural_superposition or model.shape[1] <= 1:
        return

    N, R = model.shape
    bundle = model.bundle
    center = bundle.center_index
    periph = bundle.peripheral_indices

    chi = model.chi.copy()
    chirality = model.chirality
    cmap = model.cartridge_map
    focal = model.focal_length

    # Project donor directions into home focal plane
    cmap_periph = cmap[:, periph]
    valid_mask = cmap_periph >= 0
    i_idx, r_idx = np.nonzero(valid_mask)
    j_idx = cmap_periph[valid_mask]
    p_idx = periph[r_idx]

    donor_dirs = model.directions[j_idx]

    u_act = np.einsum('ij,ij->i', donor_dirs, model.right[i_idx]) * focal[i_idx]
    v_act = np.einsum('ij,ij->i', donor_dirs, model.up[i_idx]) * focal[i_idx]

    # Actual angles and radii in the focal plane
    act_ang = np.arctan2(v_act, u_act)
    act_rad = np.hypot(u_act, v_act)

    # Theoretical template positions at current chi
    # Template UV is derived from chi and chirality
    rot_dx, rot_dy = bundle.rotated_offsets(chi, chirality)
    tdx = rot_dx - rot_dx[:, center:center + 1]
    tdy = rot_dy - rot_dy[:, center:center + 1]

    tpl_ang = np.arctan2(tdy, tdx)[i_idx, p_idx]
    tpl_rad = np.hypot(tdx, tdy)[i_idx, p_idx]

    #Per-lens rotation error and scale ratio
    weights = tpl_rad

    ang_err = wrap_angle(act_ang - tpl_ang)

    sum_sin = np.bincount(i_idx, weights=np.sin(ang_err) * weights, minlength=N)
    sum_cos = np.bincount(i_idx, weights=np.cos(ang_err) * weights, minlength=N)
    lens_err = np.arctan2(sum_sin, sum_cos)

    scale_ratios = act_rad / np.clip(tpl_rad, 1e-9, None)
    sum_scale = np.bincount(i_idx, weights=scale_ratios * weights, minlength=N)
    sum_w = np.bincount(i_idx, weights=weights, minlength=N)
    lens_scale = np.where(sum_w > 0, sum_scale / sum_w, 1.0)

    # Mask and smooth
    measurable = np.bincount(i_idx, minlength=N) >= min_donors
    lens_err[~measurable] = 0.0
    lens_scale[~measurable] = 1.0

    groups = []
    neighbours = []
    for eye in model.eyes:

        adj = eye._get_first_ring_graph()['adjacency']
        width = max((a.size for a in adj), default=0)
        nb = np.full((len(adj), max(width, 1)), -1, dtype=np.intp)

        for i, a in enumerate(adj):
            nb[i, :a.size] = a

        groups.append(eye.indices)
        neighbours.append(nb)

    smoothed_err = smooth_field_partitioned(
        values=lens_err, kind='scalar', groups=groups, neighbours=neighbours,
        n_iter=smooth_iters, mask=measurable
    )

    # Clamp the nudge to prevent flipping
    lim = np.radians(max_nudge_deg)
    smoothed_err = np.clip(smoothed_err * relax, -lim, lim)

    if adjust_scale:
        smoothed_scale = smooth_field_partitioned(
            values=lens_scale, kind='scalar', groups=groups, neighbours=neighbours,
            n_iter=smooth_iters, mask=measurable
        )
        final_scale = 1.0 + (smoothed_scale - 1.0) * relax
        # Safety clamp for scale (+-20%)
        final_scale = np.clip(final_scale, 0.8, 1.2)
    else:
        final_scale = 1.0

    # Apply
    model.chi = wrap_angle(chi + smoothed_err).astype(np.float32)

    new_dx, new_dy = bundle.rotated_offsets(model.chi, chirality, scale=final_scale)
    model.rest_offsets = np.stack([new_dx.ravel(), new_dy.ravel()], axis=-1).astype(np.float32)

    # Update microsaccade vectors to stay aligned with the new chi
    # TODO: Make this optional?
    sacc_vecs = model.buffer['saccade_dxdy']
    c, s = np.cos(smoothed_err), np.sin(smoothed_err)
    new_sacc = np.empty_like(sacc_vecs)
    new_sacc[:, 0] = c * sacc_vecs[:, 0] - s * sacc_vecs[:, 1]
    new_sacc[:, 1] = s * sacc_vecs[:, 0] + c * sacc_vecs[:, 1]
    model.buffer['saccade_dxdy'] = new_sacc.astype(np.float32)

    tip_local = np.stack([new_dx, new_dy, np.broadcast_to(-focal[:, None], (N, R))], axis=-1)
    new_dirs_world = (
            tip_local[..., 0, None] * model.right[:, None, :] +
            tip_local[..., 1, None] * model.up[:, None, :] +
            tip_local[..., 2, None] * model.directions[:, None, :]
    )

    view_dirs = -new_dirs_world
    view_dirs /= np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    model.buffer['curr_direction'] = view_dirs.astype(np.float32)
    model._conflicts_cache = None