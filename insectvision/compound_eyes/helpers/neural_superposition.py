import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Tuple, NamedTuple, Optional, List, Set, Sequence, Hashable
import numpy as np
from scipy.optimize import linear_sum_assignment, LinearConstraint, milp, Bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from insectvision.types import UNWIRED_SRC
from insectvision.utils import norm_l2
from insectvision.geometry.circular import wrap_angle
from insectvision.geometry.fields import smooth_field_partitioned
from insectvision.geometry.linalg import rotate2d, local_to_world, project_to_tangent

if TYPE_CHECKING:
    from insectvision.compound_eyes.model import Model
    from insectvision.compound_eyes.views import OmmatidiumView, NeighbourResult

logger = logging.getLogger(__name__)


class WiringCandidate(NamedTuple):
    cost: float
    slots: np.ndarray
    donors: np.ndarray      # global ommatidia indices
    dists: np.ndarray
    w: complex = 1.0 + 0j   # similarity transform that produced this snap (for debug trace only)
    is_plastic: bool = False


def get_conflict_masks(cartridge_map: np.ndarray, peripheral_indices: np.ndarray) -> 'SimpleNamespace':
    """
    Per-ommatidium conflict / completeness masks.

        - receiving: sources one of its own peripheral slots from itself
        - donation: a donor feeding more than one slot (over-subscribed)
        - any: has any receiving | donation
        - unwired_slots: (N, |periph|) bool, peripheral slots with no donor
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
    )


def get_noconflict_masks(N: int, R: int) -> 'SimpleNamespace':
    """Masks for unwired model (no superposition)."""

    return SimpleNamespace(
        receiving=np.zeros(N, dtype=bool),
        donation=np.zeros(N, dtype=bool),
        any=np.zeros(N, dtype=bool),
        unwired_slots=np.zeros((N, R), dtype=bool),
    )


def _too_similar(
        new_key: Tuple[Hashable, ...] | Set,
        existing_keys: Tuple[Hashable, ...] | Set,
        threshold: float = 0.8
    ) -> bool:
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

    k_scale = np.mean(np.linalg.norm(model.bundle.offsets_um[periph] - model.bundle.offsets_um[center], axis=1)) or 1.0

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
            'assign_radius', 'scale_dev', 'min_snap_matches', 'k_search',
            'top_k', 'identity_bias', 'unassigned_penalty', 'allow_plasticity',
            'plasticity_radius', 'time_limit_s', 'collect_trace', 'min_whiten_ring',
            'max_anisotropy'
        )}
    )


def _enumerate_candidates(
        zone: 'OmmatidiumView',
        neighb: 'NeighbourResult',
        solver_context: 'SimpleNamespace'
    ) -> Tuple[List[List['WiringCandidate']], List[dict]]:
    """
    Wiring stage 1, scoped to one zone.
    """

    o_count = len(zone)
    periph = solver_context.periph
    candidates = [[] for _ in range(o_count)]
    trace = getattr(solver_context, 'collect_trace', False)
    omm_geo = []

    for i_loc in range(o_count):
        i_glob = int(zone.indices[i_loc])
        tpl_i = solver_context.tpl_dx[i_glob] + 1j * solver_context.tpl_dy[i_glob]

        neighb_dirs = solver_context.forward[neighb.indices[i_loc]] - solver_context.forward[i_glob]
        neighb_uv = neighb_dirs @ np.stack([solver_context.right[i_glob], solver_context.up[i_glob]], axis=1)

        # Whitening: normalise the neighbour cloud by the local first-ring metric
        imm = neighb.immediate[i_loc] & (neighb.indices[i_loc] != i_glob)
        ring = neighb_uv[imm]

        n_ring = ring.shape[0]
        edge_flags = zone.is_edge

        # Anisotropic whitening is only trustworthy when the ring surrounds the point
        W = np.eye(2)
        use_aniso = n_ring >= solver_context.min_whiten_ring and not edge_flags[i_loc]
        if use_aniso:
            C = (ring.T @ ring) / n_ring
            evals, evecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-12, None)
            use_aniso = (evals.max() / evals.min()) <= solver_context.max_anisotropy ** 2

        if use_aniso:
            evals = np.maximum(evals, evals.max() / solver_context.max_anisotropy ** 2)
            W = evecs @ np.diag(evals ** -0.5) @ evecs.T
            neighb_w = neighb_uv @ W.T
            scale = np.nanmedian(np.linalg.norm(neighb_w[imm], axis=1))
        else:
            neighb_w = neighb_uv
            scale = np.nanmedian(np.linalg.norm(ring, axis=1)) if ring.size else 1.0

        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0

        neighb_i = (neighb_w[:, 0] + 1j * neighb_w[:, 1]) / scale

        if trace:
            omm_geo.append({
                'i_glob': i_glob,
                'neighb_indices': np.asarray(neighb.indices[i_loc]).copy(),
                'raw_neighb_uv': neighb_uv.copy(),
                'whitened_neighb_uv': np.column_stack([neighb_i.real, neighb_i.imag]),
                'immediate': np.asarray(neighb.immediate[i_loc]).copy(),
                'same_chir': np.asarray(neighb.same_chirality[i_loc]).copy(),
                'full_template_uv': np.column_stack([tpl_i.real, tpl_i.imag]),
                'periph_template_uv': np.column_stack([tpl_i[periph].real, tpl_i[periph].imag]),
                'scale': float(scale),
                'W': W.copy()
            })

        valid_neighb = (neighb.indices[i_loc] != i_glob) & neighb.same_chirality[i_loc]

        # Guard for if a zone has fewer finite (same-chirality, non-self) columns than peripheral slots
        if int(valid_neighb.sum()) < len(periph):
            candidates[i_loc] = []
            continue

        # Dedup candidates that are essentially the same

        ws_dedup = {(1.0, 0.0)}     # set can't contain complex numbers so stored as tuple of floats

        if (mag_ok := np.abs(tpl_i) > 1e-8).any():

            neighb_valid_idx = np.flatnonzero(valid_neighb)

            for i_a in np.flatnonzero(mag_ok):
                ws = neighb_i[neighb_valid_idx] / tpl_i[i_a]
                ok = (np.abs(np.angle(ws)) <= solver_context.ang_gate) & (np.abs(np.abs(ws) - 1.0) <= solver_context.scale_dev)

                for w in ws[ok]:
                    # Rounding for dedup
                    ws_dedup.add((float(np.round(w.real, 3)), float(np.round(w.imag, 2))))

        omm_cands, seen = [], set()

        for wr, wi in ws_dedup:
            w = wr + 1j * wi    # Rebuild the complex number from the tuple key

            d = np.where(valid_neighb[None, :], np.abs((w * tpl_i[periph])[:, None] - neighb_i[None, :]), np.inf)
            r_idx, c_idx = linear_sum_assignment(d)
            md = d[r_idx, c_idx]
            keep = (md < solver_context.assign_radius) & np.isfinite(md)

            if int(keep.sum()) >= solver_context.min_snap_matches:
                donors_g = neighb.indices[i_loc][c_idx[keep]].astype(int).tolist()
                slots_g = periph[r_idx[keep]].astype(int).tolist()

                topo = tuple(sorted(zip(slots_g, donors_g)))

                if topo not in seen and not _too_similar(topo, seen):
                    seen.add(topo)
                    omm_cands.append(
                        WiringCandidate(
                            solver_context.identity_bias * float(np.abs(w - 1.0)) ** 2,
                            np.array(slots_g),
                            np.array(donors_g),
                            md[keep],
                            w
                        )
                    )

        # Plastic matching fallback
        if not omm_cands and solver_context.allow_plasticity:

            d = np.where(valid_neighb[None, :], np.abs(tpl_i[periph][:, None] - neighb_i[None, :]), np.inf)
            r_idx, c_idx = linear_sum_assignment(d)
            md = d[r_idx, c_idx]
            keep = (md < solver_context.plasticity_radius) & np.isfinite(md)

            if keep.any():
                omm_cands.append(
                    WiringCandidate(
                        solver_context.unassigned_penalty * 0.5,
                        periph[r_idx[keep]],
                        neighb.indices[i_loc][c_idx[keep]],
                        md[keep],
                        1.0 + 0j,
                        is_plastic=True
                    )
                )

        omm_cands.sort(key=lambda t: t.cost)
        candidates[i_loc] = omm_cands[:solver_context.top_k]

    return candidates, omm_geo


def _coupling_components(
        o_count: int,
        candidates: Sequence[Sequence['WiringCandidate']]
    ) -> List[np.ndarray]:
    """
    Connected components of the ommatidia graph.
    Two ommatidia are coupled iff they share a candidate (donor, slot) pair.
    MILP is separable across these, so each is solved independently.
    """

    jr_to_ommatidia = {}

    for i_loc, cands in enumerate(candidates):
        for c in cands:
            for r, j in zip(np.asarray(c.slots, dtype=int).tolist(),
                            np.asarray(c.donors, dtype=int).tolist()):
                jr_to_ommatidia.setdefault((j, r), []).append(i_loc)

    rows, cols = [], []
    for ommatidia in jr_to_ommatidia.values():
        o0 = ommatidia[0]
        for o in ommatidia[1:]:
            rows.append(o0)
            cols.append(o)

    if rows:
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(o_count, o_count))
    else:
        A = coo_matrix((o_count, o_count))

    _, labels = connected_components(A, directed=False)

    return [np.flatnonzero(labels == c) for c in range(labels.max() + 1)] if o_count else []


def _solve_omm_subset(
        omm_in_subset: np.ndarray,
        candidates: Sequence[Sequence['WiringCandidate']],
        zone: 'OmmatidiumView',
        solver_context: 'SimpleNamespace'
    ) -> Tuple[Optional[list], Optional[dict]]:
    """
    Build + solve the MILP for one component.
    Returns assignment triples, or None if solver failed.
    """

    periph = solver_context.periph
    m = len(omm_in_subset)

    costs, z_vars, x_vars, d_vars = [], [], [], []
    var_count = 0

    for i_loc in omm_in_subset:
        cands = candidates[i_loc] or [WiringCandidate(0.0, np.array([]), np.array([]), np.array([]))]
        i_z, i_x = [], []

        for cand in cands:
            i_z.append(var_count)
            costs.append(cand.cost)
            var_count += 1
            c_x = {}

            for r, j, dist in zip(cand.slots, cand.donors, cand.dists):
                c_x[int(r)] = (var_count, int(j))
                costs.append(dist)
                var_count += 1
            i_x.append(c_x)

        z_vars.append(i_z)
        x_vars.append(i_x)

        i_d = {}
        for r in periph:
            i_d[int(r)] = var_count
            costs.append(solver_context.unassigned_penalty)
            var_count += 1

        d_vars.append(i_d)

    if var_count == 0:
        return [], {}

    rows_eq, cols_eq, data_eq, b_eq = [], [], [], []
    rows_ub, cols_ub, data_ub, b_ub = [], [], [], []
    eq_row = ub_row = 0

    for t in range(m):                    # Constraint 1: one candidate per ommatidium
        for z_idx in z_vars[t]:
            rows_eq.append(eq_row)
            cols_eq.append(z_idx)
            data_eq.append(1.0)
        b_eq.append(1.0)
        eq_row += 1

    for t in range(m):                   # Constraint 2: each slot wired or dropped
        for r in periph:
            r = int(r)
            for c_x in x_vars[t]:
                if r in c_x:
                    rows_eq.append(eq_row)
                    cols_eq.append(c_x[r][0])
                    data_eq.append(1.0)

            rows_eq.append(eq_row)
            cols_eq.append(d_vars[t][r])
            data_eq.append(1.0)
            b_eq.append(1.0)
            eq_row += 1

    for t in range(m):                    # Constraint 3: x <= z
        for c_idx, c_x in enumerate(x_vars[t]):
            z_idx = z_vars[t][c_idx]

            for r, (x_idx, _) in c_x.items():
                rows_ub.append(ub_row)
                cols_ub.append(x_idx)
                data_ub.append(1.0)
                rows_ub.append(ub_row)
                cols_ub.append(z_idx)
                data_ub.append(-1.0)
                b_ub.append(0.0)
                ub_row += 1

    jr_to_x = {}                           # Constraint 4: each (donor, slot) used only once
    for t in range(m):
        for c_x in x_vars[t]:
            for r, (x_idx, j) in c_x.items():
                jr_to_x.setdefault((j, r), []).append(x_idx)

    for x_list in jr_to_x.values():

        for x_idx in x_list:
            rows_ub.append(ub_row)
            cols_ub.append(x_idx)
            data_ub.append(1.0)

        b_ub.append(1.0)
        ub_row += 1

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
        options={'time_limit': solver_context.time_limit_s, 'mip_rel_gap': 0.01, 'disp': False}
    )

    if not result.success:
        return None, None

    chosen = set(np.flatnonzero(result.x > 0.5))
    chosen_cand = {}
    if getattr(solver_context, 'collect_trace', False):
        for t, i_loc in enumerate(omm_in_subset):
            chosen_cand[i_loc] = next((c for c, z in enumerate(z_vars[t]) if z in chosen), None)

    out = []
    for t, i_loc in enumerate(omm_in_subset):
        i_glob = int(zone.indices[i_loc])
        for c_x in x_vars[t]:
            for r, (x_idx, j) in c_x.items():
                if x_idx in chosen:
                    out.append((i_glob, r, j))
    return out, chosen_cand


def solve_zone(zone: 'OmmatidiumView', solver_context: 'SimpleNamespace') -> Tuple[list, int, Optional[dict]]:
    """
    Wire one chirality zone.
    """

    trace = getattr(solver_context, 'collect_trace', False)
    o_count = len(zone)
    if o_count < 2:
        return [], 0, ({} if trace else None)

    if trace:
        # Query whole eye just so the trace can store different-chirality neighbours
        eye = zone.model.eyes[zone.eye_index[0]]
        neighb = eye.neighbours(query=zone.indices, k=min(solver_context.k_search, len(eye) - 1))
    else:
        neighb = zone.neighbours(query=zone.indices, k=min(solver_context.k_search, o_count - 1))

    if not neighb:
        return [], 0, ({} if trace else None)

    candidates, omm_geo = _enumerate_candidates(zone=zone, neighb=neighb, solver_context=solver_context)
    components = _coupling_components(o_count=o_count, candidates=candidates)

    triples, n_failed, chosen_all = [], 0, {}
    for cid, comp in enumerate(components):
        res, chosen = _solve_omm_subset(omm_in_subset=comp, candidates=candidates, zone=zone, solver_context=solver_context)

        if res is None:
            n_failed += 1
        else:
            triples.extend(res)
            if trace:
                for i_loc in comp:
                    chosen_all[i_loc] = (cid, chosen.get(i_loc))

    if not trace:
        return triples, n_failed, None

    wired = {}
    for i_glob, r, j in triples:
        wired.setdefault(i_glob, {})[int(r)] = int(j)

    zone_trace = {}
    for i_loc in range(o_count):
        geo = omm_geo[i_loc]

        i_glob = geo['i_glob']
        cid, c_idx = chosen_all.get(i_loc, (None, None))
        assignment = {int(r): None for r in solver_context.periph}
        assignment.update(wired.get(i_glob, {}))

        zone_trace[i_glob] = {
            **geo,
            'candidates': [{'w': c.w, 'slots': c.slots, 'donors': c.donors,
                            'dists': c.dists, 'cost': c.cost,
                            'is_plastic': c.is_plastic} for c in candidates[i_loc]],
            'chosen': c_idx,
            'assignment': assignment,
            'component': cid,
        }
    return triples, n_failed, zone_trace


def wire_neural_superposition(
        model: 'Model',
        assign_radius: float = 0.5,
        angular_dev: float = 40.0,
        scale_dev: float = 0.75,
        min_snap_matches: int = 2,
        k_search: int = 20,
        top_k: int = 10,
        identity_bias: float = 0.05,
        unassigned_penalty: float = 10.0,
        allow_plasticity: bool = True,
        plasticity_radius: float = 2.0,
        time_limit_s: float = 60.0,
        max_anisotropy: float = 3.0,
        min_whiten_ring: int = 5,
        n_jobs: int = -1,
        apply: bool = True,
        collect_trace: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Neural-superposition wiring allowing for local distortions and missing connections.

    Solves a per-zone MILP that lets individual rhabdomere slots be left unwired
    (with a penalty) and uses 'plastic' connections in highly distorted lattice
    regions where a rigid transform would fail.

    Args:
        - assign_radius: Max distance (μm) for standard rhabdomere-to-ommaditium assignment.
        - angular_dev: Max angular deviation (deg) allowed for bundle rotation.
        - scale_dev: Max scaling allowed (fraction) relative to the bundle size.
        - min_snap_matches: Minimum number of rhabdomeres that must align for a valid candidate.
        - k_search: Number of nearest neighbours to search for candidates.
        - top_k: Number of best-scoring candidates per ommatidium to pass to the MILP solver.
        - identity_bias: Preference for the "default" orientation (0 rotation, 1 scale).
        - unassigned_penalty: Cost incurred for every receptor slot left unwired.
        - allow_plasticity: If True, allows 'best-effort' wiring in distorted regions.
        - plasticity_radius: Increased search radius (μm) used when in plastic mode.
        - time_limit_s: Hard timeout for the MILP solver.
        - n_jobs: Number of jobs to run in parallel, -1 for auto.
        - collect_trace: bool, whether to save the wiring algorithm steps (for diagnostics).
    """

    N, R = model.shape
    cartridge_map = np.full((N, R), -1, dtype=np.intp)

    if R == 1:
        return cartridge_map, np.ones((N, 1), bool)

    center = model.bundle.center_index
    cartridge_map[:, center] = model.omm_indices

    solver_context = _make_solver_context(
        model=model, angular_dev=angular_dev, assign_radius=assign_radius, scale_dev=scale_dev,
        min_snap_matches=min_snap_matches, k_search=k_search, top_k=top_k, identity_bias=identity_bias,
        unassigned_penalty=unassigned_penalty, allow_plasticity=allow_plasticity, min_whiten_ring=min_whiten_ring,
        plasticity_radius=plasticity_radius, time_limit_s=time_limit_s, max_anisotropy=max_anisotropy,
        collect_trace=collect_trace
    )

    zones = [(eye.eye_index, sign, zv)
             for eye in model.eyes
             for sign, zv in eye.ommatidia_by_chirality().items()]

    if n_jobs == 1:
        results = [solve_zone(zone=zv, solver_context=solver_context) for _, _, zv in zones]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(solve_zone)(zv, solver_context) for _, _, zv in zones)

    full_trace = {} if collect_trace else None
    for (eye_idx, sign, _), (triples, n_failed, ztrace) in zip(zones, results):
        if n_failed:
            logger.warning(f"Eye {eye_idx} (chirality {sign:+d}): {n_failed} MILP "
                           f"component(s) failed, affected slots left unwired.")

        for i_glob, r, j in triples:
            cartridge_map[i_glob, r] = j

        if collect_trace and ztrace:
            for i_glob, rec in ztrace.items():
                rec['zone'] = (int(eye_idx), int(sign))
                full_trace[i_glob] = rec

    if collect_trace:
        model._wiring_trace = full_trace

    cs_signed = cartridge_map * R + np.arange(R)
    unwired_mask = (cartridge_map < 0).astype(bool)
    cartridge_src = np.where(unwired_mask, UNWIRED_SRC, cs_signed).astype(np.uint32)

    if apply:
        # Unwired peripheral slots fallback to self
        own_src = (np.arange(model._N)[:, None] * model._R + np.arange(model._R)).astype(np.uint32)
        cartridge_src = np.where(unwired_mask, own_src, cartridge_src).astype(np.uint32)

        model._buf['cartridge_src'] = cartridge_src
        model._buf['is_wired'] = ~unwired_mask

        model._superposition_wired = True
        model._conflicts_cache = get_conflict_masks(model.cartridge_map, model._bundle.peripheral_indices)

    return cartridge_src, unwired_mask


def refine_chi(
        model: 'Model',
        min_donors: int = 3,
        smooth_iters: int = 3,
        relax: float = 0.5,
        max_nudge_deg: float = 15.0,
        adjust_scale: bool = True,
        adjust_anisotropy: bool = False
):
    """
    Nudges individual ommatidium bundle yaw (chi) and radial scale (or 2x2 stretch matrix)
    to better align the theoretical receptor lines of sight with the targeted lenses
    selected by the neural superposition wiring solver.
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

    # Project target cartridge directions into donor's focal plane
    cmap_periph = cmap[:, periph]
    valid_mask = cmap_periph >= 0
    i_idx, r_idx = np.nonzero(valid_mask)
    j_idx = cmap_periph[valid_mask]
    p_idx = periph[r_idx]

    target_dirs = model.directions[i_idx]

    # Required offsets for rhabdomere in donor j to look at target i
    u_req = -np.einsum('ij,ij->i', target_dirs, model.right[j_idx]) * focal[j_idx]
    v_req = -np.einsum('ij,ij->i', target_dirs, model.up[j_idx]) * focal[j_idx]

    act_ang = np.arctan2(v_req, u_req)
    act_rad = np.hypot(u_req, v_req)

    # Theoretical template positions at current chi
    rot_dx, rot_dy = bundle.rotated_offsets(chi, chirality)
    tdx = rot_dx - rot_dx[:, center:center + 1]
    tdy = rot_dy - rot_dy[:, center:center + 1]

    tpl_ang = np.arctan2(tdy, tdx)[j_idx, p_idx]
    tpl_rad = np.hypot(tdx, tdy)[j_idx, p_idx]

    # Per-ommatidium rotation error
    weights = tpl_rad
    ang_err = wrap_angle(act_ang - tpl_ang)

    # Accumulate into donor (j_idx)
    sum_sin = np.bincount(j_idx, weights=np.sin(ang_err) * weights, minlength=N)
    sum_cos = np.bincount(j_idx, weights=np.cos(ang_err) * weights, minlength=N)
    omm_err = np.arctan2(sum_sin, sum_cos)

    measurable = (np.bincount(j_idx, minlength=N) >= min_donors).astype(bool)
    omm_err[~measurable] = 0.0

    groups = [eye.indices for eye in model.eyes]
    neighbours = [eye._get_first_ring_graph()['adjacency'] for eye in model.eyes]

    smoothed_err = smooth_field_partitioned(
        values=omm_err, neighbours=neighbours, kind='scalar', groups=groups, mask=measurable, n_iter=smooth_iters
    )

    # Clamp the nudge to prevent flipping
    lim = np.radians(max_nudge_deg)
    smoothed_err = np.clip(smoothed_err * relax, -lim, lim)

    # Apply rotation nudge
    model.chi = wrap_angle(chi + smoothed_err).astype(np.float32)

    if adjust_anisotropy:
        # fit a continuous 2x2 matrix T for each donor j
        # Recompute base templates with the new chi
        new_dx, new_dy = bundle.rotated_offsets(model.chi, chirality)
        base_dx = new_dx - new_dx[:, center:center + 1]
        base_dy = new_dy - new_dy[:, center:center + 1]

        p_x = base_dx[j_idx, p_idx]
        p_y = base_dy[j_idx, p_idx]

        Sxx = np.bincount(j_idx, weights=p_x * p_x, minlength=N)
        Syy = np.bincount(j_idx, weights=p_y * p_y, minlength=N)
        Sxy = np.bincount(j_idx, weights=p_x * p_y, minlength=N)

        Sxu = np.bincount(j_idx, weights=p_x * u_req, minlength=N)
        Syu = np.bincount(j_idx, weights=p_y * u_req, minlength=N)
        Sxv = np.bincount(j_idx, weights=p_x * v_req, minlength=N)
        Syv = np.bincount(j_idx, weights=p_y * v_req, minlength=N)

        det = Sxx * Syy - Sxy * Sxy
        valid_det = (det > 1e-12).astype(bool)
        safe_det = np.where(valid_det, det, 1.0)

        T_11 = np.where(valid_det, (Syy * Sxu - Sxy * Syu) / safe_det, 1.0)
        T_12 = np.where(valid_det, (Sxx * Syu - Sxy * Sxu) / safe_det, 0.0)
        T_21 = np.where(valid_det, (Syy * Sxv - Sxy * Syv) / safe_det, 0.0)
        T_22 = np.where(valid_det, (Sxx * Syv - Sxy * Sxv) / safe_det, 1.0)

        # Smooth the 2x2 matrix components
        if smooth_iters > 0:
            for T_arr in (T_11, T_12, T_21, T_22):
                T_arr[:] = smooth_field_partitioned(
                    T_arr, neighbours, kind='scalar', groups=groups, mask=valid_det, n_iter=smooth_iters
                )

        T_11 = 1.0 + (T_11 - 1.0) * relax
        T_12 = T_12 * relax
        T_21 = T_21 * relax
        T_22 = 1.0 + (T_22 - 1.0) * relax

        # Prevent extreme distortions (folding / flattening) (+/- 30%)
        T_11 = np.clip(T_11, 0.7, 1.3)
        T_22 = np.clip(T_22, 0.7, 1.3)
        T_12 = np.clip(T_12, -0.3, 0.3)
        T_21 = np.clip(T_21, -0.3, 0.3)

        # Apply the stretch
        final_dx = T_11[:, None] * base_dx + T_12[:, None] * base_dy
        final_dy = T_21[:, None] * base_dx + T_22[:, None] * base_dy

        # Center should not have drifted
        final_dx += new_dx[:, center:center + 1]
        final_dy += new_dy[:, center:center + 1]

    elif adjust_scale:
        scale_ratios = act_rad / np.clip(tpl_rad, 1e-9, None)
        sum_scale = np.bincount(j_idx, weights=scale_ratios * weights, minlength=N)
        sum_w = np.bincount(j_idx, weights=weights, minlength=N)
        sum_w = np.where(sum_w == 0, np.nanmedian(sum_w), sum_w)
        omm_scale = np.where(sum_w > 0, sum_scale / sum_w, 1.0)

        omm_scale[~measurable] = 1.0

        smoothed_scale = smooth_field_partitioned(
            values=omm_scale,
            neighbours=neighbours,
            kind='scalar',
            groups=groups,
            mask=measurable,
            n_iter=smooth_iters,
        )

        final_scale = 1.0 + (smoothed_scale - 1.0) * relax
        final_scale = np.clip(final_scale, 0.7, 1.3)  # safety clamp (+/- 30%)

        final_dx, final_dy = bundle.rotated_offsets(model.chi, chirality, scale=final_scale)
    else:
        final_dx, final_dy = bundle.rotated_offsets(model.chi, chirality)

    model.rest_offsets = np.stack([final_dx.ravel(), final_dy.ravel()], axis=-1).astype(np.float32)

    # Update microsaccade vectors to stay aligned with the new chi
    sacc_vecs = model.buffer['saccade_dxdy']
    new_sacc = rotate2d(sacc_vecs, smoothed_err)
    model.buffer['saccade_dxdy'] = new_sacc.astype(np.float32)

    tip_local = np.stack([final_dx, final_dy, np.broadcast_to(-focal[:, None], (N, R))], axis=-1)
    tip_world = local_to_world(tip_local, model.right, model.up, model.directions)
    view_dirs = norm_l2(-tip_world)

    model.buffer['curr_direction'] = view_dirs.astype(np.float32)
    model._conflicts_cache = None