import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, LinearConstraint, milp, Bounds
from scipy.sparse import coo_matrix

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



def wire_neural_superposition(
    model: 'Model',
    assign_radius: float = 0.5,
    angular_dev: float = 40.0,
    scale_dev: float = 0.75,
    min_snap_matches: int = 2,
    neighbour_dist_factor: float = 1.25,
    k_search: int = 20,
    top_k: int = 6,
    identity_bias: float = 0.05,
    unassigned_penalty: float = 10.0,
    allow_plasticity: bool = True,
    plasticity_radius: float = 2.0,
    time_limit_s: float = 60.0,
    assume_aligned: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Neural-superposition wiring allowing for local distortions and missing connections.

    Solves a per-eye MILP that lets individual rhabdomere slots be left unwired
    (with a penalty) and uses 'plastic' connections in highly distorted lattice
    regions where a rigid transform would fail.

    Args:
        - assign_radius: Max distance (μm) for standard rhabdomere-to-lens assignment.
        - angular_dev: Max angular deviation (deg) allowed for bundle rotation.
        - scale_dev: Max scaling allowed (fraction) relative to the bundle size.
        - min_snap_matches: Minimum number of rhabdomeres that must align for a valid candidate.
        - neighbour_dist_factor: Multiplier to define "immediate" lattice neighbours.
        - k_search: Number of nearest neighbours to search for candidates.
        - top_k: Number of best-scoring candidates per lens to pass to the MILP solver.
        - identity_bias: Preference for the "default" orientation (0 rotation, 1 scale).
        - unassigned_penalty: Cost incurred for every receptor slot left unwired.
        - allow_plasticity: If True, allows 'best-effort' wiring in distorted regions.
        - plasticity_radius: Increased search radius (μm) used when in plastic mode.
        - time_limit_s: Hard timeout for the MILP solver.
    """

    N, R = model.shape

    if R == 1:
        return np.full((N, 1), -1, dtype=np.intp)

    center = model.bundle.center_index
    periph = model.bundle.peripheral_indices

    cartridge_map = np.full((N, R), -1, dtype=np.intp)
    cartridge_map[:, center] = model._omm_indices       # TODO: this name might change

    rot = model.buffer['rot_offset']
    rot_dx, rot_dy = rot[..., 0], rot[..., 1]

    k_scale = float(np.mean(np.linalg.norm(
        model.bundle.offsets_um[periph] - model.bundle.offsets_um[center], axis=1))) or 1.0
    tpl_dx = (rot_dx - rot_dx[:, center:center + 1]) / k_scale
    tpl_dy = (rot_dy - rot_dy[:, center:center + 1]) / k_scale
    ang_gate = np.radians(angular_dev)

    forward = model.buffer['forward']
    right = model.buffer['right']
    up = model.buffer['up']

    for eye in model.eyes:
        n_e = len(eye)
        if n_e < 2:
            continue

        neighb = eye.neighbours(
            query=eye.indices,
            k=min(k_search, n_e - 1),
            neighbour_dist_factor=neighbour_dist_factor
        )

        # Stage 1: Enumerate candidates
        candidates = [[] for _ in range(n_e)]
        for i_loc in range(n_e):
            i_glob = eye.indices[i_loc]
            tpl_i = (tpl_dx[i_glob] + 1j * tpl_dy[i_glob])

            # Setup neighbour coordinates in tangent plane (using optical axes)
            nb_dirs = forward[neighb.indices[i_loc]] - forward[i_glob]
            nb_uv = (nb_dirs @ np.stack([right[i_glob], up[i_glob]], axis=1))

            # Normalised by local angular spacing
            ang_spacing = np.nanmedian(np.where(neighb.is_immediate[i_loc], np.linalg.norm(nb_uv, axis=1), np.nan))
            if np.isnan(ang_spacing) or ang_spacing < 1e-9:
                ang_spacing = 1.0

            nb_i = (nb_uv[:, 0] + 1j * nb_uv[:, 1]) / ang_spacing
            valid_nb = (neighb.indices[i_loc] != i_glob) & neighb.same_chirality[i_loc]

            # Candidate transforms (identity only when the bundle is lattice-pinned).
            ws_set = [1.0 + 0j]
            if not assume_aligned and (mag_ok := np.abs(tpl_i) > 1e-8).any():
                nb_valid_idx = np.flatnonzero(valid_nb)
                for i_a in np.flatnonzero(mag_ok):
                    ws = nb_i[nb_valid_idx] / tpl_i[i_a]
                    ok = (np.abs(np.angle(ws)) <= ang_gate) & (np.abs(np.abs(ws) - 1.0) <= scale_dev)
                    ws_set.extend(ws[ok].tolist())

            lens_cands, seen = [], set()
            for w in ws_set:
                d = np.where(valid_nb[None, :], np.abs((w * tpl_i)[:, None] - nb_i[None, :]), np.inf)
                r_idx, c_idx = linear_sum_assignment(d)
                md = d[r_idx, c_idx]

                keep = (md < assign_radius) & np.isfinite(md)
                if int(keep.sum()) >= min_snap_matches:
                    donors_g, slots_g = neighb.indices[i_loc][c_idx[keep]], periph[r_idx[keep]]
                    key = tuple(sorted(zip(slots_g, donors_g)))
                    if key not in seen:
                        seen.add(key)
                        lens_cands.append((identity_bias * float(np.abs(w - 1.0)), slots_g, donors_g, md[keep]))

            # Fallback: if no candidates found, generate one 'plastic' candidate
            if not lens_cands and allow_plasticity:
                d = np.where(valid_nb[None, :], np.abs(tpl_i[periph][:, None] - nb_i[None, :]), np.inf)

                r_idx, c_idx = linear_sum_assignment(d)
                md = d[r_idx, c_idx]

                keep = (md < plasticity_radius) & np.isfinite(md)
                if keep.any():
                    lens_cands.append((
                        unassigned_penalty * 0.5,
                        periph[r_idx[keep]],
                        neighb.indices[i_loc][c_idx[keep]],
                        md[keep]
                    ))

            lens_cands.sort(key=lambda t: t[0])
            candidates[i_loc] = lens_cands[:top_k]

        # Stage 2: MILP selection (with graceful drops)
        costs = []
        z_vars, x_vars, d_vars = [], [], []
        var_count = 0

        for i_loc in range(n_e):
            cands = candidates[i_loc]
            if not cands:
                # Isolated lens fallback
                cands = [(0.0, np.array([]), np.array([]), np.array([]))]

            i_z, i_x = [], []
            for w_cost, slots_g, donors_g, dists in cands:
                # Template selection variable (z)
                i_z.append(var_count)
                costs.append(w_cost)
                var_count += 1

                # Individual slot connection variables (x)
                c_x = {}
                for r, j, dist in zip(slots_g, donors_g, dists):
                    c_x[r] = (var_count, j)
                    costs.append(dist)
                    var_count += 1
                i_x.append(c_x)

            z_vars.append(i_z)
            x_vars.append(i_x)

            # Drop variables (d)
            i_d = {}
            for r in periph:
                i_d[r] = var_count
                costs.append(unassigned_penalty)
                var_count += 1
            d_vars.append(i_d)

        if var_count == 0:
            continue

        rows_eq, cols_eq, data_eq, b_eq = [], [], [], []
        rows_ub, cols_ub, data_ub, b_ub = [], [], [], []
        eq_row, ub_row = 0, 0

        # Constraint 1: Pick exactly one candidate rotation per lens
        for i_loc in range(n_e):
            for z_idx in z_vars[i_loc]:
                rows_eq.append(eq_row)
                cols_eq.append(z_idx)
                data_eq.append(1.0)
            b_eq.append(1.0)
            eq_row += 1

        # Constraint 2: Every slot is either wired (via chosen x) or dropped (d)
        for i_loc in range(n_e):
            for r in periph:
                for c_x in x_vars[i_loc]:
                    if r in c_x:
                        x_idx, _ = c_x[r]
                        rows_eq.append(eq_row)
                        cols_eq.append(x_idx)
                        data_eq.append(1.0)
                rows_eq.append(eq_row)
                cols_eq.append(d_vars[i_loc][r])
                data_eq.append(1.0)
                b_eq.append(1.0)
                eq_row += 1

        # Constraint 3: Can only wire a slot if its parent template candidate is chosen
        for i_loc in range(n_e):
            for c_idx, c_x in enumerate(x_vars[i_loc]):
                z_idx = z_vars[i_loc][c_idx]
                for r, (x_idx, _) in c_x.items():
                    rows_ub.append(ub_row)
                    cols_ub.append(x_idx)
                    data_ub.append(1.0)

                    rows_ub.append(ub_row)
                    cols_ub.append(z_idx)
                    data_ub.append(-1.0)

                    b_ub.append(0.0)
                    ub_row += 1

        # Constraint 4: Global donor limit (no donation conflicts)
        jr_to_x = {}
        for i_loc in range(n_e):
            for c_x in x_vars[i_loc]:
                for r, (x_idx, j) in c_x.items():
                    key = (j, r)
                    if key not in jr_to_x:
                        jr_to_x[key] = []
                    jr_to_x[key].append(x_idx)

        for x_list in jr_to_x.values():
            for x_idx in x_list:
                rows_ub.append(ub_row)
                cols_ub.append(x_idx)
                data_ub.append(1.0)
            b_ub.append(1.0)
            ub_row += 1

        # Build and solve MILP
        constraints = []
        if len(b_eq) > 0:
            A_eq = coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(len(b_eq), var_count))
            constraints.append(LinearConstraint(A_eq, lb=b_eq, ub=b_eq))
        if len(b_ub) > 0:
            A_ub = coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(len(b_ub), var_count))
            constraints.append(LinearConstraint(A_ub, lb=-np.inf, ub=b_ub))

        result = milp(
            c=costs,
            constraints=constraints,
            bounds=Bounds(0, 1),
            integrality=np.ones(var_count, dtype=int),
            options={'time_limit': time_limit_s, 'mip_rel_gap': 0.01, 'disp': False},
        )

        if not result.success:
            logger.warning(
                f"Eye {eye.eye_index}: MILP failed ({result.message}). Falling back to unassigned."
            )
            continue

        # Map the MILP solution back to the cartridge map
        chosen = set(np.flatnonzero(result.x > 0.5))
        for i_loc in range(n_e):
            i_glob = eye.indices[i_loc]
            for c_x in x_vars[i_loc]:
                for r, (x_idx, j) in c_x.items():
                    if x_idx in chosen:
                        cartridge_map[i_glob, r] = j

    # source receptor = donor_omm * R + slot (same slot in the donor's bundle)
    cs_signed = cartridge_map * R + np.arange(R)
    unwired_mask = cartridge_map < 0
    cartridge_src = np.where(unwired_mask, UNWIRED_SRC, cs_signed).astype(np.uint32)

    return cartridge_src, unwired_mask