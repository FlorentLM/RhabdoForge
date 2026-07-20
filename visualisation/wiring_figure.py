"""
Neural-superposition wiring figure.

A-C: Per-cartridge wiring diagnostics for three representative cases
     (interior / eye boundary / equator), showing the candidate
     similarity transforms, the chosen snap, and the resulting donor links.
D: Retinotopic wiring-completeness map for the example eye (with A-C marked).
"""
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from insectvision.compound_eyes import Model
from insectvision.compound_eyes.helpers.alignment import BundlesAligner
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle
from insectvision.compound_eyes.helpers.neural_superposition import wire_neural_superposition

from visualisation.plot_settings import PlotSettings, Z_TEXT



LIM = 2.6                           # Half-extent of each panel
RHAB_R = 0.16                       # Rhabdomere marker radius
OMMATIDIUM_R = 0.48                 # Ommatidium marker radius (~half a facet spacing)
SCALE_BAR_UNITS = 1.0               # Scale bar length (whitened units ~one facet spacing)
SCALE_BAR_LABEL = r'20 $\mu$m'      # Facet spacing (in the whitened frame)


CB_ALT = ('#88CCEE', '#CC6677', '#332288', '#DDCC77', '#AA4499')  # alternative candidates


# ---------------------------------------------------------------------------

def load_wired_model(
    scaffold: str = 'assets/drosophila_scaffold.npz',
    head_pitch_deg: float = 10.1,
) -> Model:
    """Build the Drosophila model and run the wiring solver with trace collection."""

    aligner = BundlesAligner(
        ref_direction=np.array([0.0, np.sin(np.deg2rad(head_pitch_deg)),
                                np.cos(np.deg2rad(head_pitch_deg))]),  # optic flow in flight
        combing_strength=1.0,
        combing_angle_deg=45.0,
        combing_falloff=0.7,
        alignment_smoothing_iter=5,
        saccade_smoothing_iter=5,
        flip_polarity=False,
        flip_saccade_polarity=True,
        equatorial_discontinuity=True,
    )

    model = Model.from_file(
        scaffold,
        bundle=drosophila_bundle(),
        orientation=aligner,
        neural_superposition=False,      # False because it's wired just below with trace
    )

    wire_neural_superposition(model, apply=True, collect_trace=True)

    return model


def _lens_diagnostics(model: Model, trace: dict) -> Dict[int, dict]:
    """Per-lens summary table, built purely from the wiring trace + model masks."""
    is_edge = np.asarray(model.is_edge)
    n_periph = np.asarray(model.bundle.peripheral_indices).size

    rows: Dict[int, dict] = {}
    for i_glob, t in trace.items():
        nb = np.asarray(t['neighb_indices'])
        imm = np.asarray(t['immediate'], dtype=bool)
        same = np.asarray(t['same_chir'], dtype=bool)
        not_self = nb != i_glob

        imm_nb = nb[imm & not_self]
        # "near edge": on the boundary, or has a first-ring neighbour that is
        near_edge = bool(is_edge[i_glob]) or (imm_nb.size > 0 and bool(is_edge[imm_nb].any()))
        # true equator: an *immediate* neighbour of opposite chirality
        equator = bool((imm & not_self & ~same).any())

        cands = t.get('candidates', []) or []
        chosen = t.get('chosen', None)
        assignment = t.get('assignment', {}) or {}
        n_wired = sum(v is not None for v in assignment.values())
        chosen_plastic = bool(cands[chosen]['is_plastic']) if (chosen is not None and chosen < len(cands)) else False

        rows[i_glob] = dict(
            edge=bool(is_edge[i_glob]),
            near_edge=near_edge,
            equator=equator,
            n_candidates=len(cands),
            n_wired=n_wired,
            n_unwired=n_periph - n_wired,
            complete=(n_wired == n_periph),
            plastic=chosen_plastic,
        )
    return rows


def select_example_cartridges(
    model: Model,
    trace: dict,
    rng: np.random.Generator,
    eye_index: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, dict]]:
    """
    Pick one interior / edge / equator example lens.
    """

    diag = _lens_diagnostics(model, trace)
    keys = np.array(sorted(diag.keys()))
    if eye_index is not None:
        eye_id = np.asarray(model.eye_index)
        keys = keys[eye_id[keys] == eye_index]

    def pool(pred) -> np.ndarray:
        return np.array([k for k in keys if pred(diag[k])])

    # Standard interior: clean lattice, away from boundary & equator, fully wired by a rigid snap
    interior = pool(lambda d: (not d['near_edge']) and (not d['equator'])
                    and d['n_candidates'] > 1 and d['complete'] and not d['plastic'])
    if interior.size == 0:
        interior = pool(lambda d: (not d['near_edge']) and (not d['equator']) and d['n_candidates'] > 1)

    if interior.size == 0:
        interior = pool(lambda d: (not d['near_edge']) and (not d['equator']))

    # Eye boundary: on the rim, ideally with some dropped slots
    edge = pool(lambda d: d['edge'] and d['n_candidates'] > 1 and d['n_unwired'] > 0)
    if edge.size == 0:
        edge = pool(lambda d: d['edge'] and d['n_candidates'] > 1)

    if edge.size == 0:
        edge = pool(lambda d: d['edge'])

    # Equator: immediate opposite-chirality neighbours but not on the rim
    equat = pool(lambda d: d['equator'] and (not d['edge']) and d['n_candidates'] > 1)
    if equat.size == 0:
        equat = pool(lambda d: d['equator'] and d['n_candidates'] > 1)

    if equat.size == 0:
        equat = pool(lambda d: d['equator'])

    picks: Dict[str, int] = {}
    used: set = set()
    for name, p in (('interior', interior), ('edge', edge), ('equator', equat)):
        p = np.array([k for k in p if k not in used])

        if p.size == 0:                                  # last-resort: anything unused
            p = np.array([k for k in keys if k not in used])

        pick = int(rng.choice(p))
        picks[name] = pick
        used.add(pick)

    return picks, diag


def _unpack_debug_data(model: Model, trace: dict, lens_id: int) -> dict:
    """Unpack the trace record."""
    t = trace[lens_id]
    main_axis_local = model.ommatidia.major_axis_field_local[lens_id]
    home_rhabs_uv = model.rest_offsets.reshape(model.N, model.R, 2)[lens_id] * 0.05
    periph_uv = t['periph_template_uv']
    return {
        'i_glob': t['i_glob'],
        'W': t['W'],
        'scale': t['scale'],
        'home_rhabs_uv': home_rhabs_uv,
        'candidates': t['candidates'],
        'chosen': t['chosen'],
        'assignment': t['assignment'],
        'chi_deg': np.degrees(model.chi[lens_id]),
        'chiral_val': t['zone'][1],
        'main_axis': complex(*main_axis_local),
        'periph_template': periph_uv[:, 0] + 1j * periph_uv[:, 1],
    }


def whitened_neighbours(model: Model, i_glob: int, W: np.ndarray, scale: float, k: int = 100):
    """Neighbour cloud around `i_glob`, in the whitened frame the solver used."""
    omm = model.ommatidia
    eye = model.eyes[int(np.asarray(omm.eye_index)[i_glob])]
    res = eye.neighbours(query=[i_glob], k=min(k, len(eye) - 1))
    ids = np.asarray(res.indices).reshape(-1).astype(int)
    same = np.asarray(res.same_chirality).reshape(-1).astype(bool)
    dvec = np.asarray(omm.directions)[ids] - np.asarray(omm.directions)[i_glob]
    uv = dvec @ np.stack([omm.right[i_glob], omm.up[i_glob]], axis=1)
    return (uv @ W.T) / scale, ids, same


# Panels A-C: per-cartridge wiring diagnostics

def _draw_empty_region(ax, pos, s: PlotSettings):

    pos = np.asarray(pos)
    if pos.shape[0] == 0:
        return

    outward = -pos.mean(axis=0)

    n = np.linalg.norm(outward)
    if n < 1e-6:
        return

    outward /= n
    tangent = np.array([-outward[1], outward[0]])
    p0 = 0.55 * outward       # boundary just past the home lens
    big = 3.0 * LIM
    corners = np.array([p0 + big * tangent, p0 - big * tangent,
                        p0 - big * tangent + big * outward, p0 + big * tangent + big * outward])

    prev = plt.rcParams.get('hatch.linewidth', 1.0)
    plt.rcParams['hatch.linewidth'] = 0.4
    patch = Polygon(corners, closed=True, facecolor='none', edgecolor=s.frame,
                    lw=0.0, hatch='////', alpha=0.20, zorder=0.5)

    ax.add_patch(patch)
    plt.rcParams['hatch.linewidth'] = prev

    ax.text(*(LIM * 0.68 * outward), 'outside eye\n(no facets)', color=s.frame,
            fontsize=s.tiny, ha='center', va='center', alpha=0.9, zorder=Z_TEXT)


def draw_wiring_panel(ax, s: PlotSettings, model: Model, trace: dict,
                      lens_id: int, title: str, show_legend: bool = False,
                      show_empty_region: bool = False):
    d = _unpack_debug_data(model, trace, lens_id)
    pos, ids, same = whitened_neighbours(model, d['i_glob'], d['W'], d['scale'])

    if show_empty_region:
        _draw_empty_region(ax, pos, s)

    wired_donor_ids = {int(v) for v in d['assignment'].values() if v is not None} if d['assignment'] else set()
    id_to_pos = {int(ids[j]): pos[j] for j in range(len(ids))}
    id_to_pos[int(lens_id)] = np.array([0.0, 0.0])

    # Neighbouring ommatidia
    for j in range(len(pos)):
        gid, uv = ids[j], pos[j]
        if int(gid) in wired_donor_ids:
            f, e, a, lw = s.yellorange, 'goldenrod', 0.20, 1.2
        else:
            f = s.grid if same[j] else s.red    # opposite-chirality neighbours flagged in red
            e, a, lw = s.frame, 0.15, 0.4
        ax.add_patch(Circle(uv, OMMATIDIUM_R, facecolor=f, edgecolor=e, alpha=a, lw=lw, zorder=3))

    # Home ommatidium + rhabdomere cluster
    ax.add_patch(Circle((0, 0), OMMATIDIUM_R, facecolor=s.yellorange,
                        edgecolor='goldenrod', alpha=0.25, lw=1.5, zorder=2))
    ax.scatter(d['home_rhabs_uv'][:, 0], d['home_rhabs_uv'][:, 1],
               c='black', s=6, zorder=Z_TEXT)

    # Peripheral rhabdomere templates for each candidate
    for c_idx, cand in enumerate(d['candidates']):
        chosen = (c_idx == d['chosen'])
        col = s.green if chosen else CB_ALT[c_idx % len(CB_ALT)]
        coords = np.column_stack([(cand['w'] * d['periph_template']).real,
                                  (cand['w'] * d['periph_template']).imag])

        for k, uv in enumerate(coords):
            r_idx = model.bundle.peripheral_indices[k]
            is_wired = (d['assignment'].get(r_idx) is not None) if (chosen and d['assignment']) else False
            if chosen:
                if is_wired:
                    ax.add_patch(Circle(uv, RHAB_R, facecolor=col, alpha=0.3, zorder=4))
                    ax.add_patch(Circle(uv, RHAB_R, facecolor='none', edgecolor=col,
                                        lw=1.2, zorder=Z_TEXT))         # crisp ring stays vector
                else:
                    ax.scatter(uv[0], uv[1], marker='x', color=col, s=28, lw=1.6, zorder=Z_TEXT + 3)
            else:
                ax.add_patch(Circle(uv, RHAB_R, facecolor=col, alpha=0.3, zorder=4))

            if chosen and is_wired:
                donor_id = d['assignment'].get(r_idx)
                if int(donor_id) in id_to_pos:
                    tgt = id_to_pos[int(donor_id)]
                    ax.plot([uv[0], tgt[0]], [uv[1], tgt[1]], color=col, lw=1.2, zorder=Z_TEXT)

    # Direction rays + inter-candidate arcs
    m0 = d['main_axis']
    ref_ang = np.degrees(np.angle(-m0))
    ax.plot([0, -m0.real * LIM * 0.8], [0, -m0.imag * LIM * 0.8],
            color='black', lw=s.curve_lw, zorder=Z_TEXT)

    for c_idx, cand in enumerate(d['candidates']):
        zc = cand['w'] * m0
        chosen = (c_idx == d['chosen'])
        col = s.green if chosen else CB_ALT[c_idx % len(CB_ALT)]

        cand_ang = np.degrees(np.angle(-zc))
        ax.plot([0, -zc.real * LIM * 0.75], [0, -zc.imag * LIM * 0.75],
                color=col, lw=1.8 if chosen else 0.9, ls='-' if chosen else (0, (3, 2)), zorder=Z_TEXT)

        diff = (cand_ang - ref_ang + 180) % 360 - 180
        arc_radius = 0.8 + c_idx * 0.35
        theta1, theta2 = min(ref_ang, ref_ang + diff), max(ref_ang, ref_ang + diff)
        ax.add_patch(Arc((0, 0), arc_radius * 2, arc_radius * 2,
                         theta1=theta1, theta2=theta2, color=col, lw=s.grid_lw + 0.4, zorder=Z_TEXT))

        mid = np.deg2rad(ref_ang + diff / 2)
        ax.text((arc_radius + 0.18) * np.cos(mid), (arc_radius + 0.18) * np.sin(mid),
                f'{abs(diff):.0f}\u00b0', color=col, fontsize=s.tiny, fontweight='bold',
                ha='center', va='center', zorder=Z_TEXT)

    # Scale bar
    ax.plot([-LIM + 0.3, -LIM + 0.3 + SCALE_BAR_UNITS], [-LIM + 0.3, -LIM + 0.3],
            color='black', lw=1.8, zorder=Z_TEXT)
    ax.text(-LIM + 0.3 + SCALE_BAR_UNITS / 2, -LIM + 0.42, SCALE_BAR_LABEL,
            ha='center', va='bottom', fontsize=s.tiny)

    title_obj = ax.set_title(title, fontsize=s.title, loc='left', pad=3)
    ax.text(0.98, 0.98, f"lens {lens_id} | $\\chi$={d['chi_deg']:.0f}\u00b0",
            transform=ax.transAxes, ha='right', va='top', fontsize=s.tiny, color=s.dark, zorder=Z_TEXT)

    ax.set_aspect('equal')
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(s.frame)
        sp.set_linewidth(s.axis_lw)

    if show_legend:
        h = [Line2D([], [], color='black', lw=1.2, label='Bundle main axis'),
             Line2D([], [], marker='o', ls='none', mfc=s.yellorange, mec='goldenrod', ms=7, label='Wired donors'),
             Line2D([], [], marker='x', ls='none', color=s.green, mew=1.6, ms=6, label='Unwired slots'),
             Line2D([], [], marker='o', ls='none', mfc=CB_ALT[0], alpha=0.4, mec='none', ms=7, label='Alt. candidate'),
             Line2D([], [], marker='o', ls='none', mfc=s.red, mec=s.frame, ms=7, label='Opposite chirality')]
        leg = ax.legend(handles=h, loc='upper left', fontsize=s.tiny,
                        handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(0.15)
        leg.get_frame().set_facecolor('white')
        leg.set_zorder(Z_TEXT)

    return title_obj


# Retinotopic projection

def _stereo_frame(dirs: np.ndarray):
    dirs = np.asarray(dirs, float)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    c = dirs.mean(0); c /= np.linalg.norm(c)
    ref = np.array([0.0, 1.0, 0.0]) if abs(c[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    r = np.cross(ref, c); r /= np.linalg.norm(r)
    u = np.cross(c, r)
    return c, r, u


def _project(dirs: np.ndarray, frame) -> np.ndarray:
    c, r, u = frame
    dirs = np.asarray(dirs, float)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    denom = 1.0 + dirs @ c
    return np.column_stack([(dirs @ r) / denom, (dirs @ u) / denom])


# Panel D: retinotopic completeness map

def panel_completeness_map(ax, s: PlotSettings, model: Model,
                           examples: Dict[str, int], eye_index: int, cax=None):
    eye = next(e for e in model.eyes if int(e.eye_index) == eye_index)
    gids = np.asarray(eye.indices)
    xy = _project(eye.directions, _stereo_frame(eye.directions))

    n_periph = np.asarray(model.bundle.peripheral_indices).size
    unwired = np.asarray(model.unwired_slots)[gids].sum(axis=1)
    is_edge = np.asarray(model.is_edge)[gids]

    cmap = LinearSegmentedColormap.from_list('completeness', [s.green, s.yellorange, s.red])
    sc = ax.scatter(xy[~is_edge, 0], xy[~is_edge, 1], c=unwired[~is_edge],
                    cmap=cmap, vmin=0, vmax=n_periph, s=14, lw=0, zorder=2)
    ax.scatter(xy[is_edge, 0], xy[is_edge, 1], facecolor='none',
               edgecolor=s.frame, s=16, lw=0.5, zorder=3)

    g2local = {int(g): k for k, g in enumerate(gids)}
    for letter, name in (('A', 'interior'), ('B', 'edge'), ('C', 'equator')):
        lid = examples[name]
        if lid in g2local:
            p = xy[g2local[lid]]
            ax.scatter(*p, marker='o', facecolor='none', edgecolor='black', s=90, lw=1.2, zorder=Z_TEXT)
            ax.annotate(letter, p, textcoords='offset points', xytext=(4, 4),
                        fontsize=s.small, fontweight='bold', zorder=Z_TEXT)

    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.05)
    ax.set_xticks([]); ax.set_yticks([])
    title_obj = ax.set_title('Wiring completeness map', fontsize=s.title, loc='left', pad=3)
    for sp in ax.spines.values():
        sp.set_edgecolor(s.frame); sp.set_linewidth(s.axis_lw)

    if cax is not None:
        cb = ax.figure.colorbar(sc, cax=cax)
    else:
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)

    cb.solids.set_rasterized(False)      # colour bars rasterise by default, keep vector
    cb.set_label('unwired slots', fontsize=s.tiny, labelpad=2)
    cb.set_ticks(range(0, n_periph + 1))
    cb.ax.tick_params(labelsize=s.tiny, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)

    return title_obj


def _place_panel_letters(fig, s: PlotSettings, items, dx_left: float = 0.010):

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    for letter, ax, title_obj in items:
        bb = title_obj.get_window_extent(renderer=r)
        top = inv.transform((bb.x0, bb.y1))[1]
        x = ax.get_position().x0 - dx_left
        fig.text(x, top, letter, ha='right', va='top', fontsize=s.initials,
                 fontweight='bold', color='black', zorder=Z_TEXT)


def make_figure(s: PlotSettings, model: Model, seed: int = 0) -> plt.Figure:
    trace = model.wiring_trace

    main_eye = max(model.eyes, key=len)
    eye_index = int(main_eye.eye_index)

    rng = np.random.default_rng(seed)
    picks, _ = select_example_cartridges(model, trace, rng, eye_index=eye_index)

    fig = s.new_figure()
    gs = GridSpec(2, 2, figure=fig, wspace=0.12, hspace=0.20)

    axA = fig.add_subplot(gs[0, 0])   # Interior
    axB = fig.add_subplot(gs[0, 1])   # Boundary
    axC = fig.add_subplot(gs[1, 0])   # Equator

    d_cell = gs[1, 1].subgridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.06)
    axD = fig.add_subplot(d_cell[0, 0])
    caxD = fig.add_subplot(d_cell[0, 1])

    tA = draw_wiring_panel(axA, s, model, trace, picks['interior'], 'Interior', show_legend=True)
    tB = draw_wiring_panel(axB, s, model, trace, picks['edge'], 'Boundary', show_empty_region=True)
    tC = draw_wiring_panel(axC, s, model, trace, picks['equator'], 'Equator (opposite-chirality)')
    tD = panel_completeness_map(axD, s, model, picks, eye_index, cax=caxD)

    fig.subplots_adjust(left=0.06, right=0.955, top=0.92, bottom=0.055)

    _place_panel_letters(fig, s, [('A', axA, tA), ('B', axB, tB), ('C', axC, tC), ('D', axD, tD)])

    return fig


if __name__ == '__main__':

    settings = PlotSettings.nature_double(aspect=1.03, rasterize=False).apply()

    model = load_wired_model()

    fig = make_figure(settings, model, seed=4)
    settings.savefig(fig, 'wiring', formats=['svg', 'eps', 'png', 'pdf'])
    plt.show()