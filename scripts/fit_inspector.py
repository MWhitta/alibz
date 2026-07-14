#!/usr/bin/env python
"""Interactive web-based LIBS fit inspector.

Walk through every stage of ``alibz.pipeline.analyze_spectrum`` and see, at
each step, what the fit looks like and how it maps onto the per-(element,
ion-stage) composition.  Then edit the fit interactively — delete a peak,
add or remove an element/ion stage — and watch the composition re-solve
(at the fixed plasma state, so it is instant), so you can attribute each
element's abundance to the peaks and stages that drive it.

Run:
    python scripts/fit_inspector.py                 # picks a sample dir
    python scripts/fit_inspector.py --data DIR      # a directory of *.csv
    python scripts/fit_inspector.py --file FILE.csv # one spectrum
then open http://127.0.0.1:8050 .

Backend compute is alibz; the browser only displays and sends edits.  A
delete/toggle re-solves with ``PeakyIndexerV3.solve_at`` at the stage's
fitted (T, nₑ) — no re-optimisation, so it is interactive and basin-free.
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from alibz.inspection import estimate_peak_uncertainties          # noqa: E402
from alibz.minor_lines import (match_and_scale,                   # noqa: E402
                               recover_residual_lines,
                               seed_minor_lines)
from alibz.peaky_finder import PeakyFinder                        # noqa: E402
from alibz.peaky_indexer_v3 import PeakyIndexerV3                 # noqa: E402
from alibz.profiles import analyze_peak_profiles, deblend_shoulders  # noqa: E402
from alibz.refinement import refine_fit                           # noqa: E402
from alibz.pipeline import (CONFIDENT_MIN_REFS, DEEPEN_BARS,      # noqa: E402
                            ESTABLISHED_MIN_FRACTION, SUPPORT_GA_FLOOR,
                            SUPPORT_TOL_NM, composition_collapsed,
                            load_spectrum_csv, sample_name, _halpha_ne)
from alibz.elements import element_color, element_sort_key        # noqa: E402
from alibz.utils.colors import wavelength_to_rgb                  # noqa: E402
from alibz.utils.database import Database                         # noqa: E402
from alibz.utils.voigt import multi_voigt, voigt_width            # noqa: E402
from alibz.utils.wavelength import (estimate_wavelength_shift,    # noqa: E402
                                    estimate_wavelength_shift_segments,
                                    shift_at)

DBPATH = os.path.join(REPO, "db")
_DB = Database(DBPATH)


# ===========================================================================
# Action logger — every interaction is appended as one JSON line, so a
# session becomes a (state, action, outcome) trace usable as training data:
# the edits encode a human's judgement about the fit (which peak is
# spurious, which element stage is a phantom) AND its measured impact on the
# composition.
# ===========================================================================
class ActionLog:
    def __init__(self, path, sample, n_points, stages):
        import datetime
        import json
        import uuid
        self._json = json
        self._dt = datetime.datetime
        self.path = path
        self.session = uuid.uuid4().hex[:12]
        self.sample = sample
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._write(dict(event="session_start", n_points=int(n_points),
                         stages=[s["name"] for s in stages]))

    def _write(self, rec):
        rec = dict(ts=self._dt.now().isoformat(timespec="milliseconds"),
                   session=self.session, sample=self.sample, **rec)
        with open(self.path, "a") as fh:
            fh.write(self._json.dumps(rec) + "\n")

    def action(self, action, **kw):
        self._write(dict(event="action", action=action, **kw))

    def state(self, **kw):
        self._write(dict(event="state", **kw))


def _comp_summary(rows, top=20):
    """Compact composition record for the log (training label)."""
    tot = sum(r["emission"] for r in rows) or 1.0
    return [dict(el=r["element"], ion=r["ion"],
                 frac=round(r["emission"] / tot, 5),
                 z=(round(r["z"], 2) if np.isfinite(r["z"]) else None))
            for r in rows[:top]]


# ===========================================================================
# Backend: stage capture + interactive re-solve
# ===========================================================================
def _ion(sp):
    return f"{sp.element} {'I' * int(sp.ion)}"


def _species_rows(idx, res, amp_sigma):
    """Per-(element, ion) rows: concentration, contribution, detection z."""
    A = np.asarray(getattr(idx, "_last_A", np.empty((0, 0))))
    if A.size == 0 or not res.concentrations.size:
        return []
    n_pk = len(idx._obs_wl)
    contrib = (A[:n_pk] * res.concentrations)
    z = (res.convergence_info or {}).get("detection_z", {})
    rows = []
    for s, sp in enumerate(res.species):
        c = float(res.concentrations[s])
        if c <= 0:
            continue
        rows.append(dict(
            key=(sp.element, int(sp.ion)), label=_ion(sp),
            element=sp.element, ion=int(sp.ion), conc=c,
            emission=float(np.sum(contrib[:, s])),
            z=float(z.get(sp.element, float("nan"))),
        ))
    rows.sort(key=lambda r: (-r["emission"]))
    return rows


def _amp_sigma(x, y, bg, peaks):
    return estimate_peak_uncertainties(x, y - bg, peaks)[:, 0]


def capture(x, y):
    """Replicate analyze_spectrum, capturing every stage.

    Returns ``(stages, meta)``: ``stages`` is a list of dicts with a
    ``name``, the ``peaks`` array at that step, and — for the stages where
    an indexer solved — ``result`` (FitResult), ``plasma`` (T, ne, sigma,
    gamma), ``species`` rows and the observed→db ``shift``.
    """
    bg0 = None
    finder = PeakyFinder.__new__(PeakyFinder)
    fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False,
                              n_sigma=0)
    bg0 = np.asarray(fit.get("background", np.zeros_like(y)), dtype=float)
    stages = [dict(name="1 · blind fit", peaks=fit["sorted_parameter_array"],
                   fit=fit)]

    shift0, _ = estimate_wavelength_shift(fit["sorted_parameter_array"], _DB)
    refined, dec_data = refine_fit(x, y, fit, db=_DB, shift_nm=shift0,
                                   asymmetric="defer")
    SHIFT, _ = estimate_wavelength_shift_segments(
        refined["sorted_parameter_array"], _DB)
    stages.append(dict(name="3a · data-only refine (asym deferred)",
                       peaks=refined["sorted_parameter_array"], fit=refined))

    rp = refined["sorted_parameter_array"]
    ne_init, ne_bounds = _halpha_ne(rp)
    run_kw = dict(sa_doublets=True, n_calls=40, verbose=False)
    idx_kw = dict(dbpath=DBPATH)
    if ne_init is not None:
        idx_kw["ne_init"] = ne_init
        run_kw["ne_bounds"] = ne_bounds

    def dbf(p):
        o = p.copy()
        o[:, 1] -= shift_at(SHIFT, o[:, 1])
        return o

    idx1 = PeakyIndexerV3(dbf(rp), **idx_kw)
    idx1._amp_sigma = _amp_sigma(x, y, bg0, rp)
    res1 = idx1.run(**run_kw)
    stages.append(dict(
        name="4 · pass-1 indexer", peaks=rp, fit=refined,
        result=res1, shift=SHIFT,
        plasma=(res1.temperature, res1.ne, res1.sigma, res1.gamma),
        species=_species_rows(idx1, res1, idx1._amp_sigma)))
    established = sorted([e for e, f in res1.element_fractions.items()
                          if f >= ESTABLISHED_MIN_FRACTION], key=element_sort_key)

    posterior = sorted({sp.element for sp in res1.species})
    refined, dec_phys = refine_fit(x, y, refined, db=_DB,
                                   elements=posterior or None,
                                   shift_nm=SHIFT, asymmetric="only")
    decisions = dec_data + dec_phys
    sa_zones = []
    for dec in decisions:
        if (dec.get("action") == "sa-tag"
                and str(dec.get("verdict", "")).startswith("asymmetric")
                and dec.get("params_asym") is not None):
            pS = dec.get("params_single")
            if pS is None:
                pS = dec["params_asym"]
            hw = 1.5 * max(float(voigt_width(max(pS[2], 1e-6),
                                             max(pS[3], 1e-6))), 0.15)
            sa_zones.append((float(pS[1]), hw))
    stages.append(dict(name="3b · self-absorption tags",
                       peaks=refined["sorted_parameter_array"], fit=refined))

    final = refined
    if established:
        final, _ = seed_minor_lines(x, y, refined, _DB, established,
                                    shift_nm=SHIFT, exclude=tuple(sa_zones))
    stages.append(dict(name="5 · Boltzmann-seeded minor lines",
                       peaks=final["sorted_parameter_array"], fit=final))
    final, _ = recover_residual_lines(x, y, final, exclude=tuple(sa_zones))
    stages.append(dict(name="6 · residual recovery",
                       peaks=final["sorted_parameter_array"], fit=final))
    prof = analyze_peak_profiles(x, y, final)
    final, _ = deblend_shoulders(x, y, final, prof, exclude=tuple(sa_zones))
    stages.append(dict(name="7 · shoulder deblends",
                       peaks=final["sorted_parameter_array"], fit=final))

    fp = final["sorted_parameter_array"]
    idx2 = PeakyIndexerV3(dbf(fp), dbpath=DBPATH,
                          temperature_init=res1.temperature, ne_init=res1.ne)
    idx2._amp_sigma = _amp_sigma(x, y, bg0, fp)
    res2 = idx2.run(**run_kw)
    stages.append(dict(
        name="8 · pass-2 indexer", peaks=fp, fit=final, result=res2,
        shift=SHIFT, plasma=(res2.temperature, res2.ne, res2.sigma, res2.gamma),
        species=_species_rows(idx2, res2, idx2._amp_sigma)))

    confirmed = sorted([e for e, f in res2.element_fractions.items()
                        if f >= ESTABLISHED_MIN_FRACTION], key=element_sort_key)
    result, fidx = res2, idx2
    if confirmed:
        scales, _ = match_and_scale(fp, _DB, confirmed, shift_nm=SHIFT)
        confident = sorted({e for (e, _s), i in scales.items()
                            if i["n_ref"] >= CONFIDENT_MIN_REFS},
                           key=element_sort_key)
        sup = []
        for el in confident:
            if el in _DB.no_lines:
                continue
            arr = _DB.lines(el)
            if arr.size == 0:
                continue
            mk = ((arr[:, 0].astype(float) <= 2)
                  & (arr[:, 3].astype(float) >= SUPPORT_GA_FLOOR))
            wl = arr[mk, 1].astype(float)
            if wl.size:
                sup.append(wl + shift_at(SHIFT, wl))
        supported = np.concatenate(sup) if sup else np.empty(0)
        work = final
        for bar in DEEPEN_BARS:
            if not confident:
                break
            work, corr = seed_minor_lines(x, y, work, _DB, confident,
                                          shift_nm=SHIFT, accept_snr=bar,
                                          min_expected_snr=bar,
                                          robust_elements=set(confident),
                                          exclude=tuple(sa_zones))
            work, rec = recover_residual_lines(
                x, y, work, exclude=tuple(sa_zones), supported_lines=supported,
                snr_min_supported=bar, accept_snr_supported=bar,
                support_tol_nm=SUPPORT_TOL_NM)
            if (sum(1 for r in corr if r.get("action") == "added")
                    + sum(1 for r in rec if r.get("action") == "added")) == 0:
                continue
            idxN = PeakyIndexerV3(dbf(work["sorted_parameter_array"]),
                                  dbpath=DBPATH, temperature_init=res2.temperature,
                                  ne_init=res2.ne)
            idxN._amp_sigma = _amp_sigma(x, y, bg0, work["sorted_parameter_array"])
            idxN.build_candidate_matrix(sa_doublets=True)
            resN = idxN.solve_at(res2.temperature, res2.ne, res2.sigma,
                                 res2.gamma)
            if composition_collapsed(res2.element_fractions,
                                     resN.element_fractions):
                break
            final, result, fidx = work, resN, idxN
        stages.append(dict(
            name="9 · iterative deepening (final)",
            peaks=final["sorted_parameter_array"], fit=final, result=result,
            shift=SHIFT,
            plasma=(result.temperature, result.ne, result.sigma, result.gamma),
            species=_species_rows(fidx, result, fidx._amp_sigma)))

    meta = dict(x=x, y=y, bg=bg0, shift=SHIFT)
    return stages, meta


def resolve(stage, meta, deleted, extra_species, removed_species):
    """Re-solve a stage at its fixed plasma state after edits.

    ``deleted``: set of peak-row indices to drop.  ``extra_species``: list
    of ``(element, ion)`` to force-include.  ``removed_species``: set of
    ``(element, ion)`` to exclude.  Returns ``(species_rows, r2, npeaks)``.
    """
    if "plasma" not in stage:
        return None
    T, ne, sg, gm = stage["plasma"]
    x, y, bg, shift = meta["x"], meta["y"], meta["bg"], meta["shift"]
    peaks = np.atleast_2d(np.asarray(stage["peaks"], dtype=float)).copy()
    keep = np.array([i not in deleted for i in range(peaks.shape[0])])
    peaks = peaks[keep]
    if peaks.shape[0] == 0:
        return [], float("nan"), 0
    dbp = peaks.copy()
    dbp[:, 1] -= shift_at(shift, dbp[:, 1])
    idx = PeakyIndexerV3(dbp, dbpath=DBPATH, temperature_init=T, ne_init=ne)
    idx._amp_sigma = _amp_sigma(x, y, bg, peaks)
    # min_init_relative_intensity=0 keeps ALL in-range species as candidates
    # so a user can ADD one the default prefilter would have dropped; the
    # keep-mask below restores the baseline set and applies the edits
    extra = [tuple(k) for k in extra_species]
    prefilter = 0.0 if extra else 1e-3
    idx.build_candidate_matrix(sa_doublets=True,
                               min_init_relative_intensity=prefilter)
    lt = idx.line_table
    base = {tuple(r["key"]) for r in (stage.get("species") or [])}
    want = (base | set(extra)) - {tuple(k) for k in removed_species}
    mask = np.array([(sp.element, int(sp.ion)) in want for sp in lt.species])
    if mask.any() and not mask.all():
        lt.filter_species(mask)
        # pseudo-observation weights + overlap must be rebuilt after a
        # species filter before the fixed-state solve
        idx._select_pseudo_wavelengths(T, ne)
    res = idx.solve_at(T, ne, sg, gm)
    return _species_rows(idx, res, idx._amp_sigma), float(res.r_squared), \
        int(peaks.shape[0])


def db_species_choices():
    """(element, ion) options a user can add, sorted by element."""
    out = []
    for el in sorted(_DB.elements, key=element_sort_key):
        if el in _DB.no_lines:
            continue
        for ion in (1, 2):
            out.append((el, ion))
    return out


# ===========================================================================
# Frontend: Plotly figures + Dash app
# ===========================================================================
def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb[:3])


def spectrum_figure(stage, meta, selected=None, deleted=None,
                    highlight_species=None):
    import plotly.graph_objects as go
    deleted = deleted or set()
    x, y, bg = meta["x"], meta["y"], meta["bg"]
    ybg = y - bg
    peaks = np.atleast_2d(np.asarray(stage["peaks"], dtype=float))
    model = (multi_voigt(x, np.ravel(peaks[:, :4])) if peaks.size
             else np.zeros_like(x))
    fig = go.Figure()
    # wavelength-coloured spectrum, drawn as short coloured segments
    step = max(1, x.size // 1400)
    xs, ys = x[::step], np.maximum(ybg[::step], 1e-1)
    seg_colors = [_hex(c) for c in wavelength_to_rgb(xs)]
    for i in range(len(xs) - 1):
        fig.add_trace(go.Scattergl(
            x=xs[i:i + 2], y=ys[i:i + 2], mode="lines",
            line=dict(color=seg_colors[i], width=1), hoverinfo="skip",
            showlegend=False))
    fig.add_trace(go.Scattergl(
        x=x, y=np.maximum(model, 1e-1), mode="lines",
        line=dict(color="rgba(90,90,90,0.9)", width=1), name="model"))
    # peaks as markers coloured by wavelength; click to select
    if peaks.size:
        idxs = np.arange(peaks.shape[0])
        alive = np.array([i not in deleted for i in idxs])
        colors = [_hex(c) for c in wavelength_to_rgb(peaks[:, 1])]
        for state, opacity, size in ((alive, 1.0, 8), (~alive, 0.25, 6)):
            if not np.any(state):
                continue
            fig.add_trace(go.Scattergl(
                x=peaks[state, 1], y=np.maximum(peaks[state, 0], 1.0),
                mode="markers",
                marker=dict(size=size, color=[colors[i] for i in idxs[state]],
                            line=dict(width=0.5, color="#222"),
                            opacity=opacity),
                customdata=idxs[state], name="peaks",
                hovertemplate="%{x:.3f} nm  area %{y:.1f}<extra>peak "
                              "%{customdata}</extra>", showlegend=False))
    if highlight_species is not None:
        hp = np.atleast_2d(highlight_species)
        if hp.size:
            fig.add_trace(go.Scattergl(
                x=hp[:, 0], y=np.maximum(hp[:, 1], 1.0), mode="markers",
                marker=dict(size=15, symbol="circle-open",
                            color="#000", line=dict(width=2)),
                name="supports selected", hoverinfo="skip"))
    if selected is not None and peaks.size and selected < peaks.shape[0]:
        p = peaks[selected]
        fig.add_vline(x=float(p[1]), line=dict(color="#000", width=1,
                                               dash="dot"))
    fig.update_yaxes(type="log", title="counts")
    fig.update_xaxes(title="wavelength [nm]")
    fig.update_layout(margin=dict(l=55, r=15, t=30, b=40), height=430,
                      title=stage["name"], plot_bgcolor="white",
                      hovermode="closest", uirevision="spectrum")
    return fig


def composition_figure(rows, baseline=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="no solved composition at this stage",
                          height=430)
        return fig
    labels = [r["label"] for r in rows]
    fr = np.array([r["emission"] for r in rows], dtype=float)
    fr = fr / max(fr.sum(), 1e-30)
    colors = [element_color(r["element"]) for r in rows]
    fig.add_trace(go.Bar(
        x=labels, y=fr, marker_color=colors,
        text=[f"z={r['z']:.0f}" if np.isfinite(r["z"]) else "" for r in rows],
        textposition="outside", name="composition"))
    if baseline is not None:
        bmap = {r["label"]: r["emission"] for r in baseline}
        bt = sum(bmap.values()) or 1.0
        by = [bmap.get(l, 0.0) / bt for l in labels]
        fig.add_trace(go.Scatter(
            x=labels, y=by, mode="markers",
            marker=dict(symbol="line-ew", size=16, color="#333",
                        line=dict(width=2)), name="baseline"))
    fig.update_yaxes(title="emission-weighted fraction")
    fig.update_layout(margin=dict(l=55, r=15, t=30, b=90), height=430,
                      barmode="overlay", plot_bgcolor="white",
                      title="per (element, ion-stage) — bars edited, ticks baseline",
                      uirevision="comp")
    return fig


def build_app(stages, meta, sample, log):
    from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

    app = Dash(__name__, title=f"fit inspector — {sample}")
    stage_opts = [{"label": s["name"], "value": i}
                  for i, s in enumerate(stages)]
    solved = [i for i, s in enumerate(stages) if "plasma" in s]
    default_stage = solved[-1] if solved else len(stages) - 1
    add_opts = [{"label": f"{el} {'I' * ion}", "value": f"{el}|{ion}"}
                for el, ion in db_species_choices()]

    app.layout = html.Div(style={"font-family": "sans-serif",
                                 "margin": "0 14px"}, children=[
        html.H3(f"LIBS fit inspector — {sample}"),
        html.Div(style={"display": "flex", "gap": "12px",
                        "align-items": "center", "flex-wrap": "wrap"}, children=[
            html.Label("stage:"),
            dcc.Dropdown(id="stage", options=stage_opts, value=default_stage,
                         clearable=False, style={"width": "360px"}),
            html.Button("◀ prev", id="prev", n_clicks=0),
            html.Button("next ▶", id="next", n_clicks=0),
            html.Button("⟲ reset edits", id="reset", n_clicks=0,
                        style={"margin-left": "20px"}),
            html.Span(id="status", style={"color": "#666"}),
        ]),
        html.Div(style={"display": "flex", "gap": "10px"}, children=[
            html.Div(style={"flex": "3"}, children=[
                dcc.Graph(id="spectrum")]),
            html.Div(style={"flex": "2"}, children=[
                dcc.Graph(id="composition")]),
        ]),
        html.Div(style={"display": "flex", "gap": "20px",
                        "margin-top": "6px"}, children=[
            html.Div(style={"flex": "1"}, children=[
                html.B("selected peak"),
                html.Div(id="peak-info",
                         style={"font-family": "monospace", "font-size": "13px",
                                "min-height": "44px"}),
                html.Button("🗑 delete this peak & re-solve", id="delpeak",
                            n_clicks=0, disabled=True),
            ]),
            html.Div(style={"flex": "1"}, children=[
                html.B("add an element / ion stage"),
                html.Div(style={"display": "flex", "gap": "6px"}, children=[
                    dcc.Dropdown(id="addspec", options=add_opts,
                                 placeholder="element + ion…",
                                 style={"width": "220px"}),
                    html.Button("＋ add & re-solve", id="addbtn", n_clicks=0),
                ]),
                html.Div("click a bar to remove that (element, ion) stage",
                         style={"color": "#888", "font-size": "12px",
                                "margin-top": "6px"}),
                html.Div(id="edits", style={"font-size": "12px",
                                            "margin-top": "6px"}),
            ]),
        ]),
        dcc.Store(id="edit-state",
                  data=dict(deleted=[], added=[], removed=[])),
    ])

    def _rows_for(stage_i, edit):
        stage = stages[stage_i]
        if "plasma" not in stage:
            return stage.get("species") or [], None, stage.get("species")
        base = stage.get("species") or []
        out = resolve(stage, meta,
                      set(map(int, edit["deleted"])),
                      [tuple(a) for a in edit["added"]],
                      {tuple(r) for r in edit["removed"]})
        if out is None:
            return base, None, base
        rows, r2, npk = out
        return rows, (r2, npk), base

    @app.callback(
        Output("stage", "value"),
        Input("prev", "n_clicks"), Input("next", "n_clicks"),
        State("stage", "value"), prevent_initial_call=True)
    def _step(_p, _n, cur):
        if ctx.triggered_id == "prev":
            return max(0, cur - 1)
        return min(len(stages) - 1, cur + 1)

    @app.callback(
        Output("edit-state", "data"),
        Output("peak-info", "children"), Output("delpeak", "disabled"),
        Input("stage", "value"), Input("reset", "n_clicks"),
        Input("spectrum", "clickData"), Input("delpeak", "n_clicks"),
        Input("addbtn", "n_clicks"), Input("composition", "clickData"),
        State("addspec", "value"), State("edit-state", "data"),
        prevent_initial_call=True)
    def _edit(stage_i, _r, click, _d, _a, comp_click, addval, edit):
        trig = ctx.triggered_id
        if trig in ("stage", "reset"):
            return dict(deleted=[], added=[], removed=[]), "—", True
        if trig in ("stage", "reset"):
            log.action(trig, stage=int(stage_i),
                       stage_name=stages[stage_i]["name"])
        sel = edit.get("_sel")
        if trig == "spectrum" and click and click.get("points"):
            pt = click["points"][0]
            cd = pt.get("customdata")
            if cd is not None:
                edit = dict(edit); edit["_sel"] = int(cd)
                stage = stages[stage_i]
                p = np.atleast_2d(stage["peaks"])[int(cd)]
                log.action("select_peak", stage=int(stage_i), peak=int(cd),
                           wavelength=round(float(p[1]), 3),
                           area=round(float(p[0]), 2))
                info = (f"peak {int(cd)}   λ={p[1]:.3f} nm   area={p[0]:.1f}\n"
                        f"σ={p[2]:.4f}  γ={p[3]:.4f}  "
                        f"FWHM={1000*voigt_width(max(p[2],1e-6),max(p[3],1e-6)):.0f} pm")
                return edit, info, False
            return no_update, no_update, no_update
        if trig == "delpeak" and sel is not None:
            stage = stages[stage_i]
            p = np.atleast_2d(stage["peaks"])[int(sel)]
            log.action("delete_peak", stage=int(stage_i), peak=int(sel),
                       wavelength=round(float(p[1]), 3),
                       area=round(float(p[0]), 2))
            edit = dict(edit); edit["deleted"] = list(edit["deleted"]) + [int(sel)]
            edit["_sel"] = None
            return edit, "deleted — re-solved", True
        if trig == "addbtn" and addval:
            el, ion = addval.split("|")
            log.action("add_species", stage=int(stage_i), element=el,
                       ion=int(ion))
            edit = dict(edit)
            edit["added"] = list(edit["added"]) + [[el, int(ion)]]
            return edit, no_update, no_update
        if trig == "composition" and comp_click and comp_click.get("points"):
            lab = comp_click["points"][0].get("x", "")
            parts = lab.split()
            if len(parts) == 2:
                el, ionr = parts[0], len(parts[1])
                log.action("remove_species", stage=int(stage_i), element=el,
                           ion=int(ionr))
                edit = dict(edit)
                edit["removed"] = list(edit["removed"]) + [[el, ionr]]
                return edit, no_update, no_update
        return no_update, no_update, no_update

    @app.callback(
        Output("spectrum", "figure"), Output("composition", "figure"),
        Output("status", "children"), Output("edits", "children"),
        Input("stage", "value"), Input("edit-state", "data"))
    def _render(stage_i, edit):
        stage = stages[stage_i]
        rows, delta, base = _rows_for(stage_i, edit)
        sel = edit.get("_sel")
        deleted = set(map(int, edit["deleted"]))
        # highlight the peaks supporting the (single) selected/added species? show plain
        specfig = spectrum_figure(stage, meta, selected=sel, deleted=deleted)
        compfig = composition_figure(rows, baseline=base if (edit["deleted"]
                                     or edit["added"] or edit["removed"]) else None)
        peaks = np.atleast_2d(stage["peaks"])
        n = peaks.shape[0] - len(deleted)
        status = f"{peaks.shape[0]} peaks"
        if "plasma" in stage:
            T, ne, _, _ = stage["plasma"]
            status += f"   T={T:.0f} K  log ne={ne:.2f}"
            if delta:
                status += f"   → re-solved r²={delta[0]:.3f}, {delta[1]} peaks"
        ed = []
        if edit["deleted"]:
            ed.append(f"deleted peaks: {edit['deleted']}")
        if edit["added"]:
            ed.append("added: " + ", ".join(f"{e} {'I'*i}" for e, i in edit["added"]))
        if edit["removed"]:
            ed.append("removed: " + ", ".join(f"{e} {'I'*int(i)}" for e, i in edit["removed"]))
        # log the resolved state — the (edits -> composition) training label
        log.state(stage=int(stage_i), stage_name=stage["name"],
                  edits=dict(deleted=list(edit["deleted"]),
                             added=edit["added"], removed=edit["removed"]),
                  n_peaks=int(n),
                  r2=(round(delta[0], 4) if delta else
                      (round(float(stage["result"].r_squared), 4)
                       if "result" in stage else None)),
                  composition=_comp_summary(rows))
        return specfig, compfig, status, " | ".join(ed)

    return app


def _pick_data(args):
    if args.file:
        return [args.file]
    cands = [args.data] if args.data else [
        os.path.join(REPO, "data", "remote_samples"),
        ("/Users/mwhittaker/Library/CloudStorage/GoogleDrive-mwhittaker@lbl.gov/"
         "My Drive/CMI/Data/LIBS/JChristensen/2026-07-01"),
    ]
    for d in cands:
        fs = sorted(f for f in glob.glob(os.path.join(d, "*.csv"))
                    if os.path.basename(f) not in ("summary.csv", "detections.csv"))
        if fs:
            return fs
    raise SystemExit("no spectra found; pass --file or --data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--data")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--log", default=os.path.join(
        REPO, "logs", "fit_inspector_actions.jsonl"),
        help="JSONL action log (training-data trace); appended to")
    args = ap.parse_args()
    try:
        import dash  # noqa: F401
        import plotly  # noqa: F401
    except ImportError:
        raise SystemExit("this tool needs Dash + Plotly:\n"
                         "    pip install dash")
    files = _pick_data(args)
    path = files[0]
    sample = sample_name(path)
    print(f"loading {sample} …", flush=True)
    x, y = load_spectrum_csv(path)
    print("running pipeline + capturing stages (~1-3 min)…", flush=True)
    stages, meta = capture(x, y)
    log = ActionLog(args.log, sample, len(x), stages)
    print(f"captured {len(stages)} stages; logging actions to {args.log}",
          flush=True)
    print(f"open http://127.0.0.1:{args.port}", flush=True)
    app = build_app(stages, meta, sample, log)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
