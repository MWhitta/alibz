"""Diagnose the indexer pass-2 r^2 collapse (amplitude space) on a spectrum.

Mirrors analyze_spectrum's pass-1 / pass-2 configuration, then runs
controlled experiments separating (a) the peak-table change from (b) the
optimiser basin from (c) the indexer configuration.
"""
import sys, time, copy
import numpy as np
sys.path.insert(0, "/Users/mwhittaker/Projects/github/alibz")
from alibz.pipeline import (load_spectrum_csv, resolve_dbpath, _halpha_ne,
                            ESTABLISHED_MIN_FRACTION, POISSON_GAIN_COUNTS,
                            _get_db, _get_sb)
from alibz.elements import element_sort_key
from alibz.peaky_finder import PeakyFinder
from alibz.refinement import refine_fit
from alibz.minor_lines import seed_minor_lines, recover_residual_lines
from alibz.profiles import analyze_peak_profiles, deblend_shoulders
from alibz.peaky_indexer_v3 import PeakyIndexerV3
from alibz.inspection import estimate_peak_uncertainties
from alibz.detector import (correct_segment_response, estimate_segment_response,
                            segment_response_fallback)
from alibz.utils.wavelength import (estimate_wavelength_shift,
                                    estimate_wavelength_shift_segments, shift_at)
from alibz.utils.voigt import voigt_width

f = sys.argv[1]
x, y = load_spectrum_csv(f"/Users/mwhittaker/Projects/github/alibz/data/remote_samples/{f}.csv")
dbpath = resolve_dbpath(); db = _get_db(dbpath); sb = _get_sb(dbpath)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

finder = PeakyFinder.__new__(PeakyFinder)
fit = finder.fit_spectrum(x, y, subtract_background=True, plot=False, n_sigma=0)
bg0 = np.asarray(fit["background"], float)
shift0, _ = estimate_wavelength_shift(fit["sorted_parameter_array"], db)
refined, dec_data = refine_fit(x, y, fit, db=db, shift_nm=shift0, asymmetric="defer")
shift, _ = estimate_wavelength_shift_segments(refined["sorted_parameter_array"], db)
rp = refined["sorted_parameter_array"]
ne_init, ne_bounds = _halpha_ne(rp)
ne_prior = ((float(ne_init), 0.15) if ne_init is not None else (17.0, 0.5))
_resid = y - bg0
_noise = 1.4826 * float(np.median(np.abs(_resid - np.median(_resid))))
fallback, fallback_meta = segment_response_fallback(edges=(620.0,), return_metadata=True)
seg_response, _ = estimate_segment_response(
    x, bg0, edges=(620.0,), noise_scale=_noise, fallback=fallback,
    fallback_uncertainty=[r.get("uncertainty") for r in fallback_meta], return_metadata=True)
idx_kwargs = dict(dbpath=dbpath, db=db, sb=sb, weighted_solve=False, ne_prior=ne_prior,
                  amp_sigma_floor=_noise, amp_sigma_poisson_gain=POISSON_GAIN_COUNTS)
run_kwargs = dict(sa_doublets=True, n_calls=40, verbose=False,
                  sa_stimulated_emission=False, search="gp", random_state=0)
if ne_init is not None:
    idx_kwargs["ne_init"] = ne_init; run_kwargs["ne_bounds"] = ne_bounds

def dbf(p):
    o = p.copy(); o[:, 1] -= shift_at(shift, o[:, 1])
    return correct_segment_response(o, seg_response, edges=(620.0,))
def amp_sigma(p):
    sig = estimate_peak_uncertainties(x, y - bg0, p)[:, 0]
    seg = np.searchsorted(np.asarray([620.0]), np.atleast_2d(p)[:, 1])
    return sig / seg_response[seg]

def summarize(tag, idx, res):
    fr = sorted(res.element_fractions.items(), key=lambda kv: -kv[1])[:8]
    log(f"{tag}: n_peaks={idx.n_peaks} T={res.temperature:.0f} ne={res.ne:.2f} "
        f"sig={res.sigma:.3f} gam={res.gamma:.3f} cost={res.cost:.3e} r2={res.r_squared:.3f} "
        f"n_species={len(res.species)} top={[(e, round(v,3)) for e, v in fr]}")

log("pass 1 ...")
idx1 = PeakyIndexerV3(dbf(rp), amp_sigma=amp_sigma(rp), **idx_kwargs)
res1 = idx1.run(**run_kwargs)
summarize("PASS1", idx1, res1)
established = sorted([e for e, v in res1.element_fractions.items() if v >= ESTABLISHED_MIN_FRACTION], key=element_sort_key)
posterior = sorted({sp.element for sp in res1.species})
refined, dec_phys = refine_fit(x, y, refined, db=db, elements=posterior or None, shift_nm=shift, asymmetric="only")
sa_zones = []
for dec in dec_data + dec_phys:
    if dec.get("action") == "sa-tag" and str(dec.get("verdict", "")).startswith("asymmetric") and dec.get("params_asym") is not None:
        pS = dec.get("params_single")
        if pS is None: pS = dec["params_asym"]
        sa_zones.append((float(pS[1]), 1.5 * max(float(voigt_width(max(pS[2], 1e-6), max(pS[3], 1e-6))), 0.15)))
final = refined
if established:
    final, _ = seed_minor_lines(x, y, refined, db, established, shift_nm=shift, exclude=tuple(sa_zones))
n5 = final["sorted_parameter_array"].shape[0]
final, _ = recover_residual_lines(x, y, final, exclude=tuple(sa_zones))
n6 = final["sorted_parameter_array"].shape[0]
prof = analyze_peak_profiles(x, y, final)
final, _ = deblend_shoulders(x, y, final, prof, exclude=tuple(sa_zones))
fp = final["sorted_parameter_array"]
log(f"table: pass1 {rp.shape[0]} -> 3b {refined['sorted_parameter_array'].shape[0]} -> seed {n5} -> recover {n6} -> deblend {fp.shape[0]}")

log("pass 2 ...")
kw2 = dict(idx_kwargs, temperature_init=res1.temperature, ne_init=res1.ne)
idx2 = PeakyIndexerV3(dbf(fp), amp_sigma=amp_sigma(fp), **kw2)
res2 = idx2.run(**run_kwargs)
summarize("PASS2", idx2, res2)

# E1: pass-2 table at the pass-1 plasma state
idxA = PeakyIndexerV3(dbf(fp), amp_sigma=amp_sigma(fp), **kw2)
idxA.build_candidate_matrix(sa_doublets=True)
resA = idxA.solve_at(res1.temperature, res1.ne, res1.sigma, res1.gamma)
summarize("E1 pass2-table @ pass1-state", idxA, resA)
# E1b: pass-2 table at the pass-2 state via solve_at (sanity: should equal PASS2)
resA2 = idxA.solve_at(res2.temperature, res2.ne, res2.sigma, res2.gamma)
summarize("E1b pass2-table @ pass2-state (solve_at)", idxA, resA2)
# E3: pass-1 table, pass-2 config (warm start) re-optimised
idxB = PeakyIndexerV3(dbf(rp), amp_sigma=amp_sigma(rp), **kw2)
resB = idxB.run(**run_kwargs)
summarize("E3 pass1-table, pass2-config", idxB, resB)
# E4: pass-2 table, pass-1 config (no warm start)
idxC = PeakyIndexerV3(dbf(fp), amp_sigma=amp_sigma(fp), **idx_kwargs)
resC = idxC.run(**run_kwargs)
summarize("E4 pass2-table, pass1-config", idxC, resC)
# E5: grid search instead of gp on the pass-2 table
idxD = PeakyIndexerV3(dbf(fp), amp_sigma=amp_sigma(fp), **kw2)
resD = idxD.run(**dict(run_kwargs, search="grid"))
summarize("E5 pass2-table, search=grid", idxD, resD)

# E6: pass-1 table with PHYSICAL width bounds (2x median fitted width) and gp
from alibz.utils.voigt import voigt_width as _vw
def wb(p):
    s_med = float(np.median(np.maximum(p[:,2],1e-3))); g_med = float(np.median(np.maximum(p[:,3],1e-3)))
    return dict(sigma_bounds=(0.01, max(2*s_med,0.02)), gamma_bounds=(0.01, max(2*g_med,0.02)))
log(f"width bounds pass1-table: {wb(rp)}  pass2-table: {wb(fp)}")
idxE = PeakyIndexerV3(dbf(rp), amp_sigma=amp_sigma(rp), **idx_kwargs)
resE = idxE.run(**dict(run_kwargs, **wb(rp)))
summarize("E6 pass1-table, gp, physical width bounds", idxE, resE)
idxF = PeakyIndexerV3(dbf(fp), amp_sigma=amp_sigma(fp), **idx_kwargs)
resF = idxF.run(**dict(run_kwargs, **wb(fp)))
summarize("E7 pass2-table, gp, physical width bounds", idxF, resF)
idxG = PeakyIndexerV3(dbf(rp), amp_sigma=amp_sigma(rp), **idx_kwargs)
resG = idxG.run(**dict(run_kwargs, search="grid"))
summarize("E8 pass1-table, search=grid", idxG, resG)
idxH = PeakyIndexerV3(dbf(fp), amp_sigma=amp_sigma(fp), **idx_kwargs)
resH = idxH.run(**dict(run_kwargs, search="grid", **wb(fp)))
summarize("E9 pass2-table, grid + physical width bounds", idxH, resH)
# E2: r2 decomposition of PASS2: which peaks carry the residual
obs, pred = np.asarray(res2.observed), np.asarray(res2.predicted)
r = obs - pred; ss_tot = np.sum((obs - obs.mean()) ** 2)
order = np.argsort(-np.abs(r))[:12]
cen = idx2.peak_array[:, 1]
log("PASS2 top residual peaks (center_db_frame, obs, pred, share of SS_res, assigned):")
assign = {pa[0] if isinstance(pa, (tuple, list)) else i: pa for i, pa in enumerate(res2.peak_assignments)}
for i in order:
    pa = res2.peak_assignments[i] if i < len(res2.peak_assignments) else None
    print(f"   {cen[i]:9.3f}  obs={obs[i]:10.1f} pred={pred[i]:10.1f}  {r[i]**2/np.sum(r**2):5.1%}  {pa}")
# r2 on peaks present in the pass-1 table (match centers within 0.02 nm in db frame)
c1 = idx1.peak_array[:, 1]
in1 = np.array([np.min(np.abs(c1 - c)) < 0.02 for c in cen])
def r2_sub(m):
    o, p = obs[m], pred[m]
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
log(f"PASS2 r2 on peaks also in pass-1 table ({in1.sum()}/{in1.size}): {r2_sub(in1):.3f}; on new peaks only: {r2_sub(~in1):.3f}")
# same decomposition for E1 (pass-1 state)
obsA, predA = np.asarray(resA.observed), np.asarray(resA.predicted)
def r2_sub2(o, p, m):
    o, p = o[m], p[m]; return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
log(f"E1 r2 on old peaks: {r2_sub2(obsA, predA, in1):.3f}; new peaks: {r2_sub2(obsA, predA, ~in1):.3f}")
# PASS1 top residual peaks
obs1, pred1 = np.asarray(res1.observed), np.asarray(res1.predicted); r1 = obs1 - pred1
log("PASS1 top residual peaks:")
for i in np.argsort(-np.abs(r1))[:6]:
    print(f"   {c1[i]:9.3f}  obs={obs1[i]:10.1f} pred={pred1[i]:10.1f}  {r1[i]**2/np.sum(r1**2):5.1%}  {res1.peak_assignments[i] if i < len(res1.peak_assignments) else None}")
log("done")
