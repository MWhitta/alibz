"""Canonical NIST-backed forward model for the CF-LIBS benchmark.

Bridges the alibz atomic database (``db/`` — real NIST lines, degeneracies,
level energies and ionization energies for 92 elements) into the physically
complete ``dev/dev1/cflibs`` forward model (per-line thermal-Doppler +
electron-impact-Stark Voigt profiles, Saha ionization balance, self-absorption
escape factors, VarPro linear-in-concentration design).

Why this bridge exists
----------------------
The stock cflibs demo builds a *procedural* atomic database and only closes the
forward->inverse loop against itself. The alibz ``peaky_maker`` uses real NIST
lines but with a single, non-physical, T/Ne-independent Voigt width. Neither is
a trustworthy generator for training/benchmarking inverse models. This module
gives the cflibs engine REAL atomic data so it can serve as the canonical
generator (Objective 2).

Unit / convention notes
-----------------------
* alibz ``db`` per-element array columns used here:
  col0 = ion stage (1-based; 1 = neutral), col1 = wavelength (nm, air),
  col3 = gA = g_k * A_ki (s^-1), col4 = E_i (eV), col5 = E_k (eV),
  col12 = g_i, col13 = g_k.
* cflibs ``Line`` wants A_ki alone, so A_ki = gA / g_k.
* cflibs ``ion_stage`` is 0-based (0 = neutral) = (alibz stage - 1).
* Ionization energies come from alibz ``db/ionization`` ([Z, 0-based stage,
  E_ion eV]); we push them into ``cflibs.atomic.IONIZATION_ENERGIES_EV`` so the
  Saha chain uses consistent, per-element data rather than the stock 20-element
  hardcoded table.

Partition functions
-------------------
Built by direct summation over the union of lower/upper levels that appear in
the element's *full* line list for each stage (NOT restricted to the in-band
emission lines), truncated by the Debye continuum-lowering cutoff via cflibs'
``PartitionFunction``. This is the same level-union approximation alibz uses,
but with the physically-correct plasma cutoff. Tabulated U(T) can be dropped in
later per (element, stage) with ``partition_tables=``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --- the dev/dev1/cflibs prototype this module bridges to was REMOVED from
#     the worktree (dev1 ideas were salvaged into alibz; see commit 7e36dad).
#     Import is guarded so bench/ stays importable from a clean checkout:
#     the cflibs engine reports unavailable instead of breaking the whole
#     benchmark package.  The in-repo generator now lives in
#     bench/synth_cases.py (alibz.synthetic). ---
_CFLIBS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dev", "dev1", "cflibs")
)
if _CFLIBS_ROOT not in sys.path:
    sys.path.insert(0, _CFLIBS_ROOT)

try:
    from cflibs import atomic as catomic          # noqa: E402
    from cflibs.atomic import Level, Line, PartitionFunction  # noqa: E402
    from cflibs.forward import ForwardModel        # noqa: E402
    CFLIBS_AVAILABLE = True
    CFLIBS_UNAVAILABLE_REASON = None
except ImportError as _exc:                        # pragma: no cover
    catomic = Level = Line = PartitionFunction = ForwardModel = None
    CFLIBS_AVAILABLE = False
    CFLIBS_UNAVAILABLE_REASON = (
        f"cflibs prototype not present at {_CFLIBS_ROOT}: {_exc}")

from alibz.utils.database import Database       # noqa: E402

# Column indices into the alibz per-element line array (see module docstring).
# Order below == the columns of ``num`` used throughout: stage, wl, gA, Ei, Ek, gi, gk.
_COL_ORDER = [0, 1, 3, 4, 5, 12, 13]

STARK_NE_REF_CM3 = 1.0e17


def stark_fwhm_ref_nm(
    wavelength_nm: float,
    Ek_eV: float,
    chi_eV: float,
    coeff_nm: float = 0.018,
    high_level_gain: float = 3.0,
) -> float:
    """Approximate electron-impact Stark FWHM (nm) at ``STARK_NE_REF_CM3``.

    A first-order, deliberately simple model — tabulated Stark parameters do not
    exist for most lines, and the well-documented 20-50% systematic uncertainty
    in those that do is absorbed downstream by the fitted ``stark_scale``
    nuisance parameter. Two physically-motivated dependences are kept:

    * ``(lambda / 500 nm)^2`` — Stark width in wavelength units grows ~ lambda^2
      (the impact width is roughly constant in angular-frequency units).
    * levels lying close to the ionization limit are more polarizable and
      Stark-broaden more strongly, captured by ``1 + gain * (Ek / chi)``.

    Returns a per-line reference width; the actual width scales linearly with
    Ne in :func:`cflibs.lineshape.stark_lorentz_fwhm_nm`.
    """
    lam_factor = (wavelength_nm / 500.0) ** 2
    if chi_eV and chi_eV > 0:
        level_factor = 1.0 + high_level_gain * min(max(Ek_eV / chi_eV, 0.0), 1.0)
    else:
        level_factor = 1.0
    return coeff_nm * lam_factor * level_factor


def _ionization_chi_list(db: Database, element: str, n_stage: int) -> List[float]:
    """Return [chi(0->1), chi(1->2), ...] in eV from the alibz ionization db."""
    ion = np.asarray(db.ionization_energy(element), dtype=float)
    chis: List[float] = []
    for j in range(n_stage):
        row = ion[ion[:, 1].astype(int) == j]
        chis.append(float(row[0, -1]) if row.size else np.inf)
    return chis


def _stage_levels(num: np.ndarray, stage_1based: int) -> List[Level]:
    """Deduplicated Level list for one ion stage from the full line array.

    Uses the union of lower (Ei, gi) and upper (Ek, gk) levels across ALL lines
    of the stage (not grid-limited) so the partition sum is as complete as the
    line list allows.
    """
    m = num[:, 0] == stage_1based
    if not np.any(m):
        return [Level(energy_eV=0.0, g=1.0)]
    sub = num[m]
    energies = np.concatenate([sub[:, 3], sub[:, 4]])          # Ei, Ek
    degens = np.concatenate([sub[:, 5], sub[:, 6]])            # gi, gk
    good = np.isfinite(energies) & np.isfinite(degens) & (degens > 0)
    energies, degens = energies[good], degens[good]
    if energies.size == 0:
        return [Level(energy_eV=0.0, g=1.0)]
    # deduplicate on rounded (energy, g)
    key = np.round(np.column_stack([energies, degens]), 6)
    _, uniq = np.unique(key, axis=0, return_index=True)
    return [Level(energy_eV=float(energies[i]), g=float(degens[i])) for i in uniq]


def build_nist_forward(
    elements: Sequence[str],
    wavelength_grid_nm: np.ndarray,
    dbpath: str = "db",
    max_stage: int = 2,
    window_pad_nm: float = 2.0,
    min_gA: float = 0.0,
    partition_tables: Optional[Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[ForwardModel, Dict]:
    """Construct a cflibs :class:`ForwardModel` from real NIST atomic data.

    Parameters
    ----------
    elements : sequence of str
        Element symbols to include (e.g. ``["Ca", "Fe", "Si", ...]``).
    wavelength_grid_nm : ndarray
        Instrument wavelength axis (nm) the spectrum is synthesized onto.
    max_stage : int
        Number of ionization stages per element (0-based count; 2 = neutral+ion).
    window_pad_nm : float
        Include lines slightly outside the grid so their wings can contribute.
    min_gA : float
        Drop emission lines with gA below this (0 keeps all).
    partition_tables : optional
        Per (element, 0-based stage) tabulated ``(T_array, U_array)`` to override
        the level-sum partition function.

    Returns
    -------
    (ForwardModel, meta_dict)
        ``meta`` reports per-element line/level counts and the ionization
        energies used.
    """
    if not CFLIBS_AVAILABLE:
        raise ImportError(CFLIBS_UNAVAILABLE_REASON)
    db = Database(dbpath)
    grid = np.asarray(wavelength_grid_nm, dtype=float)
    lo, hi = grid.min() - window_pad_nm, grid.max() + window_pad_nm

    lines: List[Line] = []
    pf: Dict[Tuple[str, int], PartitionFunction] = {}
    n_stages: Dict[str, int] = {}
    meta: Dict[str, Dict] = {}
    partition_tables = partition_tables or {}

    for el in elements:
        arr = db.atom_dict.get(el)
        if arr is None:
            raise KeyError(f"element {el!r} not in database")
        num = arr[:, _COL_ORDER].astype(float)  # cols: stage, wl, gA, Ei, Ek, gi, gk
        chis = _ionization_chi_list(db, el, max_stage)
        # Push consistent ionization energies into the cflibs Saha table.
        catomic.IONIZATION_ENERGIES_EV[el] = [c for c in chis if np.isfinite(c)] or [np.inf]

        n_stages[el] = max_stage
        el_line_count = 0
        el_level_count = 0
        for stage0 in range(max_stage):
            stage1 = stage0 + 1
            chi = chis[stage0] if stage0 < len(chis) else np.inf

            # Partition function over the full stage level list (with Debye cutoff),
            # unless a tabulated U(T) was supplied.
            if (el, stage0) in partition_tables:
                T_tab, U_tab = partition_tables[(el, stage0)]
                pf[(el, stage0)] = PartitionFunction(table_T=T_tab, table_U=U_tab)
            else:
                levels = _stage_levels(num, stage1)
                el_level_count += len(levels)
                pf[(el, stage0)] = PartitionFunction(
                    levels=levels, chi_eV=(chi if np.isfinite(chi) else None)
                )

            # Emission lines for this stage, restricted to the (padded) grid.
            m = (
                (num[:, 0] == stage1)
                & (num[:, 1] >= lo)
                & (num[:, 1] <= hi)
                & (num[:, 2] > min_gA)
                & np.isfinite(num[:, 5])
                & (num[:, 6] > 0)
            )
            for _, wl, gA, Ei, Ek, gi, gk in num[m]:
                Aki = gA / gk if gk > 0 else gA
                lines.append(
                    Line(
                        wavelength_nm=float(wl),
                        Aki=float(Aki),
                        gk=float(gk),
                        gi=float(gi),
                        Ek_eV=float(Ek),
                        Ei_eV=float(Ei),
                        element=el,
                        ion_stage=stage0,
                        stark_fwhm_ref_nm=stark_fwhm_ref_nm(float(wl), float(Ek), chi),
                        stark_Ne_ref_cm3=STARK_NE_REF_CM3,
                    )
                )
                el_line_count += 1

        meta[el] = dict(
            n_lines=el_line_count,
            n_levels=el_level_count,
            ionization_eV=[c for c in chis if np.isfinite(c)],
        )

    fm = ForwardModel(lines, grid, list(elements), pf, n_stages)
    meta["_total_lines"] = len(fm.lines)
    return fm, meta


def default_rock_elements() -> List[str]:
    """Major rock-forming elements + H (for the H-alpha Ne anchor).

    Chosen for the argon-vs-air validation target (a Ca-rich rock/mineral).
    """
    return ["Si", "Al", "Fe", "Ca", "Mg", "Na", "K", "Ti", "Mn", "H"]
