"""Saha-consistent synthetic round-trip cases (suite S).

Builds replayable scenes for :class:`alibz.synthetic.SyntheticSpectrumGenerator`
whose ion-stage split is CONSISTENT with the Saha balance at the scene's
(T, n_e) — the generator itself takes direct stage abundances and applies no
ionization physics, so without this step a round-trip against the
Saha-Boltzmann inverse model would carry built-in model mismatch that has
nothing to do with the inverse engine under test.

Cases are identified by ``SyntheticScene.digest``, so a pinned case list is
just (name, composition, T, log_ne, seed) — regeneration is deterministic and
verifiable.  Measured baseline (2026-07-16, gp search, defaults of the day):
the quartz-carbonate 4-element case MISSED Si (0.40 truth -> absent) and
hallucinated Be at 0.23 with n_e railed at 14.0; the binary case recovered
Ca 0.76/Mg 0.24 against 0.60/0.40 truth with T 7000 K against 10000 K.
Those numbers are the floor the fix program ratchets down from.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from alibz.synthetic import (PlasmaComponent, SyntheticScene,
                             SyntheticSpectrumGenerator, dry_air_component)
from alibz.utils.sahaboltzmann import SahaBoltzmann

#: rendering scales matched to real SciAps exports: bright-line peaks in the
#: 10^3-10^4 count range, vendor-background-subtracted pedestal near zero.
DEFAULT_EMISSION_SCALE = 0.1
DEFAULT_CONTINUUM_SCALE = 60.0


def saha_stage_map(composition: Mapping[str, float], temperature_k: float,
                   log_ne_cm3: float, dbpath: str = "db",
                   sb: Optional[SahaBoltzmann] = None,
                   ) -> Dict[Tuple[str, int], float]:
    """Element fractions -> per-(element, stage) fractions via Saha at (T, ne)."""
    sb = sb if sb is not None else SahaBoltzmann(dbpath)
    stage_map: Dict[Tuple[str, int], float] = {}
    for element, fraction in composition.items():
        ci, _Z = sb.ionization_distribution(element, temperature_k, log_ne_cm3)
        ions, _levels = sb._element_levels(element)
        for stage, stage_fraction in zip(ions, np.atleast_2d(ci)[0]):
            if float(stage_fraction) > 1e-6 and int(stage) <= 3:
                stage_map[(element, int(stage))] = (
                    float(fraction) * float(stage_fraction))
    return stage_map


def make_scene(composition: Mapping[str, float], temperature_k: float = 10_000.0,
               log_ne_cm3: float = 17.0, seed: int = 0,
               emission_scale: float = DEFAULT_EMISSION_SCALE,
               continuum_scale: float = DEFAULT_CONTINUUM_SCALE,
               argon_fraction: Optional[float] = None,
               dbpath: str = "db",
               sb: Optional[SahaBoltzmann] = None) -> SyntheticScene:
    """Build a Saha-consistent scene; ``argon_fraction`` adds an air/argon
    ambient component (the argon-vs-air validation axis)."""
    stage_map = saha_stage_map(composition, temperature_k, log_ne_cm3,
                               dbpath=dbpath, sb=sb)
    target = PlasmaComponent.from_mapping(
        stage_map, temperature_k=temperature_k, log_ne_cm3=log_ne_cm3,
        emission_scale=emission_scale, continuum_scale=continuum_scale)
    ambient = None
    if argon_fraction is not None:
        sb_local = sb if sb is not None else SahaBoltzmann(dbpath)
        gas_stages = {}
        for element in ("N", "O", "Ar"):
            ci, _Z = sb_local.ionization_distribution(
                element, temperature_k, log_ne_cm3)
            ions, _levels = sb_local._element_levels(element)
            split = np.zeros(3)
            for stage, f in zip(ions, np.atleast_2d(ci)[0]):
                if int(stage) <= 3:
                    split[int(stage) - 1] = float(f)
            total = split.sum()
            gas_stages[element] = (split / total if total > 0
                                   else np.array([1.0, 0.0, 0.0]))
        ambient = dry_air_component(
            argon_purge_fraction=float(argon_fraction),
            stage_fractions=gas_stages,
            temperature_k=temperature_k, log_ne_cm3=log_ne_cm3,
            emission_scale=emission_scale)
    return SyntheticScene(target=target, ambient_gas=ambient, seed=int(seed),
                          metadata={"truth_composition": dict(composition),
                                    "truth_temperature_k": temperature_k,
                                    "truth_log_ne_cm3": log_ne_cm3})


def render_case(composition: Mapping[str, float], seed: int = 0,
                dbpath: str = "db", add_noise: bool = True,
                generator: Optional[SyntheticSpectrumGenerator] = None,
                **scene_kwargs):
    """Render one case; returns ``(x, y, scene)``."""
    generator = (generator if generator is not None
                 else SyntheticSpectrumGenerator(dbpath))
    scene = make_scene(composition, seed=seed, dbpath=dbpath, **scene_kwargs)
    spectrum = generator.render(scene, add_noise=add_noise)
    return spectrum.wavelength_nm, spectrum.intensity_counts, scene


#: named frozen cases for suite S.  Keep names stable — scoreboard history
#: is keyed on them.  Compositions are element nuclei fractions of the
#: emitting plasma (the quantity the inverse model reports).
CASES: Dict[str, dict] = {
    # the fast-CI binary: resolvable, no confounder traps
    "binary_ca_mg": dict(composition={"Ca": 0.6, "Mg": 0.4},
                         temperature_k=10_000.0, log_ne_cm3=17.0),
    # 4-element silicate-like mix; measured to expose the Si-miss/Be
    # hallucination failure at baseline
    "silicate_4el": dict(composition={"Si": 0.4, "Ca": 0.3,
                                      "Fe": 0.2, "Mg": 0.1},
                         temperature_k=10_000.0, log_ne_cm3=17.0),
    # alkali-bearing feldspar-like mix: exercises the K/Na resonance-SA
    # degeneracy that moves alkalis 30-60% on real spectra
    "feldspar_k_na": dict(composition={"Si": 0.5, "Al": 0.2,
                                       "K": 0.2, "Na": 0.1},
                          temperature_k=9_000.0, log_ne_cm3=17.0),
    # line-rich matrix at low T: the composition-collapse regime
    "iron_rich_low_t": dict(composition={"Fe": 0.6, "Ti": 0.2, "Ca": 0.2},
                            temperature_k=7_000.0, log_ne_cm3=16.5),
    # argon-ambient variant of the binary (argon-vs-air validation axis)
    "binary_argon_ambient": dict(composition={"Ca": 0.6, "Mg": 0.4},
                                 temperature_k=10_000.0, log_ne_cm3=17.0,
                                 argon_fraction=1.0),
}


def score_recovery(truth: Mapping[str, float],
                   recovered: Mapping[str, float],
                   major_floor: float = 0.05) -> dict:
    """Composition-recovery metrics for one case.

    ``majors_missed`` counts truth elements >= ``major_floor`` absent from
    the recovery (fraction < major_floor / 2); ``spurious_mass`` is total
    recovered fraction on elements not in the truth at all.
    """
    truth = {el: float(f) for el, f in truth.items()}
    rec = {el: float(f) for el, f in recovered.items() if f > 0}
    majors = {el: f for el, f in truth.items() if f >= major_floor}
    missed = [el for el in majors if rec.get(el, 0.0) < major_floor / 2.0]
    rel_err = {el: abs(rec.get(el, 0.0) - f) / f for el, f in majors.items()}
    spurious = {el: f for el, f in rec.items() if el not in truth}
    return dict(
        majors_missed=missed,
        n_majors=len(majors),
        rel_err=rel_err,
        median_rel_err=float(np.median(list(rel_err.values())))
        if rel_err else float("nan"),
        max_rel_err=float(max(rel_err.values())) if rel_err else float("nan"),
        spurious_mass=float(sum(spurious.values())),
        spurious={el: round(f, 4) for el, f in
                  sorted(spurious.items(), key=lambda kv: -kv[1])[:5]},
        dominant_correct=(max(rec, key=rec.get) == max(truth, key=truth.get))
        if rec else False,
    )
