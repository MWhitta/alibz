"""Rock-stratified whole-rock composition priors for synthetic LIBS scenes.

The model artifact is built from the Gard, Hasterok, and Halpin (2019)
global whole-rock compilation.  It is deliberately separate from the plasma
renderer: this module samples a bulk-composition-shaped nuclei prior, while
``alibz.synthetic`` independently samples ion stages and plasma nuisance
parameters without a Saha constraint.

Missing database fields are unknown, not chemical zeros.  Negative database
values encode left-censored (below-detection-limit) measurements; they are
sampled below their reported limits and are tracked in the returned record.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

from alibz.elements import ATOMIC_NUMBER, ELEMENTS_BY_ATOMIC_NUMBER
from alibz.synthetic import ATOMIC_MASS_U, N_ELEMENTS, N_STAGES


WHOLE_ROCK_MODEL_SCHEMA = "alibz-whole-rock-prior-v1"
DEFAULT_TRAINING_EXCLUSIONS = (
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Pa",
)


def _readonly(value, dtype=float):
    out = np.asarray(value, dtype=dtype).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class WholeRockComposition:
    """One replayable composition draw in the fixed 92-element schema."""

    mass_fraction: np.ndarray
    nuclei_fraction: np.ndarray
    modeled_mask: np.ndarray
    below_detection_mask: np.ndarray
    stratum: str
    seed: int
    source_doi: str

    def __post_init__(self):
        mass = _readonly(self.mass_fraction)
        nuclei = _readonly(self.nuclei_fraction)
        modeled = _readonly(self.modeled_mask, bool)
        censored = _readonly(self.below_detection_mask, bool)
        for name, value in (
            ("mass_fraction", mass),
            ("nuclei_fraction", nuclei),
            ("modeled_mask", modeled),
            ("below_detection_mask", censored),
        ):
            if value.shape != (N_ELEMENTS,):
                raise ValueError(f"{name} must have 92 H--U positions")
        if np.any(mass < 0) or np.any(nuclei < 0):
            raise ValueError("composition fractions cannot be negative")
        if not np.isclose(mass.sum(), 1.0, atol=1e-10):
            raise ValueError("mass fractions must sum to one")
        if not np.isclose(nuclei.sum(), 1.0, atol=1e-10):
            raise ValueError("nuclei fractions must sum to one")
        object.__setattr__(self, "mass_fraction", mass)
        object.__setattr__(self, "nuclei_fraction", nuclei)
        object.__setattr__(self, "modeled_mask", modeled)
        object.__setattr__(self, "below_detection_mask", censored)


class WholeRockCompositionModel:
    """Censored empirical-marginal Gaussian-copula composition model.

    Marginal concentration distributions are empirical log10(ppm) quantiles.
    A positive-semidefinite, shrinkage-regularized Gaussian copula preserves
    the joint major/trace-element structure within rock strata despite the
    database's heterogeneous reporting.  Copula fitting uses pairwise-complete
    positive measurements; censored values affect marginal lower tails but
    are never treated as detections.
    """

    REQUIRED_ARRAYS = {
        "schema", "elements", "strata", "stratum_sample_count",
        "quantile_probability", "detected_log10_ppm_quantile",
        "limit_log10_ppm_quantile", "detected_count", "censored_count",
        "reporting_rate", "censored_fraction", "modeled_mask",
        "correlation", "training_mask", "source_doi",
    }

    def __init__(self, artifact):
        self.artifact = Path(artifact)
        with np.load(self.artifact, allow_pickle=False) as data:
            missing = self.REQUIRED_ARRAYS.difference(data.files)
            if missing:
                raise ValueError(f"whole-rock artifact is missing {sorted(missing)}")
            schema = str(np.asarray(data["schema"]).item())
            if schema != WHOLE_ROCK_MODEL_SCHEMA:
                raise ValueError(f"unsupported whole-rock schema: {schema}")
            elements = tuple(str(x) for x in data["elements"].tolist())
            if elements != tuple(ELEMENTS_BY_ATOMIC_NUMBER):
                raise ValueError("whole-rock artifact must retain H--U ordering")
            self.strata = tuple(str(x) for x in data["strata"].tolist())
            self.stratum_sample_count = np.asarray(
                data["stratum_sample_count"], dtype=np.int64
            )
            self.quantile_probability = np.asarray(
                data["quantile_probability"], dtype=float
            )
            self.detected_log10_ppm_quantile = np.asarray(
                data["detected_log10_ppm_quantile"], dtype=float
            )
            self.limit_log10_ppm_quantile = np.asarray(
                data["limit_log10_ppm_quantile"], dtype=float
            )
            self.detected_count = np.asarray(data["detected_count"], dtype=np.int64)
            self.censored_count = np.asarray(data["censored_count"], dtype=np.int64)
            self.reporting_rate = np.asarray(data["reporting_rate"], dtype=float)
            self.censored_fraction = np.asarray(
                data["censored_fraction"], dtype=float
            )
            self.modeled_mask = np.asarray(data["modeled_mask"], dtype=bool)
            self.correlation = np.asarray(data["correlation"], dtype=float)
            self.training_mask = np.asarray(data["training_mask"], dtype=bool)
            self.source_doi = str(np.asarray(data["source_doi"]).item())

        self._validate()
        self._stratum_index = {name: i for i, name in enumerate(self.strata)}
        self._copula_root = []
        for matrix in self.correlation:
            values, vectors = np.linalg.eigh(matrix)
            root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
            self._copula_root.append(root)

        manifest_path = self.artifact.with_name(
            self.artifact.stem + "_manifest.json"
        )
        self.manifest = None
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())

    @classmethod
    def load(cls, artifact="db/whole_rock_prior_v1.npz"):
        return cls(artifact)

    def _validate(self):
        n_strata = len(self.strata)
        n_q = self.quantile_probability.size
        expected = (n_strata, N_ELEMENTS)
        if not self.strata or len(set(self.strata)) != n_strata:
            raise ValueError("whole-rock strata must be nonempty and unique")
        if self.stratum_sample_count.shape != (n_strata,):
            raise ValueError("invalid stratum_sample_count shape")
        if (n_q < 3 or np.any(np.diff(self.quantile_probability) <= 0)
                or self.quantile_probability[0] <= 0
                or self.quantile_probability[-1] >= 1):
            raise ValueError("quantile probabilities must increase within (0, 1)")
        if self.detected_log10_ppm_quantile.shape != expected + (n_q,):
            raise ValueError("invalid detected-quantile shape")
        if self.limit_log10_ppm_quantile.shape != expected + (n_q,):
            raise ValueError("invalid censor-limit-quantile shape")
        for name in (
            "detected_count", "censored_count", "reporting_rate",
            "censored_fraction", "modeled_mask",
        ):
            if np.asarray(getattr(self, name)).shape != expected:
                raise ValueError(f"invalid {name} shape")
        if self.correlation.shape != (n_strata, N_ELEMENTS, N_ELEMENTS):
            raise ValueError("invalid correlation shape")
        if self.training_mask.shape != (N_ELEMENTS,):
            raise ValueError("training_mask must have 92 positions")
        if np.any((self.reporting_rate < 0) | (self.reporting_rate > 1)):
            raise ValueError("reporting rates must lie in [0, 1]")
        if np.any((self.censored_fraction < 0) | (self.censored_fraction > 1)):
            raise ValueError("censored fractions must lie in [0, 1]")
        for matrix in self.correlation:
            if (not np.allclose(matrix, matrix.T, atol=2e-6)
                    or not np.allclose(np.diag(matrix), 1.0, atol=2e-6)
                    or np.linalg.eigvalsh(matrix).min() < -2e-5):
                raise ValueError("correlation matrices must be PSD with unit diagonal")

    @property
    def selectable_strata(self) -> Tuple[str, ...]:
        return tuple(name for name in self.strata if name != "global")

    @property
    def training_strata(self) -> Tuple[str, ...]:
        """Classified strata eligible for equal-weight training draws."""
        return tuple(
            name for name in self.strata
            if name not in ("global", "unclassified")
        )

    def _choose_stratum(self, rng, stratum, stratum_policy):
        if stratum is not None:
            if stratum not in self._stratum_index:
                raise ValueError(
                    f"unknown rock stratum {stratum!r}; expected one of {self.strata}"
                )
            return self._stratum_index[stratum]
        candidates = np.asarray([
            i for i, name in enumerate(self.strata)
            if name not in ("global", "unclassified")
        ], dtype=int)
        if candidates.size == 0:
            candidates = np.asarray([
                i for i, name in enumerate(self.strata) if name != "global"
            ], dtype=int)
        if stratum_policy == "balanced":
            weights = np.ones(candidates.size, dtype=float)
        elif stratum_policy == "corpus":
            weights = self.stratum_sample_count[candidates].astype(float)
        else:
            raise ValueError("stratum_policy must be 'balanced' or 'corpus'")
        weights /= weights.sum()
        return int(rng.choice(candidates, p=weights))

    @staticmethod
    def _interp_quantile(probability, knots, values):
        finite = np.isfinite(values)
        if not finite.any():
            return np.nan
        x = knots[finite]
        y = values[finite]
        if y.size == 1:
            return float(y[0])
        return float(np.interp(probability, x, y))

    def sample(
        self,
        seed,
        *,
        stratum: Optional[str] = None,
        stratum_policy: str = "balanced",
        min_nuclei_fraction: float = 1.0e-10,
        excluded_elements: Sequence[str] = DEFAULT_TRAINING_EXCLUSIONS,
    ) -> WholeRockComposition:
        """Draw one joint bulk-composition-shaped nuclei prior.

        ``min_nuclei_fraction`` is a numerical sparsification boundary, not a
        detection limit.  Exact-zero scenes remain a separate coverage
        augmentation.  The default is below the current 1e-8 training floor
        so weak geologic trace constituents can still enter a scene.
        """
        if (not np.isfinite(min_nuclei_fraction)
                or min_nuclei_fraction < 0
                or min_nuclei_fraction >= 1):
            raise ValueError("min_nuclei_fraction must lie in [0, 1)")
        unknown = [el for el in excluded_elements if el not in ATOMIC_NUMBER]
        if unknown:
            raise ValueError(f"unknown excluded elements: {unknown}")

        rng = np.random.default_rng(int(seed))
        si = self._choose_stratum(rng, stratum, stratum_policy)
        normal = self._copula_root[si] @ rng.standard_normal(N_ELEMENTS)
        uniform = np.clip(ndtr(normal), 1e-9, 1.0 - 1e-9)
        concentration_ppm = np.zeros(N_ELEMENTS, dtype=float)
        below_limit = np.zeros(N_ELEMENTS, dtype=bool)
        active_model = self.modeled_mask[si] & self.training_mask

        for ei in np.flatnonzero(active_model):
            u = float(uniform[ei])
            fraction = float(self.censored_fraction[si, ei])
            detected_q = self.detected_log10_ppm_quantile[si, ei]
            limit_q = self.limit_log10_ppm_quantile[si, ei]
            has_limit = np.isfinite(limit_q).any()
            if has_limit and fraction > 0 and u < fraction:
                q = np.clip(u / fraction, 0.0, 1.0)
                log_limit = self._interp_quantile(
                    q, self.quantile_probability, limit_q
                )
                # The database identifies the censor boundary, not a value
                # below it.  A bounded exponential tail preserves that fact
                # without placing all censored draws at limit/2.
                depth_dex = min(float(rng.exponential(0.5)), 3.0)
                log_ppm = log_limit - depth_dex
                below_limit[ei] = True
            else:
                if fraction < 1.0:
                    q = np.clip((u - fraction) / (1.0 - fraction), 0.0, 1.0)
                else:
                    q = u
                log_ppm = self._interp_quantile(
                    q, self.quantile_probability, detected_q
                )
                if not np.isfinite(log_ppm) and has_limit:
                    log_ppm = self._interp_quantile(
                        q, self.quantile_probability, limit_q
                    ) - min(float(rng.exponential(0.5)), 3.0)
                    below_limit[ei] = True
            if np.isfinite(log_ppm):
                concentration_ppm[ei] = 10.0 ** np.clip(log_ppm, -12.0, 7.0)

        for element in excluded_elements:
            concentration_ppm[ATOMIC_NUMBER[element] - 1] = 0.0
            below_limit[ATOMIC_NUMBER[element] - 1] = False
        if not np.any(concentration_ppm > 0):
            raise RuntimeError("whole-rock prior produced no modeled concentration")

        mass = concentration_ppm / concentration_ppm.sum()
        nuclei = mass / ATOMIC_MASS_U
        nuclei /= nuclei.sum()
        if min_nuclei_fraction > 0:
            small = nuclei < min_nuclei_fraction
            mass[small] = 0.0
            nuclei[small] = 0.0
            if not np.any(nuclei > 0):  # pragma: no cover - defensive
                raise RuntimeError("composition floor removed every element")
            mass /= mass.sum()
            nuclei = mass / ATOMIC_MASS_U
            nuclei /= nuclei.sum()
            below_limit[small] = False

        return WholeRockComposition(
            mass_fraction=mass,
            nuclei_fraction=nuclei,
            modeled_mask=active_model,
            below_detection_mask=below_limit,
            stratum=self.strata[si],
            seed=int(seed),
            source_doi=self.source_doi,
        )

    def sample_stage_abundance(
        self,
        seed,
        *,
        stage_alpha=(1.0, 1.0, 1.0),
        **composition_kwargs,
    ):
        """Draw composition and independent I--III fractions per element."""
        alpha = np.asarray(stage_alpha, dtype=float)
        if alpha.shape != (N_STAGES,) or np.any(~np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ValueError("stage_alpha must contain three positive values")
        composition = self.sample(seed, **composition_kwargs)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x57484F4C]))
        split = rng.dirichlet(alpha, N_ELEMENTS)
        abundance = composition.nuclei_fraction[:, None] * split
        abundance[~composition.modeled_mask] = 0.0
        abundance /= abundance.sum()
        return composition, abundance


class WholeRockCompositionMixture:
    """Equal-stratum mixture of separately versioned composition priors."""

    def __init__(self, models, labels=None):
        self.models = tuple(models)
        if not self.models:
            raise ValueError("at least one whole-rock model is required")
        if labels is None:
            labels = tuple(f"prior_{i}" for i in range(len(self.models)))
        self.labels = tuple(str(value) for value in labels)
        if len(self.labels) != len(self.models) or len(set(self.labels)) != len(self.labels):
            raise ValueError("whole-rock mixture labels must be unique and match models")
        reference_mask = np.asarray(self.models[0].training_mask, dtype=bool)
        if any(
            not np.array_equal(reference_mask, np.asarray(model.training_mask, dtype=bool))
            for model in self.models[1:]
        ):
            raise ValueError("whole-rock models must use the same training mask")
        self.training_mask = reference_mask.copy()
        self.modeled_mask = np.logical_or.reduce([
            np.asarray(model.modeled_mask[0], dtype=bool) for model in self.models
        ])
        dois = tuple(dict.fromkeys(model.source_doi for model in self.models))
        self.source_doi = "+".join(dois)
        self._choices = []
        for mi, (label, model) in enumerate(zip(self.labels, self.models)):
            for stratum in model.training_strata:
                si = model._stratum_index[stratum]
                self._choices.append((
                    mi,
                    stratum,
                    f"{label}:{stratum}",
                    int(model.stratum_sample_count[si]),
                ))
        if not self._choices:
            raise ValueError("whole-rock models have no classified training strata")
        self.strata = tuple(choice[2] for choice in self._choices)
        self.stratum_sample_count = np.asarray(
            [choice[3] for choice in self._choices], dtype=np.int64
        )
        self.manifest = {
            "schema": "alibz-whole-rock-mixture-v1",
            "labels": list(self.labels),
            "strata": list(self.strata),
            "default_policy": "balanced",
            "release_balance": "exact quota per stratum",
        }

    @classmethod
    def load_default(cls, root="db"):
        root = Path(root)
        return cls(
            (
                WholeRockCompositionModel.load(root / "whole_rock_prior_v1.npz"),
                WholeRockCompositionModel.load(
                    root / "whole_rock_carbonate_volatile_prior_v1.npz"
                ),
            ),
            labels=("anhydrous", "carbonate_volatile"),
        )

    @property
    def selectable_strata(self):
        return self.strata

    @property
    def training_strata(self):
        return self.strata

    def _choose(self, rng, stratum, stratum_policy):
        if stratum is not None:
            matches = [
                i for i, choice in enumerate(self._choices)
                if stratum in (choice[1], choice[2])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"stratum {stratum!r} is unknown or ambiguous; expected one of {self.strata}"
                )
            return matches[0]
        if stratum_policy == "balanced":
            weights = np.ones(len(self._choices), dtype=float)
        elif stratum_policy == "corpus":
            weights = self.stratum_sample_count.astype(float)
        else:
            raise ValueError("stratum_policy must be 'balanced' or 'corpus'")
        weights /= weights.sum()
        return int(rng.choice(len(self._choices), p=weights))

    def balanced_stratum_schedule(self, repeats_per_stratum, seed=0):
        """Return a replayable schedule with exactly equal stratum counts."""
        repeats = int(repeats_per_stratum)
        if repeats < 1:
            raise ValueError("repeats_per_stratum must be a positive integer")
        scheduled = np.repeat(np.asarray(self.strata), repeats)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(scheduled)
        return tuple(str(value) for value in scheduled.tolist())

    def sample(self, seed, *, stratum=None, stratum_policy="balanced", **kwargs):
        rng = np.random.default_rng(int(seed))
        choice_index = self._choose(rng, stratum, stratum_policy)
        model_index, source_stratum, qualified, _count = self._choices[choice_index]
        child_seed = int(np.random.SeedSequence([
            int(seed), choice_index, 0x4D495854,
        ]).generate_state(1, dtype=np.uint32)[0])
        value = self.models[model_index].sample(
            child_seed,
            stratum=source_stratum,
            stratum_policy=stratum_policy,
            **kwargs,
        )
        return WholeRockComposition(
            mass_fraction=value.mass_fraction,
            nuclei_fraction=value.nuclei_fraction,
            modeled_mask=value.modeled_mask,
            below_detection_mask=value.below_detection_mask,
            stratum=qualified,
            seed=int(seed),
            source_doi=value.source_doi,
        )

    def sample_stage_abundance(
        self,
        seed,
        *,
        stage_alpha=(1.0, 1.0, 1.0),
        **composition_kwargs,
    ):
        alpha = np.asarray(stage_alpha, dtype=float)
        if alpha.shape != (N_STAGES,) or np.any(~np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ValueError("stage_alpha must contain three positive values")
        composition = self.sample(seed, **composition_kwargs)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x57484F4C]))
        split = rng.dirichlet(alpha, N_ELEMENTS)
        abundance = composition.nuclei_fraction[:, None] * split
        abundance[~composition.modeled_mask] = 0.0
        abundance /= abundance.sum()
        return composition, abundance


__all__ = [
    "DEFAULT_TRAINING_EXCLUSIONS",
    "WHOLE_ROCK_MODEL_SCHEMA",
    "WholeRockComposition",
    "WholeRockCompositionMixture",
    "WholeRockCompositionModel",
]
