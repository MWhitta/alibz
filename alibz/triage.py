"""Conservative, inexpensive physical evidence triage for candidate elements.

This module ranks wavelength-coincident candidates before the costly matrix
construction and optimisation stages.  Its scores are deliberately heuristic:
they are evidence summaries, not probabilities.  The sole rejection rule is a
strict, noise-backed missing-companion test; all other weak or ambiguous cases
remain available to downstream analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence
import weakref

import numpy as np

from alibz.utils.constants import BOLTZMANN


_COL_ION = 0
_COL_WAVELENGTH = 1
_COL_GA = 3
_COL_EI = 4
_COL_EK = 5
_UPPER_ID_COLS = (9, 10, 11)


@dataclass(frozen=True)
class TriageConfig:
    """Settings for :func:`triage_candidates`.

    The defaults favour retention.  In particular, ion balance and pair-ratio
    diagnostics never reject an element, and absent lines are ignored unless
    a separate local-noise model is supplied.
    """

    match_tolerance_nm: float = 0.10
    weak_match_tolerance_nm: float = 0.12
    unresolved_merge_nm: float = 0.10
    reversal_max_span_nm: float = 0.40
    self_absorption_ei_max_ev: float = 0.50
    max_ion_stage: int = 3
    strongest_per_stage: int = 20
    temperatures_k: tuple[float, ...] = (4000.0, 6500.0, 10000.0, 16000.0, 25000.0)
    ratio_local_span_nm: float = 20.0
    ratio_isolation_nm: float = 0.12
    ratio_tolerance_factor: float = 2.0
    upper_energy_tolerance_ev: float = 1e-5
    absence_sigma: float = 5.0
    min_missing_companions: int = 2
    # Enabling this is an explicit caller assertion that detector response is
    # bounded locally and the selected reference/companions are unsaturated,
    # optically thin, and free of unmodelled masks.  The engine cannot prove
    # those experimental conditions from a peak table, so the safe default is
    # diagnostic-only companion evidence.
    allow_companion_rejection: bool = False

    def __post_init__(self) -> None:
        positive = {
            "match_tolerance_nm": self.match_tolerance_nm,
            "weak_match_tolerance_nm": self.weak_match_tolerance_nm,
            "unresolved_merge_nm": self.unresolved_merge_nm,
            "reversal_max_span_nm": self.reversal_max_span_nm,
            "ratio_local_span_nm": self.ratio_local_span_nm,
            "ratio_isolation_nm": self.ratio_isolation_nm,
            "ratio_tolerance_factor": self.ratio_tolerance_factor,
            "upper_energy_tolerance_ev": self.upper_energy_tolerance_ev,
            "absence_sigma": self.absence_sigma,
        }
        if any(not np.isfinite(v) or v <= 0 for v in positive.values()):
            raise ValueError("triage tolerances and thresholds must be finite and positive")
        if self.self_absorption_ei_max_ev < 0:
            raise ValueError("self_absorption_ei_max_ev must be non-negative")
        if self.max_ion_stage < 1 or self.strongest_per_stage < 1:
            raise ValueError("max_ion_stage and strongest_per_stage must be positive")
        if self.min_missing_companions < 2:
            raise ValueError("min_missing_companions must be at least two")
        temps = np.asarray(self.temperatures_k, dtype=float)
        if temps.ndim != 1 or temps.size < 2 or np.any(~np.isfinite(temps)) or np.any(temps <= 0):
            raise ValueError("temperatures_k must contain at least two finite positive values")


@dataclass(frozen=True)
class RatioDiagnostic:
    wavelengths_nm: tuple[float, float]
    kind: str
    status: str
    observed_ratio: float | None = None
    expected_bounds: tuple[float, float] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SpeciesEvidence:
    element: str
    ion_stage: int
    candidate_feature_count: int
    matched_feature_count: int
    direct_feature_count: int
    reversal_feature_count: int
    matched_peak_indices: tuple[int, ...]
    strength_coverage: float
    ambiguity_weighted_support: float
    rules: tuple[str, ...] = ()
    ratio_diagnostics: tuple[RatioDiagnostic, ...] = ()


@dataclass(frozen=True)
class ElementEvidence:
    element: str
    decision: str
    score: float
    matched_feature_count: int
    matched_stage_count: int
    weak_wavelength_match_count: int
    observed_wavelength_match_count: int
    reversal_guard_count: int
    reasons: tuple[str, ...]
    species: tuple[SpeciesEvidence, ...] = ()
    proposed_missing_companions_nm: tuple[float, ...] = ()


@dataclass(frozen=True)
class TriageResult:
    keep_elements: tuple[str, ...]
    rejected_elements: tuple[str, ...]
    deferred_elements: tuple[str, ...]
    element_evidence: Mapping[str, ElementEvidence]
    species_evidence: tuple[SpeciesEvidence, ...]
    wavelength_range: tuple[float, float]
    coverage_intervals: tuple[tuple[float, float], ...]
    score_interpretation: str = "Uncalibrated evidence score; not a probability."
    rejection_scope: str = (
        "Rejection suggestions are conditional on the supplied peak list, coverage, "
        "and local-noise model."
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "keep_elements": list(self.keep_elements),
            "rejected_elements": list(self.rejected_elements),
            "deferred_elements": list(self.deferred_elements),
            "element_evidence": {
                key: _json_safe(asdict(value)) for key, value in self.element_evidence.items()
            },
            "species_evidence": [_json_safe(asdict(value)) for value in self.species_evidence],
            "wavelength_range": list(self.wavelength_range),
            "coverage_intervals": [list(interval) for interval in self.coverage_intervals],
            "score_interpretation": self.score_interpretation,
            "rejection_scope": self.rejection_scope,
        }


@dataclass
class _Feature:
    element: str
    ion: int
    wavelength: float
    wavelengths: np.ndarray
    gA: np.ndarray
    Ei: np.ndarray
    Ek: np.ndarray
    strengths: np.ndarray
    upper_ids: tuple[tuple[str, str, str], ...]
    selected: bool = False
    covered: bool = False
    direct_peak: int | None = None
    reversal_peaks: tuple[int, int] | None = None
    ambiguity: float = 1.0
    isolated: bool = False

    @property
    def peak_indices(self) -> tuple[int, ...]:
        if self.direct_peak is not None:
            return (self.direct_peak,)
        return self.reversal_peaks or ()

_DB_CACHE: "weakref.WeakKeyDictionary[Any, dict[str, tuple[np.ndarray, np.ndarray]]]" = weakref.WeakKeyDictionary()
_OBSERVED_CACHE: "weakref.WeakKeyDictionary[Any, dict[str, np.ndarray]]" = weakref.WeakKeyDictionary()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _database_lines(db: Any, element: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        cache = _DB_CACHE.setdefault(db, {})
    except TypeError:  # A deliberately unhashable test/user database.
        cache = {}
    cached = cache.get(element)
    if cached is not None:
        return cached
    raw = np.asarray(db.lines(element))
    if raw.size == 0:
        result = (np.empty((0, 5), dtype=float), np.empty((0, 3), dtype=str))
    else:
        raw = np.atleast_2d(raw)
        if raw.shape[1] < 14:
            raise ValueError(f"database lines for {element} must have at least 14 columns")
        try:
            numeric = raw[:, [_COL_ION, _COL_WAVELENGTH, _COL_GA, _COL_EI, _COL_EK]].astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"database lines for {element} contain non-numeric physical fields") from exc
        upper = raw[:, _UPPER_ID_COLS].astype(str)
        valid = np.all(np.isfinite(numeric), axis=1) & (numeric[:, 1] > 0) & (numeric[:, 2] > 0)
        result = (numeric[valid], upper[valid])
    cache[element] = result
    return result


def _normalise_intervals(
    wavelength_range: Sequence[float] | None,
    coverage_intervals: Sequence[Sequence[float]] | None,
    peak_wavelengths: np.ndarray,
) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
    if wavelength_range is None:
        if peak_wavelengths.size == 0:
            raise ValueError("wavelength_range is required when peak_array is empty")
        wl_range = (float(np.min(peak_wavelengths)), float(np.max(peak_wavelengths)))
    else:
        if len(wavelength_range) != 2:
            raise ValueError("wavelength_range must contain exactly two values")
        wl_range = tuple(float(v) for v in wavelength_range)
    if not np.all(np.isfinite(wl_range)) or wl_range[0] >= wl_range[1]:
        raise ValueError("wavelength_range must be finite and increasing")

    raw_intervals = (wl_range,) if coverage_intervals is None else coverage_intervals
    intervals: list[tuple[float, float]] = []
    for interval in raw_intervals:
        if len(interval) != 2:
            raise ValueError("each coverage interval must contain exactly two values")
        lo, hi = (float(v) for v in interval)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("coverage intervals must be finite and increasing")
        lo, hi = max(lo, wl_range[0]), min(hi, wl_range[1])
        if lo < hi:
            intervals.append((lo, hi))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError("coverage intervals must not overlap")
    return wl_range, tuple(intervals)


def _is_covered(wavelengths: np.ndarray | float, intervals: tuple[tuple[float, float], ...]) -> np.ndarray:
    values = np.asarray(wavelengths, dtype=float)
    covered = np.zeros(values.shape, dtype=bool)
    for lo, hi in intervals:
        covered |= (values >= lo) & (values <= hi)
    return covered


def _make_noise_function(local_noise: Any) -> Callable[[np.ndarray], np.ndarray] | None:
    if local_noise is None:
        return None
    if callable(local_noise):
        def evaluate(values: np.ndarray) -> np.ndarray:
            out = np.asarray(local_noise(values), dtype=float)
            if out.ndim == 0:
                out = np.full(values.shape, float(out))
            try:
                out = np.broadcast_to(out, values.shape).astype(float, copy=False)
            except ValueError as exc:
                raise ValueError("local_noise callable returned an incompatible shape") from exc
            return out
        return evaluate
    if not isinstance(local_noise, (tuple, list)) or len(local_noise) != 2:
        raise ValueError("local_noise must be callable or (wavelengths, area_sigma)")
    x = np.asarray(local_noise[0], dtype=float).reshape(-1)
    y = np.asarray(local_noise[1], dtype=float).reshape(-1)
    if x.size < 2 or x.shape != y.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("local_noise arrays must be equal-length finite vectors with at least two points")
    if np.any(np.diff(x) <= 0) or np.any(y <= 0):
        raise ValueError("local_noise wavelengths must increase and uncertainties must be positive")
    return lambda values: np.interp(values, x, y, left=np.nan, right=np.nan)


def _group_stage(
    element: str,
    ion: int,
    numeric: np.ndarray,
    upper: np.ndarray,
    temperatures: np.ndarray,
    config: TriageConfig,
) -> list[_Feature]:
    order = np.argsort(numeric[:, 1])
    numeric, upper = numeric[order], upper[order]
    groups: list[np.ndarray] = []
    start = 0
    for idx in range(1, len(numeric)):
        # A dense forest must not chain across many resolution elements merely
        # because each adjacent gap is small.  Every member must remain within
        # one merge width of the first line in the group.
        if numeric[idx, 1] - numeric[start, 1] > config.unresolved_merge_nm:
            groups.append(np.arange(start, idx))
            start = idx
    groups.append(np.arange(start, len(numeric)))

    features: list[_Feature] = []
    kT = BOLTZMANN * temperatures[:, None]
    for indices in groups:
        rows = numeric[indices]
        line_strength = (
            rows[None, :, 2] / rows[None, :, 1]
            * np.exp(-rows[None, :, 4] / kT)
        )
        strengths = np.sum(line_strength, axis=1)
        representative_weights = np.max(line_strength, axis=0)
        wavelength = float(np.average(rows[:, 1], weights=representative_weights))
        identities = tuple(tuple(v.strip().strip('"') for v in row) for row in upper[indices])
        features.append(_Feature(
            element=element,
            ion=ion,
            wavelength=wavelength,
            wavelengths=rows[:, 1].copy(),
            gA=rows[:, 2].copy(),
            Ei=rows[:, 3].copy(),
            Ek=rows[:, 4].copy(),
            strengths=strengths,
            upper_ids=identities,
        ))
    if not features:
        return features
    strength_matrix = np.column_stack([feature.strengths for feature in features])
    selected: set[int] = set()
    count = min(config.strongest_per_stage, len(features))
    for row in strength_matrix:
        selected.update(np.argpartition(row, -count)[-count:].tolist())
    for idx in selected:
        features[idx].selected = True
    return features


def _nearest_peak(wavelength: float, sorted_wavelengths: np.ndarray) -> tuple[int | None, float]:
    if sorted_wavelengths.size == 0:
        return None, np.inf
    pos = int(np.searchsorted(sorted_wavelengths, wavelength))
    choices = [idx for idx in (pos - 1, pos) if 0 <= idx < len(sorted_wavelengths)]
    idx = min(choices, key=lambda item: abs(sorted_wavelengths[item] - wavelength))
    return idx, float(abs(sorted_wavelengths[idx] - wavelength))


def _match_features(
    features: list[_Feature],
    sorted_peak_wl: np.ndarray,
    sorted_to_original: np.ndarray,
    intervals: tuple[tuple[float, float], ...],
    config: TriageConfig,
) -> None:
    for feature in features:
        feature.covered = bool(_is_covered(feature.wavelength, intervals))
        if not feature.selected or not feature.covered:
            continue
        peak_idx, distance = _nearest_peak(feature.wavelength, sorted_peak_wl)
        if peak_idx is not None and distance <= config.match_tolerance_nm:
            feature.direct_peak = int(sorted_to_original[peak_idx])
            continue
        if np.min(feature.Ei) > config.self_absorption_ei_max_ev or len(sorted_peak_wl) < 2:
            continue
        pos = int(np.searchsorted(sorted_peak_wl, feature.wavelength))
        if 0 < pos < len(sorted_peak_wl):
            left, right = pos - 1, pos
            span = sorted_peak_wl[right] - sorted_peak_wl[left]
            midpoint = 0.5 * (sorted_peak_wl[right] + sorted_peak_wl[left])
            if span <= config.reversal_max_span_nm and abs(midpoint - feature.wavelength) <= config.match_tolerance_nm:
                feature.reversal_peaks = (int(sorted_to_original[left]), int(sorted_to_original[right]))

    # One fitted feature is one item of evidence per species, even if several
    # unresolved database groups happen to land on it.
    by_species: dict[tuple[str, int], list[_Feature]] = {}
    for feature in features:
        if feature.peak_indices:
            by_species.setdefault((feature.element, feature.ion), []).append(feature)
    for species_features in by_species.values():
        claimed: set[int] = set()
        for feature in sorted(species_features, key=lambda f: float(np.max(f.strengths)), reverse=True):
            peaks = set(feature.peak_indices)
            if peaks & claimed:
                feature.direct_peak = None
                feature.reversal_peaks = None
            else:
                claimed.update(peaks)


def _mark_ambiguity_and_isolation(features: list[_Feature], config: TriageConfig) -> None:
    claims: dict[int, set[str]] = {}
    for feature in features:
        for peak in feature.peak_indices:
            claims.setdefault(peak, set()).add(feature.element)
    selected = [feature for feature in features if feature.selected and feature.covered]
    wavelengths = np.array([feature.wavelength for feature in selected], dtype=float)
    for feature in selected:
        if feature.peak_indices:
            claim_count = max(len(claims[peak]) for peak in feature.peak_indices)
            feature.ambiguity = 1.0 / claim_count
        neighbours = np.abs(wavelengths - feature.wavelength) <= config.ratio_isolation_nm
        feature.isolated = bool(np.sum(neighbours) == 1 and len(feature.wavelengths) == 1)


def _line_matches(numeric: np.ndarray, sorted_peak_wl: np.ndarray, tolerance: float) -> int:
    if numeric.size == 0 or sorted_peak_wl.size == 0:
        return 0
    wavelengths = np.sort(numeric[:, 1])
    positions = np.searchsorted(sorted_peak_wl, wavelengths)
    left = np.clip(positions - 1, 0, len(sorted_peak_wl) - 1)
    right = np.clip(positions, 0, len(sorted_peak_wl) - 1)
    distances = np.minimum(abs(wavelengths - sorted_peak_wl[left]), abs(wavelengths - sorted_peak_wl[right]))
    return int(np.sum(distances <= tolerance))


def _line_reversal_matches(
    numeric: np.ndarray,
    sorted_peak_wl: np.ndarray,
    config: TriageConfig,
) -> int:
    """Count all low-Ei database lines guarded by a bracketing peak pair.

    This deliberately examines the full covered stage-limited line list, not
    just the top-strength features, so a self-reversed weak line cannot be
    mistaken for zero wavelength support.
    """
    if numeric.size == 0 or len(sorted_peak_wl) < 2:
        return 0
    count = 0
    for wavelength in numeric[numeric[:, 3] <= config.self_absorption_ei_max_ev, 1]:
        pos = int(np.searchsorted(sorted_peak_wl, wavelength))
        if 0 < pos < len(sorted_peak_wl):
            left, right = sorted_peak_wl[pos - 1], sorted_peak_wl[pos]
            if (
                right - left <= config.reversal_max_span_nm
                and abs(0.5 * (left + right) - wavelength) <= config.match_tolerance_nm
            ):
                count += 1
    return count


def _observed_matches(
    db: Any,
    element: str,
    sorted_peak_wl: np.ndarray,
    intervals: tuple[tuple[float, float], ...],
    config: TriageConfig,
) -> int:
    """Count wavelength-only catalog matches without inventing physical data."""
    observed_method = getattr(db, "observed_lines", None)
    if not callable(observed_method) or sorted_peak_wl.size == 0:
        return 0
    try:
        cache = _OBSERVED_CACHE.setdefault(db, {})
    except TypeError:
        cache = {}
    numeric = cache.get(element)
    if numeric is None:
        rows = observed_method(element)
        parsed: list[tuple[float, float]] = []
        for row in rows:
            try:
                ion = float(row["ion_stage"])
                wavelength = float(row["wavelength_nm"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(ion) and np.isfinite(wavelength) and wavelength > 0:
                parsed.append((ion, wavelength))
        numeric = np.asarray(parsed, dtype=float).reshape((-1, 2))
        cache[element] = numeric
    if numeric.size == 0:
        return 0
    usable = (
        (numeric[:, 0] >= 1)
        & (numeric[:, 0] <= config.max_ion_stage)
        & _is_covered(numeric[:, 1], intervals)
    )
    if not np.any(usable):
        return 0
    pseudo_lines = np.column_stack((
        numeric[usable, 0], numeric[usable, 1],
        np.ones((int(np.sum(usable)), 3), dtype=float),
    ))
    return _line_matches(pseudo_lines, sorted_peak_wl, config.weak_match_tolerance_nm)


def _shared_upper(feature: _Feature) -> tuple[str, str, str] | None:
    if len(feature.upper_ids) != 1:
        return None
    identity = feature.upper_ids[0]
    if any(not value or value.lower() in {"nan", "none", "?"} for value in identity):
        return None
    return identity


def _ratio_diagnostics(
    species_features: list[_Feature],
    peak_areas: np.ndarray,
    config: TriageConfig,
    strength_uncertain: bool,
) -> tuple[RatioDiagnostic, ...]:
    if strength_uncertain:
        return ()
    diagnostics: list[RatioDiagnostic] = []
    matched = [feature for feature in species_features if feature.direct_peak is not None]
    for left_index, left in enumerate(matched):
        for right in matched[left_index + 1:]:
            if abs(left.wavelength - right.wavelength) > config.ratio_local_span_nm:
                continue
            identity = _shared_upper(left)
            shared_upper = bool(
                identity is not None
                and identity == _shared_upper(right)
                and abs(float(left.Ek[0]) - float(right.Ek[0])) <= config.upper_energy_tolerance_ev
            )
            kind = "shared_upper_branching" if shared_upper else "local_temperature_envelope"
            wavelengths = (left.wavelength, right.wavelength)
            if (
                np.min(left.Ei) <= config.self_absorption_ei_max_ev
                or np.min(right.Ei) <= config.self_absorption_ei_max_ev
                or not left.isolated
                or not right.isolated
            ):
                diagnostics.append(RatioDiagnostic(
                    wavelengths_nm=wavelengths,
                    kind=kind,
                    status="abstain",
                    reason="low_excitation_or_blended",
                ))
                continue
            if peak_areas[left.direct_peak] <= 0 or peak_areas[right.direct_peak] <= 0:
                diagnostics.append(RatioDiagnostic(
                    wavelengths_nm=wavelengths,
                    kind=kind,
                    status="abstain",
                    reason="nonpositive_fitted_area",
                ))
                continue
            observed = peak_areas[left.direct_peak] / peak_areas[right.direct_peak]
            if shared_upper:
                expected = (left.gA[0] / left.wavelength) / (right.gA[0] / right.wavelength)
                lo = expected / config.ratio_tolerance_factor
                hi = expected * config.ratio_tolerance_factor
            else:
                envelope = left.strengths / np.maximum(right.strengths, np.finfo(float).tiny)
                lo = float(np.min(envelope)) / config.ratio_tolerance_factor
                hi = float(np.max(envelope)) * config.ratio_tolerance_factor
            diagnostics.append(RatioDiagnostic(
                wavelengths_nm=wavelengths,
                kind=kind,
                status="consistent" if lo <= observed <= hi else "inconsistent",
                observed_ratio=float(observed),
                expected_bounds=(float(lo), float(hi)),
                reason="diagnostic_only",
            ))
    return tuple(diagnostics)


def _missing_companions(
    element: str,
    features: list[_Feature],
    peak_areas: np.ndarray,
    peak_sigma: np.ndarray | None,
    noise_function: Callable[[np.ndarray], np.ndarray] | None,
    uncertain_strengths: set[str],
    config: TriageConfig,
) -> tuple[float, ...]:
    if peak_sigma is None or noise_function is None or element in uncertain_strengths:
        return ()
    element_features = [f for f in features if f.element == element and f.selected and f.covered]
    references = [
        f for f in element_features
        if f.direct_peak is not None
        and f.isolated
        and np.min(f.Ei) > config.self_absorption_ei_max_ev
        and peak_sigma[f.direct_peak] > 0
        and peak_areas[f.direct_peak] >= config.absence_sigma * peak_sigma[f.direct_peak]
    ]
    best_missing: tuple[float, ...] = ()
    for reference in references:
        companions: list[float] = []
        for candidate in element_features:
            if candidate is reference or candidate.peak_indices:
                continue
            if (
                candidate.ion != reference.ion
                or np.min(candidate.Ei) <= config.self_absorption_ei_max_ev
                or not candidate.isolated
            ):
                continue
            if abs(candidate.wavelength - reference.wavelength) > config.ratio_local_span_nm:
                continue
            noise = noise_function(np.array([candidate.wavelength], dtype=float))
            if noise.shape != (1,) or not np.isfinite(noise[0]) or noise[0] <= 0:
                continue
            ratios = candidate.strengths / np.maximum(reference.strengths, np.finfo(float).tiny)
            predicted = peak_areas[reference.direct_peak] * ratios
            if np.all(predicted >= config.absence_sigma * noise[0]):
                companions.append(candidate.wavelength)
        if len(companions) > len(best_missing):
            best_missing = tuple(sorted(companions))
    if len(best_missing) < config.min_missing_companions:
        return ()
    return best_missing


def triage_candidates(
    peak_array: Any,
    db: Any,
    *,
    config: TriageConfig | None = None,
    wavelength_range: Sequence[float] | None = None,
    coverage_intervals: Sequence[Sequence[float]] | None = None,
    peak_sigma: Any | None = None,
    protected_elements: Sequence[str] = (),
    local_noise: Any | None = None,
) -> TriageResult:
    """Rank physical element evidence and propose only high-confidence pruning.

    ``peak_array`` rows are ``[integrated_area, database-frame_nm, sigma_nm,
    gamma_nm, ...]``.  ``peak_sigma`` describes uncertainty of fitted peak
    areas; it is never treated as the noise at wavelengths where no peak was
    fitted.  Supply ``local_noise`` separately to enable missing-line logic.
    """
    config = TriageConfig() if config is None else config
    if not isinstance(config, TriageConfig):
        raise TypeError("config must be a TriageConfig")
    peaks = np.asarray(peak_array, dtype=float)
    if peaks.ndim != 2 or peaks.shape[1] < 4:
        raise ValueError("peak_array must have shape (n, >=4)")
    if np.any(~np.isfinite(peaks[:, :4])):
        raise ValueError("peak_array physical columns must be finite")
    if np.any(peaks[:, 0] < 0) or np.any(peaks[:, 1] <= 0) or np.any(peaks[:, 2:4] < 0):
        raise ValueError("peak areas/widths must be non-negative and wavelengths positive")
    if peak_sigma is not None:
        peak_sigma = np.asarray(peak_sigma, dtype=float).reshape(-1)
        if peak_sigma.shape != (len(peaks),) or np.any(~np.isfinite(peak_sigma)) or np.any(peak_sigma <= 0):
            raise ValueError("peak_sigma must be a finite positive vector with one value per peak")

    elements = tuple(str(value) for value in getattr(db, "elements", ()))
    if not elements:
        raise ValueError("db.elements must contain at least one element")
    protected = tuple(dict.fromkeys(str(value) for value in protected_elements))
    unknown = set(protected) - set(elements)
    if unknown:
        raise ValueError(f"protected_elements contains unknown elements: {sorted(unknown)}")
    wl_range, intervals = _normalise_intervals(wavelength_range, coverage_intervals, peaks[:, 1])
    noise_function = _make_noise_function(local_noise)

    covered_peak_indices = np.flatnonzero(_is_covered(peaks[:, 1], intervals))
    order = covered_peak_indices[np.argsort(peaks[covered_peak_indices, 1], kind="stable")]
    sorted_peak_wl = peaks[order, 1]
    no_lines = set(getattr(db, "no_lines", ()))
    excluded = set(getattr(db, "analysis_excluded_elements", ()))
    uncertain_strengths = set(getattr(db, "strength_uncertain_elements", {"Se", "Th", "U"}))
    temperatures = np.asarray(config.temperatures_k, dtype=float)
    all_features: list[_Feature] = []
    numeric_by_element: dict[str, np.ndarray] = {}
    unavailable: set[str] = set()

    for element in elements:
        if element in no_lines or element in excluded:
            unavailable.add(element)
            continue
        numeric, upper = _database_lines(db, element)
        numeric_by_element[element] = numeric
        if numeric.size == 0:
            unavailable.add(element)
            continue
        stages = numeric[:, 0].astype(int)
        for ion in sorted(set(stages)):
            if ion < 1 or ion > config.max_ion_stage:
                continue
            mask = (stages == ion) & _is_covered(numeric[:, 1], intervals)
            if not np.any(mask):
                continue
            all_features.extend(_group_stage(
                element, ion, numeric[mask], upper[mask], temperatures, config
            ))

    _match_features(all_features, sorted_peak_wl, order, intervals, config)
    _mark_ambiguity_and_isolation(all_features, config)
    feature_by_species: dict[tuple[str, int], list[_Feature]] = {}
    for feature in all_features:
        feature_by_species.setdefault((feature.element, feature.ion), []).append(feature)

    species_evidence: list[SpeciesEvidence] = []
    species_by_element: dict[str, list[SpeciesEvidence]] = {}
    for (element, ion), features in feature_by_species.items():
        candidates = [feature for feature in features if feature.selected and feature.covered]
        matched = [feature for feature in candidates if feature.peak_indices]
        if candidates:
            strength_matrix = np.column_stack([feature.strengths for feature in candidates])
            matched_mask = np.array([bool(feature.peak_indices) for feature in candidates])
            total = np.sum(strength_matrix, axis=1)
            covered_strength = np.sum(strength_matrix[:, matched_mask], axis=1) if np.any(matched_mask) else np.zeros_like(total)
            strength_coverage = float(np.max(np.divide(
                covered_strength, total, out=np.zeros_like(total), where=total > 0
            )))
        else:
            strength_coverage = 0.0
        rules: list[str] = []
        reversal_count = sum(feature.reversal_peaks is not None for feature in matched)
        if reversal_count:
            rules.append("possible_self_reversal_counted_as_one_tentative_feature")
        ratio = _ratio_diagnostics(
            features, peaks[:, 0], config, element in uncertain_strengths
        )
        if ratio:
            rules.append("pair_ratio_is_diagnostic_only")
        evidence = SpeciesEvidence(
            element=element,
            ion_stage=ion,
            candidate_feature_count=len(candidates),
            matched_feature_count=len(matched),
            direct_feature_count=sum(feature.direct_peak is not None for feature in matched),
            reversal_feature_count=reversal_count,
            matched_peak_indices=tuple(sorted({peak for feature in matched for peak in feature.peak_indices})),
            strength_coverage=strength_coverage,
            ambiguity_weighted_support=float(sum(feature.ambiguity for feature in matched)),
            rules=tuple(rules),
            ratio_diagnostics=ratio,
        )
        species_evidence.append(evidence)
        species_by_element.setdefault(element, []).append(evidence)

    element_evidence: dict[str, ElementEvidence] = {}
    keep: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    for element in elements:
        species = tuple(species_by_element.get(element, ()))
        matched_count = sum(item.matched_feature_count for item in species)
        matched_stages = sum(item.matched_feature_count > 0 for item in species)
        weighted = sum(item.ambiguity_weighted_support for item in species)
        coverage = max((item.strength_coverage for item in species), default=0.0)
        score = float(weighted + 2.0 * coverage + 0.5 * max(0, matched_stages - 1))
        numeric = numeric_by_element.get(element, np.empty((0, 5), dtype=float))
        in_range = numeric.size > 0 and bool(np.any(
            (numeric[:, 1] >= wl_range[0]) & (numeric[:, 1] <= wl_range[1])
        ))
        if numeric.size:
            supported_stage = (
                (numeric[:, 0] >= 1)
                & (numeric[:, 0] <= config.max_ion_stage)
                & _is_covered(numeric[:, 1], intervals)
            )
            covered_lines = numeric[supported_stage]
        else:
            covered_lines = numeric
        weak_count = _line_matches(covered_lines, sorted_peak_wl, config.weak_match_tolerance_nm)
        reversal_guard_count = _line_reversal_matches(covered_lines, sorted_peak_wl, config)
        observed_count = 0
        if matched_count == 0 and weak_count == 0 and element not in excluded:
            observed_count = _observed_matches(
                db, element, sorted_peak_wl, intervals, config
            )
        reasons: list[str] = []
        missing: tuple[float, ...] = ()

        if element in protected:
            decision = "keep"
            reasons.append("protected")
        elif observed_count:
            decision = "keep"
            reasons.append("observed_catalog_wavelength_support_nonquantitative")
        elif element in unavailable:
            decision = "unavailable"
            reasons.append("analysis_excluded" if element in excluded else "unavailable_database")
        elif matched_count:
            decision = "keep"
            reasons.append("strong_wavelength_support")
            if matched_stages > 1:
                reasons.append("multiple_ion_stages")
            matched_ions = [item.ion_stage for item in species if item.matched_feature_count]
            if matched_ions and min(matched_ions) >= 3:
                reasons.append("high_ion_stage_without_lower_support_flag_only")
            missing = _missing_companions(
                element, all_features, peaks[:, 0], peak_sigma, noise_function,
                uncertain_strengths, config,
            )
            if missing and config.allow_companion_rejection:
                decision = "reject"
                reasons.append("noise_backed_missing_stronger_companions")
            elif missing:
                reasons.append("missing_companion_diagnostic_only")
        elif weak_count:
            decision = "keep"
            reasons.append("weak_wavelength_support_low_priority")
        elif reversal_guard_count:
            decision = "keep"
            reasons.append("possible_self_reversal_guard")
        else:
            decision = "defer"
            if not in_range:
                reasons.append("out_of_range")
            elif not intervals:
                reasons.append("no_covered_wavelengths")
            else:
                reasons.append("no_wavelength_support")
        if decision == "reject":
            rejected.append(element)
        elif decision != "unavailable" or element in protected:
            keep.append(element)
        if decision == "defer":
            deferred.append(element)
        element_evidence[element] = ElementEvidence(
            element=element,
            decision=decision,
            score=score,
            matched_feature_count=matched_count,
            matched_stage_count=matched_stages,
            weak_wavelength_match_count=weak_count,
            observed_wavelength_match_count=observed_count,
            reversal_guard_count=reversal_guard_count,
            reasons=tuple(reasons),
            species=species,
            proposed_missing_companions_nm=missing,
        )

    return TriageResult(
        keep_elements=tuple(keep),
        rejected_elements=tuple(rejected),
        deferred_elements=tuple(deferred),
        element_evidence=element_evidence,
        species_evidence=tuple(species_evidence),
        wavelength_range=wl_range,
        coverage_intervals=intervals,
    )


__all__ = [
    "ElementEvidence",
    "RatioDiagnostic",
    "SpeciesEvidence",
    "TriageConfig",
    "TriageResult",
    "triage_candidates",
]
