"""Quality measurements and reproducible candidate selection for notch filters.

The inverse-design result that matters physically is the spectrum predicted
from the selected structure, not only the spectrum decoded by the CVAE.  This
module therefore measures the final Tandem response and applies a
wavelength-first selection rule:

1. keep candidates whose final notch has a finite FWHM and a sufficiently low
   minimum transmittance;
2. keep candidates whose final resonance is within a wavelength tolerance;
3. when generated spectra are supplied, also require their resonances to stay
   near the requested wavelength;
4. among the jointly eligible candidates, first minimize the final-response
   wavelength error, then use valley depth and FWHM as secondary criteria.

If no candidate meets the requested notch-depth gate, the depth criterion is
relaxed only to the lowest observed minimum transmittance among candidates
with finite FWHM.  Likewise, if no candidate meets the requested wavelength
tolerance, that criterion is relaxed only to the best observed error plus one
wavelength-grid step.  Both fallbacks are exposed in the returned diagnostics
and never happen silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NotchMetrics:
    min_index: int
    min_wavelength_nm: float
    wavelength_error_nm: float
    absolute_wavelength_error_nm: float
    min_transmittance: float
    baseline_transmittance: float
    notch_depth: float
    half_depth_transmittance: float
    fwhm_nm: float
    has_finite_fwhm: bool


@dataclass(frozen=True)
class CandidateSelection:
    selected_index: int
    selection_mode: str
    wavelength_tolerance_nm: float
    effective_wavelength_limit_nm: float
    eligible_mask: np.ndarray
    metrics: tuple[NotchMetrics, ...]
    valley_scores: np.ndarray
    fwhm_scores: np.ndarray
    quality_scores: np.ndarray
    valley_reference_transmittance: float
    fwhm_reference_nm: float
    final_eligible_mask: np.ndarray
    generated_eligible_mask: np.ndarray | None
    generated_metrics: tuple[NotchMetrics, ...] | None
    generated_wavelength_tolerance_nm: float | None
    generated_grid_slack_nm: float
    generated_effective_wavelength_limit_nm: float | None
    maximum_final_min_transmittance: float
    effective_maximum_final_min_transmittance: float
    final_depth_gate_fallback_used: bool
    final_notch_valid_mask: np.ndarray
    ranking_rule: str


def _crossing_x(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    """Return a linearly interpolated x coordinate for a level crossing."""
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = (level - y0) / (y1 - y0)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(x0 + fraction * (x1 - x0))


def measure_notch(
    wavelengths_nm: Iterable[float],
    transmittance: Iterable[float],
    target_wavelength_nm: float,
    *,
    baseline_quantile: float = 0.95,
) -> NotchMetrics:
    """Measure the global transmission notch and its half-depth FWHM.

    The baseline is the 95th percentile by default, which is more robust than
    a single maximum point while retaining the usual half-depth definition for
    a transmission dip.  Crossing positions are linearly interpolated.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    spectrum = np.asarray(transmittance, dtype=float).reshape(-1)
    if wavelengths.size != spectrum.size or wavelengths.size < 3:
        raise ValueError(
            "wavelengths_nm and transmittance must have the same length >= 3"
        )
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("wavelengths_nm must be strictly increasing")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("transmittance contains NaN or infinite values")
    if not 0.5 < baseline_quantile <= 1.0:
        raise ValueError("baseline_quantile must be in (0.5, 1.0]")

    min_index = int(np.argmin(spectrum))
    min_wavelength = float(wavelengths[min_index])
    min_transmittance = float(spectrum[min_index])
    baseline = float(np.quantile(spectrum, baseline_quantile))
    notch_depth = max(0.0, baseline - min_transmittance)
    half_depth = min_transmittance + 0.5 * notch_depth

    left_candidates = np.flatnonzero(spectrum[: min_index + 1] >= half_depth)
    right_candidates = np.flatnonzero(spectrum[min_index:] >= half_depth)
    has_crossings = (
        notch_depth > np.finfo(float).eps
        and left_candidates.size > 0
        and right_candidates.size > 0
        and int(left_candidates[-1]) < min_index
        and int(right_candidates[0]) > 0
    )

    if has_crossings:
        left_upper = int(left_candidates[-1])
        left_lower = left_upper + 1
        right_upper = min_index + int(right_candidates[0])
        right_lower = right_upper - 1
        left_crossing = _crossing_x(
            wavelengths[left_upper],
            spectrum[left_upper],
            wavelengths[left_lower],
            spectrum[left_lower],
            half_depth,
        )
        right_crossing = _crossing_x(
            wavelengths[right_lower],
            spectrum[right_lower],
            wavelengths[right_upper],
            spectrum[right_upper],
            half_depth,
        )
        fwhm = max(0.0, right_crossing - left_crossing)
        finite_fwhm = bool(np.isfinite(fwhm) and fwhm > 0.0)
    else:
        fwhm = float("inf")
        finite_fwhm = False

    signed_error = min_wavelength - float(target_wavelength_nm)
    return NotchMetrics(
        min_index=min_index,
        min_wavelength_nm=min_wavelength,
        wavelength_error_nm=signed_error,
        absolute_wavelength_error_nm=abs(signed_error),
        min_transmittance=min_transmittance,
        baseline_transmittance=baseline,
        notch_depth=notch_depth,
        half_depth_transmittance=half_depth,
        fwhm_nm=float(fwhm),
        has_finite_fwhm=finite_fwhm,
    )


def select_single_notch_candidate(
    wavelengths_nm: Iterable[float],
    final_responses: np.ndarray,
    target_wavelength_nm: float,
    *,
    wavelength_tolerance_nm: float = 2.0,
    valley_weight: float = 0.5,
    fwhm_weight: float = 0.5,
    valley_reference_transmittance: float = 0.1,
    fwhm_reference_nm: float = 30.0,
    maximum_final_min_transmittance: float = 0.1,
    tie_breaker: Iterable[float] | None = None,
    generated_responses: np.ndarray | None = None,
    generated_wavelength_tolerance_nm: float | None = None,
    generated_grid_slack_nm: float = 0.0,
) -> CandidateSelection:
    """Select a final response using a depth gate and wavelength-first ranking.

    Valley quality uses ``abs(min_transmittance)`` because a physical optimum
    is zero; a negative neural-network excursion should not beat a small
    positive transmission merely because it is numerically lower.  Generated
    spectra are optional so existing callers that only have final responses
    remain supported.  The requested final-depth gate is relaxed only when no
    finite-FWHM candidate passes it, and that fallback is explicitly reported.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    responses = np.asarray(final_responses, dtype=float)
    if responses.ndim == 1:
        responses = responses.reshape(1, -1)
    if responses.ndim != 2 or responses.shape[1] != wavelengths.size:
        raise ValueError(
            "final_responses must have shape (n_candidates, n_wavelengths)"
        )
    if wavelength_tolerance_nm < 0:
        raise ValueError("wavelength_tolerance_nm must be non-negative")
    if valley_weight < 0 or fwhm_weight < 0:
        raise ValueError("quality weights must be non-negative")
    if valley_reference_transmittance <= 0:
        raise ValueError("valley_reference_transmittance must be positive")
    if fwhm_reference_nm <= 0:
        raise ValueError("fwhm_reference_nm must be positive")
    if maximum_final_min_transmittance < 0:
        raise ValueError(
            "maximum_final_min_transmittance must be non-negative"
        )
    if generated_grid_slack_nm < 0:
        raise ValueError("generated_grid_slack_nm must be non-negative")
    weight_sum = valley_weight + fwhm_weight
    if weight_sum <= 0:
        raise ValueError("at least one quality weight must be positive")
    valley_weight /= weight_sum
    fwhm_weight /= weight_sum

    metrics = tuple(
        measure_notch(wavelengths, response, target_wavelength_nm)
        for response in responses
    )
    wavelength_errors = np.asarray(
        [metric.absolute_wavelength_error_nm for metric in metrics],
        dtype=float,
    )
    finite_widths = np.asarray(
        [metric.has_finite_fwhm for metric in metrics],
        dtype=bool,
    )
    final_min_transmittances = np.asarray(
        [metric.min_transmittance for metric in metrics],
        dtype=float,
    )
    effective_maximum_final_min_transmittance = float(
        maximum_final_min_transmittance
    )
    final_notch_valid = (
        finite_widths
        & (
            final_min_transmittances
            <= effective_maximum_final_min_transmittance
        )
    )
    depth_fallback_used = False

    if not np.any(final_notch_valid):
        finite_width_indices = np.flatnonzero(finite_widths)
        if finite_width_indices.size == 0:
            raise ValueError("no candidate has a finite final-response FWHM")
        best_final_min_transmittance = float(
            np.min(final_min_transmittances[finite_width_indices])
        )
        effective_maximum_final_min_transmittance = (
            best_final_min_transmittance + 10 * np.finfo(float).eps
        )
        final_notch_valid = (
            finite_widths
            & (
                final_min_transmittances
                <= effective_maximum_final_min_transmittance
            )
        )
        depth_fallback_used = True

    final_eligible = (
        (wavelength_errors <= wavelength_tolerance_nm)
        & final_notch_valid
    )
    if depth_fallback_used:
        selection_mode = (
            "fallback_best_final_min_transmittance_"
            "within_requested_wavelength_tolerance"
        )
    else:
        selection_mode = "within_requested_wavelength_tolerance"
    effective_limit = float(wavelength_tolerance_nm)

    if not np.any(final_eligible):
        finite_error_indices = np.flatnonzero(
            final_notch_valid & np.isfinite(wavelength_errors)
        )
        if finite_error_indices.size == 0:
            raise ValueError(
                "no final-notch-valid candidate has a finite wavelength error"
            )
        grid_step = float(np.median(np.diff(wavelengths)))
        best_error = float(np.min(wavelength_errors[finite_error_indices]))
        effective_limit = best_error + grid_step + 10 * np.finfo(float).eps
        final_eligible = (
            (wavelength_errors <= effective_limit)
            & final_notch_valid
        )
        if depth_fallback_used:
            selection_mode = (
                "fallback_best_final_min_transmittance_and_"
                "best_wavelength_band"
            )
        else:
            selection_mode = "fallback_best_wavelength_band"

    generated_metrics: tuple[NotchMetrics, ...] | None = None
    generated_eligible: np.ndarray | None = None
    generated_requested_limit: float | None = None
    generated_effective_limit: float | None = None

    if generated_responses is not None:
        generated = np.asarray(generated_responses, dtype=float)
        if generated.ndim == 1:
            generated = generated.reshape(1, -1)
        if generated.shape != responses.shape:
            raise ValueError(
                "generated_responses must have the same shape as "
                "final_responses"
            )
        if generated_wavelength_tolerance_nm is None:
            generated_requested_limit = float(wavelength_tolerance_nm)
        else:
            generated_requested_limit = float(
                generated_wavelength_tolerance_nm
            )
        if generated_requested_limit < 0:
            raise ValueError(
                "generated_wavelength_tolerance_nm must be non-negative"
            )

        generated_metrics = tuple(
            measure_notch(wavelengths, response, target_wavelength_nm)
            for response in generated
        )
        generated_errors = np.asarray(
            [
                metric.absolute_wavelength_error_nm
                for metric in generated_metrics
            ],
            dtype=float,
        )
        generated_effective_limit = (
            generated_requested_limit + float(generated_grid_slack_nm)
        )
        generated_eligible = (
            generated_errors <= generated_effective_limit
        )
        eligible = final_eligible & generated_eligible

        if np.any(eligible):
            if selection_mode == "within_requested_wavelength_tolerance":
                selection_mode = (
                    "within_final_and_generated_wavelength_tolerances"
                )
            else:
                selection_mode += "_within_generated_wavelength_tolerance"
        else:
            # Preserve the final-response gate and relax only the generated-
            # spectrum gate to the best observed generated error band.  This
            # keeps selection possible while making the relaxation explicit.
            final_indices = np.flatnonzero(final_eligible)
            if final_indices.size == 0:
                raise ValueError("no candidate passes the final-response gate")
            finite_generated = final_indices[
                np.isfinite(generated_errors[final_indices])
            ]
            if finite_generated.size == 0:
                raise ValueError(
                    "no final-eligible candidate has a finite generated "
                    "wavelength error"
                )
            grid_step = float(np.median(np.diff(wavelengths)))
            best_generated_error = float(
                np.min(generated_errors[finite_generated])
            )
            generated_effective_limit = (
                best_generated_error
                + grid_step
                + 10 * np.finfo(float).eps
            )
            generated_eligible = (
                generated_errors <= generated_effective_limit
            )
            eligible = final_eligible & generated_eligible
            selection_mode += "_fallback_best_generated_wavelength_band"
    else:
        generated_errors = np.zeros(len(metrics), dtype=float)
        eligible = final_eligible.copy()

    valley_penalties = np.asarray(
        [abs(metric.min_transmittance) for metric in metrics],
        dtype=float,
    )
    fwhms = np.asarray([metric.fwhm_nm for metric in metrics], dtype=float)
    valley_scores = np.full(len(metrics), np.inf, dtype=float)
    fwhm_scores = np.full(len(metrics), np.inf, dtype=float)
    eligible_indices = np.flatnonzero(eligible)
    valley_scores[eligible_indices] = (
        valley_penalties[eligible_indices]
        / valley_reference_transmittance
    )
    fwhm_scores[eligible_indices] = (
        fwhms[eligible_indices] / fwhm_reference_nm
    )
    quality_scores = (
        valley_weight * valley_scores + fwhm_weight * fwhm_scores
    )

    if tie_breaker is None:
        tie_break_values = np.zeros(len(metrics), dtype=float)
    else:
        tie_break_values = np.asarray(tie_breaker, dtype=float).reshape(-1)
        if tie_break_values.size != len(metrics):
            raise ValueError("tie_breaker must contain one value per candidate")

    # Wavelength first, then the transmittance at the notch: the design
    # specification is where the notch sits, and among candidates that put it
    # in the same place the useful one is the deeper.  The width enters through
    # the finite-FWHM requirement rather than through the ordering, so that a
    # broader notch cannot outrank a deeper one at the same wavelength.  The
    # dual-notch selector applies the same order.
    selected_index = min(
        eligible_indices.tolist(),
        key=lambda index: (
            wavelength_errors[index],
            valley_penalties[index],
            generated_errors[index],
            tie_break_values[index],
            index,
        ),
    )
    return CandidateSelection(
        selected_index=int(selected_index),
        selection_mode=selection_mode,
        wavelength_tolerance_nm=float(wavelength_tolerance_nm),
        effective_wavelength_limit_nm=float(effective_limit),
        eligible_mask=eligible,
        metrics=metrics,
        valley_scores=valley_scores,
        fwhm_scores=fwhm_scores,
        quality_scores=quality_scores,
        valley_reference_transmittance=float(
            valley_reference_transmittance
        ),
        fwhm_reference_nm=float(fwhm_reference_nm),
        final_eligible_mask=final_eligible,
        generated_eligible_mask=generated_eligible,
        generated_metrics=generated_metrics,
        generated_wavelength_tolerance_nm=generated_requested_limit,
        generated_grid_slack_nm=float(generated_grid_slack_nm),
        generated_effective_wavelength_limit_nm=generated_effective_limit,
        maximum_final_min_transmittance=float(
            maximum_final_min_transmittance
        ),
        effective_maximum_final_min_transmittance=float(
            effective_maximum_final_min_transmittance
        ),
        final_depth_gate_fallback_used=depth_fallback_used,
        final_notch_valid_mask=final_notch_valid,
        ranking_rule=(
            "minimum_final_wavelength_error_then_final_depth_fwhm_quality_"
            "then_generated_wavelength_error_then_tie_breaker"
        ),
    )


def single_notch_verdict(
    wavelengths_nm,
    transmittance,
    ideal_transmittance,
    target_wavelength_nm,
    *,
    # 0.10, not the double-notch task's 0.35: this is the threshold
    # select_single_notch_candidate screens with, so the figures report
    # the same admissibility the designs were actually selected under.
    maximum_valley_transmittance: float = 0.10,
    minimum_notch_depth: float = 0.30,
    minimum_width_ratio: float = 0.50,
    maximum_width_ratio: float = 1.50,
):
    """Is this response a usable single notch, by the double-notch conditions?

    The manuscript states admissibility conditions only for the double-notch
    Screen. The single-notch route reports a bare ``argmin`` and checks
    neither depth nor width, so a response that lands on the target
    wavelength while being far too broad has looked like a success. This
    applies the three of the four double-notch conditions that mean anything
    for one resonance -- the depth-imbalance condition needs two.

    Returns ``(passed, failed_conditions, measurements)``; the caller reports
    the verdict beside the wavelength error rather than in place of it,
    because the inverse network returns a design either way.
    """
    metrics = measure_notch(wavelengths_nm, transmittance,
                            target_wavelength_nm)
    ideal_metrics = measure_notch(wavelengths_nm, ideal_transmittance,
                                  target_wavelength_nm)
    width_ratio = float('nan')
    if ideal_metrics.fwhm_nm > 0:
        width_ratio = metrics.fwhm_nm / ideal_metrics.fwhm_nm

    failed = []
    if not metrics.has_finite_fwhm:
        failed.append('width')
    if metrics.min_transmittance > maximum_valley_transmittance:
        failed.append('valley_depth')
    if metrics.notch_depth < minimum_notch_depth:
        failed.append('prominence')
    if not (minimum_width_ratio <= width_ratio <= maximum_width_ratio):
        failed.append('width_ratio')

    measurements = {
        'peak': float(metrics.min_wavelength_nm),
        'depth': float(metrics.min_transmittance),
        'notch_depth': float(metrics.notch_depth),
        'fwhm_nm': float(metrics.fwhm_nm),
        'ideal_fwhm_nm': float(ideal_metrics.fwhm_nm),
        'width_ratio': width_ratio,
        'error': float(metrics.absolute_wavelength_error_nm),
    }
    return not failed, failed, measurements
