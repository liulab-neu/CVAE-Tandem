import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, peak_prominences, peak_widths

import CVAE_2p


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cstdata_10496', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
FNN_CHECKPOINT = os.path.join(
    MODEL_DIR,
    'DNN_tandem_FNN_label.ckpt',
)
INN_CHECKPOINT = os.path.join(
    MODEL_DIR,
    'DNN_tandem_INN_label.ckpt',
)
CVAE_CHECKPOINT = os.path.join(MODEL_DIR, 'cvae_best.pth')

def load_structure_bounds():
    """Normalisation bounds for the checkpoints in MODEL_DIR.

    train_networks_2p.py derives these from its training set and writes them
    to model/normalisation.json, so a retrain on a wider n axis carries its
    own bounds instead of needing an edit here. Checkpoints from before that
    file existed keep the original hard-coded pair.
    """
    path = os.path.join(MODEL_DIR, 'normalisation.json')
    # The current dataset's bounds. Every model directory now ships a
    # sidecar, so this is only reached for a directory that predates
    # them -- and 3.0 here was the silent-wrong-answer trap.
    minimum = [2.5, 1.5, 1.5]
    maximum = [6.5, 3.05, 3.05]
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as handle:
            saved = json.load(handle)
        minimum = saved.get('structure_min', minimum)
        maximum = saved.get('structure_max', maximum)
    return (
        np.asarray(minimum, dtype=np.float32),
        np.asarray(maximum, dtype=np.float32),
    )


STRUCTURE_MIN, STRUCTURE_MAX = load_structure_bounds()


def load_deployed_norm():
    path = os.path.join(MODEL_DIR, 'normalisation.json')
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle).get('norm', 'bn')
    return 'bn'


DEPLOYED_NORM = load_deployed_norm()
FNN_SIZE = [3, 64, 128, 256, 512, 1024, 2048, 1024, 1001]
INN_SIZE = [1001, 1024, 2048, 1024, 512, 256, 128, 64, 3]
ENCODER_SIZE = [1001, 1024, 2048, 1024, 512, 256, 128, 20]
DECODER_SIZE = [20, 128, 256, 512, 1024, 2048, 1024, 1001]


@dataclass
class PeakDetection:
    valid: bool
    indices: np.ndarray
    wavelengths: np.ndarray
    values: np.ndarray
    all_indices: np.ndarray


@dataclass
class ResonanceQuality:
    valid: bool
    prominences: np.ndarray
    fwhm_nm: np.ndarray
    quality_factors: np.ndarray
    contrast_quality_fom: np.ndarray
    worst_valley_transmittance: float
    widest_fwhm_nm: float
    minimum_prominence: float
    minimum_contrast_quality_fom: float


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wavelength_grid():
    # The grid is identical for every sample, so read it from whichever
    # dataset layout is present: the archived text export, a bundle packed by
    # pack_dataset.py, or a per-sample .npz from gen_2peak_dataset.py.
    first_spectrum = os.path.join(DATA_DIR, '00001.b')
    if not os.path.isfile(first_spectrum):
        for candidate in (
            DATA_DIR.rstrip(os.sep).rsplit(os.sep, 1)[0] + '.npz',
            os.path.join(DATA_DIR, '00001.npz'),
        ):
            if os.path.isfile(candidate):
                with np.load(candidate) as bundle:
                    return bundle['wavelength_nm'].astype(float)
        raise FileNotFoundError(
            f'Missing wavelength-grid source: {first_spectrum}'
        )
    wavelengths = np.loadtxt(first_spectrum, usecols=0)
    if wavelengths.shape != (1001,):
        raise ValueError(
            f'Expected 1001 wavelengths, found {wavelengths.shape}.'
        )
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError('The training wavelength grid is not increasing.')
    return wavelengths.astype(float)


# Must match tandem_2p.build_dense, which uses BatchNorm1d.
def build_dense(sizes, norm=None):
    """Must stay in step with tandem_2p.build_dense, which trains the weights.

    The normalisation kind comes from the deployed checkpoints' sidecar, so a
    GroupNorm model and a BatchNorm model can both be loaded without the
    caller having to know which is on disk.
    """
    if norm is None:
        norm = DEPLOYED_NORM
    layers = []
    for index in range(len(sizes) - 2):
        width = sizes[index + 1]
        layers.append(nn.Linear(sizes[index], width))
        if norm == 'gn':
            layers.append(nn.GroupNorm(min(32, width), width))
        else:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class DualPeakTandemRuntime(nn.Module):
    """Inference-only 2-peak Tandem network without dataset side effects."""

    def __init__(self):
        super().__init__()
        self.fnn = build_dense(FNN_SIZE)
        self.inn = build_dense(INN_SIZE)
        self.register_buffer(
            'structure_min',
            torch.as_tensor(STRUCTURE_MIN),
        )
        self.register_buffer(
            'structure_max',
            torch.as_tensor(STRUCTURE_MAX),
        )

    def load_checkpoints(self, device):
        for checkpoint in (FNN_CHECKPOINT, INN_CHECKPOINT):
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f'Missing Tandem checkpoint: {checkpoint}'
                )
        self.fnn.load_state_dict(
            torch.load(
                FNN_CHECKPOINT,
                map_location=device,
                weights_only=True,
            )
        )
        self.inn.load_state_dict(
            torch.load(
                INN_CHECKPOINT,
                map_location=device,
                weights_only=True,
            )
        )
        self.to(device)
        self.eval()
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def inverse_forward(self, responses):
        response_tensor = torch.as_tensor(
            responses,
            dtype=torch.float32,
            device=self.device,
        )
        if response_tensor.ndim == 1:
            response_tensor = response_tensor.unsqueeze(0)
        with torch.no_grad():
            normalized_structure = self.inn(response_tensor)
            structures = (
                normalized_structure
                * (self.structure_max - self.structure_min)
                + self.structure_min
            )
            reconstructed = self.fnn(structures)
        return (
            structures.detach().cpu().numpy(),
            reconstructed.detach().cpu().numpy(),
        )

    def forward_only(self, structures):
        structure_tensor = torch.as_tensor(
            structures,
            dtype=torch.float32,
            device=self.device,
        )
        if structure_tensor.ndim == 1:
            structure_tensor = structure_tensor.unsqueeze(0)
        with torch.no_grad():
            responses = self.fnn(structure_tensor)
        return responses.detach().cpu().numpy()


def build_tandem_runtime(device):
    return DualPeakTandemRuntime().load_checkpoints(device)


def build_cvae(device):
    if not os.path.isfile(CVAE_CHECKPOINT):
        raise FileNotFoundError(
            f'Missing CVAE checkpoint: {CVAE_CHECKPOINT}'
        )
    model = CVAE_2p.CVAE(
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        c_dim=2,
    ).to(device)
    model.load_state_dict(
        torch.load(
            CVAE_CHECKPOINT,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    return model


def sorted_peak_pair(peak1, peak2):
    pair = np.sort(np.asarray([peak1, peak2], dtype=float))
    if not np.all(np.isfinite(pair)):
        raise ValueError('Peak wavelengths must be finite.')
    if pair[0] <= 0 or pair[1] <= pair[0]:
        raise ValueError(
            'Peak wavelengths must be positive and distinct.'
        )
    return pair


def case_tag(peak1, peak2):
    pair = sorted_peak_pair(peak1, peak2)

    def format_value(value):
        if np.isclose(value, round(value), atol=1e-9):
            return str(int(round(value)))
        return f'{value:g}'.replace('.', 'p')

    return f'{format_value(pair[0])}_{format_value(pair[1])}'


def shape_gate_verdict(
    spectrum,
    wavelengths,
    ideal_target,
    detection=None,
    prominence=0.01,
    minimum_distance_nm=20.0,
    maximum_valley_transmittance=0.35,
    minimum_prominence=0.30,
    maximum_valley_imbalance=0.10,
    minimum_width_ratio=0.50,
    maximum_width_ratio=1.50,
):
    """Apply the manuscript's admissibility conditions to one response.

    The same four conditions ``select_cvae_assisted_design`` screens candidates
    with, in one place so that the figures and sweeps cannot drift from the
    selector. They did: a reimplementation in the reporting scripts carried
    only three of them and silently admitted responses whose notches are the
    wrong width.

    Returns ``(passed, failed_conditions, measurements)``. The caller decides
    what to do with a failure -- the conditions say whether a response is a
    usable double notch, not whether a design exists, so a failing design still
    has dip positions worth reporting.
    """
    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    if detection is None:
        detection = detect_two_resonances(
            spectrum, wavelengths,
            prominence=prominence,
            minimum_distance_nm=minimum_distance_nm,
        )
    if not detection.valid:
        return False, ['two_resonances'], {}

    quality = resonance_quality_metrics(spectrum, wavelengths, detection)
    ideal_detection = detect_two_resonances(
        np.asarray(ideal_target, dtype=float).reshape(-1), wavelengths,
        prominence=prominence, minimum_distance_nm=minimum_distance_nm)
    ideal_quality = resonance_quality_metrics(
        np.asarray(ideal_target, dtype=float).reshape(-1), wavelengths,
        ideal_detection)

    depths = np.asarray(detection.values, dtype=float)
    imbalance = float(abs(depths[0] - depths[1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        width_ratios = (np.asarray(quality.fwhm_nm, dtype=float)
                        / np.asarray(ideal_quality.fwhm_nm, dtype=float))

    failed = []
    if not quality.valid:
        failed.append('width')
    if quality.worst_valley_transmittance > maximum_valley_transmittance:
        failed.append('valley_depth')
    if quality.minimum_prominence < minimum_prominence:
        failed.append('prominence')
    if imbalance > maximum_valley_imbalance:
        failed.append('imbalance')
    if not (np.all(width_ratios >= minimum_width_ratio)
            and np.all(width_ratios <= maximum_width_ratio)):
        failed.append('width_ratio')

    measurements = {
        'peaks': [float(v) for v in detection.wavelengths],
        'depths': [float(v) for v in depths],
        'worst_depth': float(quality.worst_valley_transmittance),
        'minimum_prominence': float(quality.minimum_prominence),
        'valley_imbalance': imbalance,
        'width_ratios': [float(v) for v in width_ratios],
        'separation_nm': float(detection.wavelengths[1]
                               - detection.wavelengths[0]),
    }
    return not failed, failed, measurements


# Grid of double-notch targets the supplement figures sweep. One definition,
# because plot_design_gallery.py and plot_budget_ablation.py each used to carry
# their own copy and they drifted: both capped the second notch at 640 nm and
# spaced the notches 70/110/150 nm apart, which excluded every round
# separation and so excluded all three main-text cases from the figures meant
# to show those cases are representative.
#
# Both ends are measured against what the structure family can actually do.
# Of the 11,849 training spectra, 6,484 present two admissible notches; among
# those the first notch spans 404.9-611.0 nm and the second 431.4-659.7 nm.
# Nothing admissible sits below 405 nm or above 660 nm, so a 400 nm first
# notch is not a hard target, it is an impossible one -- an earlier version of
# this grid asked for it and the CVAE answered with a 0.88-transmittance
# ripple, which is the correct answer to an unanswerable question.
SUPPLEMENT_PAIR_LOWS = (410.0, 450.0, 500.0, 550.0, 600.0)
SUPPLEMENT_PAIR_SEPARATIONS = (50.0, 100.0, 150.0)
SUPPLEMENT_PAIR_HIGH_LIMITS = (450.0, 660.0)


def supplement_pair_grid():
    """The (low, high) target pairs the supplement figures sweep, in order."""
    low_limit, high_limit = SUPPLEMENT_PAIR_HIGH_LIMITS
    return [
        (low, low + separation)
        for low in SUPPLEMENT_PAIR_LOWS
        for separation in SUPPLEMENT_PAIR_SEPARATIONS
        if low_limit <= low + separation <= high_limit
    ]


def bounded_double_lorentzian(centers, wavelengths, gamma=15.0):
    centers = np.sort(np.asarray(centers, dtype=float))
    wavelengths = np.asarray(wavelengths, dtype=float)
    if centers.shape != (2,):
        raise ValueError('Exactly two target centers are required.')
    if gamma <= 0:
        raise ValueError('gamma must be positive.')

    transmittance = np.ones_like(wavelengths, dtype=float)
    for center in centers:
        dip = 1.0 / (1.0 + ((wavelengths - center) / gamma) ** 2)
        transmittance *= 1.0 - dip
    return transmittance.astype(np.float32)


def target_grid_indices(centers, wavelengths):
    centers = np.sort(np.asarray(centers, dtype=float))
    wavelengths = np.asarray(wavelengths, dtype=float)
    return np.array(
        [
            int(np.argmin(np.abs(wavelengths - center)))
            for center in centers
        ],
        dtype=int,
    )


def detect_two_resonances(
    spectrum,
    wavelengths,
    prominence=0.01,
    minimum_distance_nm=20.0,
):
    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    if spectrum.shape != wavelengths.shape:
        raise ValueError('Spectrum and wavelength shapes do not match.')

    wavelength_step = float(np.median(np.diff(wavelengths)))
    distance_samples = max(
        1,
        int(round(minimum_distance_nm / wavelength_step)),
    )
    all_indices, _ = find_peaks(
        -spectrum,
        prominence=prominence,
        distance=distance_samples,
    )
    all_indices = np.asarray(all_indices, dtype=int)
    if len(all_indices) < 2:
        return PeakDetection(
            valid=False,
            indices=np.array([-1, -1], dtype=int),
            wavelengths=np.array([np.nan, np.nan], dtype=float),
            values=np.array([np.nan, np.nan], dtype=float),
            all_indices=all_indices,
        )

    deepest = all_indices[np.argsort(spectrum[all_indices])[:2]]
    selected = np.sort(deepest)
    return PeakDetection(
        valid=True,
        indices=selected,
        wavelengths=wavelengths[selected],
        values=spectrum[selected],
        all_indices=all_indices,
    )


def peak_error_metrics(detection, target_centers):
    target_centers = np.sort(np.asarray(target_centers, dtype=float))
    if not detection.valid:
        return {
            'signed_errors_nm': np.array([np.nan, np.nan]),
            'absolute_errors_nm': np.array([np.nan, np.nan]),
            'max_absolute_error_nm': np.nan,
            'pair_rmse_nm': np.nan,
            'midpoint_error_nm': np.nan,
            'separation_error_nm': np.nan,
        }

    signed_errors = detection.wavelengths - target_centers
    absolute_errors = np.abs(signed_errors)
    predicted_midpoint = float(np.mean(detection.wavelengths))
    target_midpoint = float(np.mean(target_centers))
    predicted_separation = float(np.diff(detection.wavelengths)[0])
    target_separation = float(np.diff(target_centers)[0])
    return {
        'signed_errors_nm': signed_errors,
        'absolute_errors_nm': absolute_errors,
        'max_absolute_error_nm': float(np.max(absolute_errors)),
        'pair_rmse_nm': float(np.sqrt(np.mean(signed_errors ** 2))),
        'midpoint_error_nm': abs(predicted_midpoint - target_midpoint),
        'separation_error_nm': abs(
            predicted_separation - target_separation
        ),
    }


def resonance_quality_metrics(spectrum, wavelengths, detection):
    """Measure depth and half-prominence width for the selected valleys.

    FWHM is evaluated with ``peak_widths`` on the inverted spectrum, so the
    half-height is defined relative to each valley's local prominence rather
    than an assumed unit baseline.  The worst (shallower/wider) of the two
    resonances is used by the candidate selector, ensuring that one excellent
    valley cannot hide a weak second valley.
    """
    empty = np.array([np.nan, np.nan], dtype=float)
    if not detection.valid:
        return ResonanceQuality(
            valid=False,
            prominences=empty.copy(),
            fwhm_nm=empty.copy(),
            quality_factors=empty.copy(),
            contrast_quality_fom=empty.copy(),
            worst_valley_transmittance=float('nan'),
            widest_fwhm_nm=float('nan'),
            minimum_prominence=float('nan'),
            minimum_contrast_quality_fom=float('nan'),
        )

    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
    wavelengths = np.asarray(wavelengths, dtype=float).reshape(-1)
    inverted = -spectrum
    prominences = peak_prominences(
        inverted,
        detection.indices,
    )[0]
    widths = peak_widths(
        inverted,
        detection.indices,
        rel_height=0.5,
        prominence_data=peak_prominences(
            inverted,
            detection.indices,
        ),
    )
    sample_positions = np.arange(len(wavelengths), dtype=float)
    left_wavelengths = np.interp(
        widths[2],
        sample_positions,
        wavelengths,
    )
    right_wavelengths = np.interp(
        widths[3],
        sample_positions,
        wavelengths,
    )
    fwhm_nm = right_wavelengths - left_wavelengths
    quality_factors = np.divide(
        detection.wavelengths,
        fwhm_nm,
        out=np.full(2, np.nan, dtype=float),
        where=fwhm_nm > 0,
    )
    contrasts = np.clip(1.0 - detection.values, 0.0, None)
    contrast_quality_fom = contrasts * quality_factors
    valid = bool(
        np.all(np.isfinite(prominences))
        and np.all(np.isfinite(fwhm_nm))
        and np.all(fwhm_nm > 0)
    )
    return ResonanceQuality(
        valid=valid,
        prominences=prominences,
        fwhm_nm=fwhm_nm,
        quality_factors=quality_factors,
        contrast_quality_fom=contrast_quality_fom,
        worst_valley_transmittance=float(np.max(detection.values)),
        widest_fwhm_nm=float(np.max(fwhm_nm)),
        minimum_prominence=float(np.min(prominences)),
        minimum_contrast_quality_fom=float(
            np.min(contrast_quality_fom)
        ),
    )


def make_condition_orders(target_indices, n_wavelengths, device):
    """The condition the CVAE was trained on: the pair sorted by wavelength.

    This used to return the reversed pair as well, and half of every candidate
    batch was decoded from it.  ``CVAE_2p.lowest_wavelength`` sorts the two
    dips ascending before forming the training condition, so the reversed pair
    is an input the model has never seen: its decoded targets miss the request
    by 54 nm at the median against 5 nm for the sorted pair, and the gate on
    the decoded target then discards them, leaving half the nominal candidate
    budget unusable.
    """
    scaled = np.asarray(target_indices, dtype=np.float32) / (
        n_wavelengths - 1
    )
    low_high = torch.as_tensor(
        np.sort(scaled)[None, :],
        dtype=torch.float32,
        device=device,
    )
    return (low_high,)


def generate_cvae_candidates(
    model,
    target_indices,
    n_wavelengths,
    n_candidates,
    device,
    base_latents=None,
):
    if n_candidates < 1:
        raise ValueError('n_candidates must be at least 1.')
    candidates_per_order = n_candidates
    if base_latents is None:
        base_latents = torch.randn(
            candidates_per_order,
            ENCODER_SIZE[-1],
            device=device,
        )
    else:
        base_latents = torch.as_tensor(
            base_latents,
            dtype=torch.float32,
            device=device,
        )
        if base_latents.shape != (
            candidates_per_order,
            ENCODER_SIZE[-1],
        ):
            raise ValueError(
                'base_latents has an incompatible shape: '
                f'{tuple(base_latents.shape)}'
            )

    conditions = make_condition_orders(
        target_indices,
        n_wavelengths,
        device,
    )
    decoded_batches = []
    for condition in conditions:
        expanded_condition = condition.expand(candidates_per_order, -1)
        with torch.no_grad():
            decoded_batches.append(
                model.decoder(base_latents, expanded_condition)
            )
    candidates = torch.cat(decoded_batches, dim=0)
    order_codes = np.zeros(len(candidates), dtype=int)
    return candidates, order_codes, base_latents


def select_cvae_assisted_design(
    model,
    tandem,
    target_centers,
    ideal_target,
    wavelengths,
    n_candidates,
    device,
    base_latents=None,
    prominence=0.01,
    minimum_distance_nm=20.0,
    position_slack_nm=2.5,
    maximum_valley_transmittance=0.35,
    minimum_prominence=0.30,
    maximum_valley_imbalance=0.10,
    minimum_width_ratio=0.50,
    maximum_width_ratio=1.50,
    # Calibrated for this system rather than copied from the single-notch
    # selector's 0.5 nm: the decoded target has to place two resonances at once
    # and its best achievable error over a hundred candidates is 2-5 nm, so a
    # sub-nanometre gate never admits anybody and collapses onto its fallback.
    # At 5 nm the gate rejects the decoded targets that are plainly off without
    # emptying the pool.
    candidate_wavelength_tolerance_nm=5.0,
    candidate_override=None,
):
    target_centers_sorted = np.sort(
        np.asarray(target_centers, dtype=float)
    )
    target_indices = target_grid_indices(target_centers, wavelengths)
    if candidate_override is None:
        candidate_tensor, order_codes, base_latents = generate_cvae_candidates(
            model=model,
            target_indices=target_indices,
            n_wavelengths=len(wavelengths),
            n_candidates=n_candidates,
            device=device,
            base_latents=base_latents,
        )
    else:
        # Candidates supplied from outside, so the same admissibility
        # conditions and ordering can be applied to a set the CVAE did not
        # produce.  Used by the matched-budget ablation.
        candidate_tensor = torch.as_tensor(
            np.asarray(candidate_override, dtype=np.float32), device=device)
        n_candidates = len(candidate_tensor)
        order_codes = np.zeros(n_candidates, dtype=int)
    candidates = candidate_tensor.detach().cpu().numpy()
    structures, responses = tandem.inverse_forward(candidate_tensor)
    response_mse = np.mean(
        (responses - np.asarray(ideal_target)[None, :]) ** 2,
        axis=1,
    )

    candidate_invalid = np.ones(n_candidates, dtype=int)
    candidate_maximum_index_error = np.full(n_candidates, np.inf)
    candidate_summed_index_error = np.full(n_candidates, np.inf)
    candidate_maximum_wavelength_error = np.full(
        n_candidates,
        np.inf,
    )
    candidate_summed_wavelength_error = np.full(
        n_candidates,
        np.inf,
    )
    final_invalid = np.ones(n_candidates, dtype=int)
    # The transmittance of the shallower of the two notches, which is what the
    # ordering uses as the intensity criterion.
    final_worst_valley = np.full(n_candidates, np.inf)
    final_maximum_index_error = np.full(n_candidates, np.inf)
    final_summed_index_error = np.full(n_candidates, np.inf)
    final_maximum_wavelength_error = np.full(n_candidates, np.inf)
    final_summed_wavelength_error = np.full(n_candidates, np.inf)
    shape_minimax_score = np.full(n_candidates, np.inf)
    valley_imbalance = np.full(n_candidates, np.inf)
    width_ratios = np.full((n_candidates, 2), np.nan)
    shape_gate_failed = np.ones(n_candidates, dtype=int)
    candidate_detections = []
    final_detections = []
    final_qualities = []
    for candidate, response in zip(candidates, responses):
        candidate_detection = detect_two_resonances(
            candidate,
            wavelengths,
            prominence=prominence,
            minimum_distance_nm=minimum_distance_nm,
        )
        final_detection = detect_two_resonances(
            response,
            wavelengths,
            prominence=prominence,
            minimum_distance_nm=minimum_distance_nm,
        )
        candidate_detections.append(candidate_detection)
        final_detections.append(final_detection)
        final_qualities.append(
            resonance_quality_metrics(
                response,
                wavelengths,
                final_detection,
            )
        )

    for index, detection in enumerate(candidate_detections):
        if detection.valid:
            index_error = np.abs(detection.indices - target_indices)
            wavelength_error = np.abs(
                detection.wavelengths - target_centers_sorted
            )
            candidate_invalid[index] = 0
            candidate_maximum_index_error[index] = float(
                np.max(index_error)
            )
            candidate_summed_index_error[index] = float(
                np.sum(index_error)
            )
            candidate_maximum_wavelength_error[index] = float(
                np.max(wavelength_error)
            )
            candidate_summed_wavelength_error[index] = float(
                np.sum(wavelength_error)
            )

    ideal_detection = detect_two_resonances(
        ideal_target,
        wavelengths,
        prominence=prominence,
        minimum_distance_nm=minimum_distance_nm,
    )
    ideal_quality = resonance_quality_metrics(
        ideal_target,
        wavelengths,
        ideal_detection,
    )
    if not ideal_quality.valid:
        raise ValueError(
            'The ideal target does not contain two measurable resonances.'
        )

    for index, detection in enumerate(final_detections):
        if detection.valid:
            index_error = np.abs(detection.indices - target_indices)
            wavelength_error = np.abs(
                detection.wavelengths - target_centers_sorted
            )
            final_invalid[index] = 0
            final_worst_valley[index] = float(np.max(detection.values))
            final_maximum_index_error[index] = float(
                np.max(index_error)
            )
            final_summed_index_error[index] = float(
                np.sum(index_error)
            )
            final_maximum_wavelength_error[index] = float(
                np.max(wavelength_error)
            )
            final_summed_wavelength_error[index] = float(
                np.sum(wavelength_error)
            )
            quality = final_qualities[index]
            if quality.valid:
                valley_imbalance[index] = float(
                    abs(detection.values[0] - detection.values[1])
                )
                width_ratios[index] = (
                    quality.fwhm_nm / ideal_quality.fwhm_nm
                )
                relative_width_error = np.max(
                    np.abs(
                        quality.fwhm_nm - ideal_quality.fwhm_nm
                    )
                    / ideal_quality.fwhm_nm
                )
                shape_minimax_score[index] = max(
                    quality.worst_valley_transmittance / 0.20,
                    valley_imbalance[index] / 0.10,
                    relative_width_error / 0.35,
                    0.50 / max(quality.minimum_prominence, 1e-12),
                )
                passes_shape_gate = (
                    quality.worst_valley_transmittance
                    <= maximum_valley_transmittance
                    and quality.minimum_prominence
                    >= minimum_prominence
                    and valley_imbalance[index]
                    <= maximum_valley_imbalance
                    and np.all(
                        width_ratios[index] >= minimum_width_ratio
                    )
                    and np.all(
                        width_ratios[index] <= maximum_width_ratio
                    )
                )
                shape_gate_failed[index] = int(
                    not passes_shape_gate
                )

    eligible_for_selection = (
        (final_invalid == 0)
        & (shape_gate_failed == 0)
    )
    # The shape gate ranks rather than rejects: ``shape_gate_failed`` is
    # already a key in the lexsort below, so gate-passing candidates always
    # win when any exist.  Failing hard when none do would drop the whole
    # case, which is not what the published sweep did -- its summary reports
    # 30 detected cases of which 28 passed the gate, i.e. two were selected
    # despite failing it (472_512 at worst-valley 0.46, 538_578 at score
    # 6.18).  So fall back to any candidate whose final response is a valid
    # two-resonance spectrum, and let ``selected_shape_gate_passed`` record
    # that the gate was not met.
    shape_gate_relaxed = False
    if not np.any(eligible_for_selection):
        eligible_for_selection = final_invalid == 0
        shape_gate_relaxed = True
    if not np.any(eligible_for_selection):
        raise ValueError(
            'No CVAE candidate produced a valid final two-resonance '
            'response.'
        )

    finite_position_errors = final_maximum_wavelength_error[
        eligible_for_selection
    ]
    if len(finite_position_errors):
        best_position_error = float(np.min(finite_position_errors))
        position_window_nm = best_position_error + position_slack_nm
    else:
        best_position_error = float('inf')
        position_window_nm = float('inf')
    outside_position_window = (
        final_maximum_wavelength_error > position_window_nm
    ).astype(int)

    # Selection is position-first after the validity and required shape
    # gates.  In particular, ``outside_position_window`` is deliberately
    # excluded from the ordering: the best + slack window remains useful as
    # a diagnostic, but candidates inside that window may no longer trade a
    # worse requested-wavelength match for a better shape score.
    #
    # ``np.lexsort`` uses the final key as the primary key.  The order is the
    # one the single-notch selector uses -- wavelength, then the transmittance
    # at the notch, then the decoded target's own wavelength, then the
    # full-spectrum error -- with the wavelength criterion taken as the worse
    # of the two resonances before their sum, and the intensity criterion as
    # the shallower of the two notches.
    # The decoded target's own resonances are gated before the ordering, as in
    # the single-notch selector.  Left to the ordering alone the search can
    # take a decoded target deliberately offset by a nanometre or two, because
    # the inverse-forward pair carries a systematic shift of the opposite sign
    # and the two cancel: the final response then lands on specification while
    # the target it was asked to realise does not.  As there, the gate relaxes
    # to the best observed error plus one grid step rather than failing.
    grid_step = float(np.median(np.diff(wavelengths)))
    slack = 0.5 * grid_step
    candidate_gate_relaxed = False
    if candidate_wavelength_tolerance_nm is None:
        candidate_eligible = np.ones(n_candidates, dtype=bool)
    else:
        limit = float(candidate_wavelength_tolerance_nm) + slack
        candidate_eligible = candidate_maximum_wavelength_error <= limit
        if not np.any(candidate_eligible):
            finite = np.isfinite(candidate_maximum_wavelength_error)
            if np.any(finite):
                limit = (float(np.min(candidate_maximum_wavelength_error[finite]))
                         + grid_step + 10 * np.finfo(float).eps)
                candidate_eligible = (
                    candidate_maximum_wavelength_error <= limit)
                candidate_gate_relaxed = True
            else:
                candidate_eligible = np.ones(n_candidates, dtype=bool)
                candidate_gate_relaxed = True

    # The manuscript's Screen and Rank, read bottom-up because np.lexsort
    # takes the last key as the primary one.
    #
    #   Screen -- CTS consistency : ~candidate_eligible, candidate_invalid
    #             response validity: final_invalid, shape_gate_failed
    #             (shape_gate_failed is the response-validity set: T_min
    #             <= 0.35 per notch, depth difference <= 0.10, half-depth
    #             width ratio in [0.5, 1.5], prominence >= 0.30)
    #   Rank   -- 1. realized notch position: larger error, then the sum
    #             2. minimum transmittance of the realized response
    #             3. CTS notch position: larger error, then the sum
    #             4. full-spectrum MSE between response and CTS
    #
    # `shape_minimax_score` is deliberately absent: it is a useful diagnostic
    # and is still computed and reported, but the manuscript's rank has four
    # criteria and it is not one of them.
    order = np.lexsort((
        response_mse,
        candidate_summed_wavelength_error,
        candidate_maximum_wavelength_error,
        candidate_invalid,
        final_worst_valley,
        final_summed_wavelength_error,
        final_maximum_wavelength_error,
        shape_gate_failed,
        final_invalid,
        ~candidate_eligible,
    ))
    selected_index = int(order[0])
    selected_detection = candidate_detections[selected_index]
    final_detection = final_detections[selected_index]
    candidate_diagnostics = []
    for index, (detection, quality) in enumerate(
        zip(final_detections, final_qualities)
    ):
        if detection.valid:
            signed_errors = (
                detection.wavelengths - target_centers_sorted
            )
            wavelengths_found = detection.wavelengths
            valley_values = detection.values
        else:
            signed_errors = np.array([np.nan, np.nan], dtype=float)
            wavelengths_found = np.array([np.nan, np.nan], dtype=float)
            valley_values = np.array([np.nan, np.nan], dtype=float)
        candidate_diagnostics.append({
            'candidate_index': index,
            'condition_order_code': int(order_codes[index]),
            'valid': int(detection.valid and quality.valid),
            'candidate_spectrum_valid': int(
                candidate_invalid[index] == 0
            ),
            'candidate_spectrum_maximum_absolute_error_nm': float(
                candidate_maximum_wavelength_error[index]
            ),
            'candidate_spectrum_summed_absolute_error_nm': float(
                candidate_summed_wavelength_error[index]
            ),
            'peak1_wavelength_nm': float(wavelengths_found[0]),
            'peak2_wavelength_nm': float(wavelengths_found[1]),
            'peak1_signed_error_nm': float(signed_errors[0]),
            'peak2_signed_error_nm': float(signed_errors[1]),
            'maximum_absolute_error_nm': float(
                final_maximum_wavelength_error[index]
            ),
            'summed_absolute_error_nm': float(
                final_summed_wavelength_error[index]
            ),
            'mean_absolute_error_nm': float(
                0.5 * final_summed_wavelength_error[index]
            ),
            'peak1_transmittance': float(valley_values[0]),
            'peak2_transmittance': float(valley_values[1]),
            'worst_valley_transmittance': float(
                quality.worst_valley_transmittance
            ),
            'peak1_fwhm_nm': float(quality.fwhm_nm[0]),
            'peak2_fwhm_nm': float(quality.fwhm_nm[1]),
            'widest_fwhm_nm': float(quality.widest_fwhm_nm),
            'peak1_prominence': float(quality.prominences[0]),
            'peak2_prominence': float(quality.prominences[1]),
            'minimum_prominence': float(quality.minimum_prominence),
            'valley_transmittance_imbalance': float(
                valley_imbalance[index]
            ),
            'peak1_fwhm_to_ideal_ratio': float(
                width_ratios[index, 0]
            ),
            'peak2_fwhm_to_ideal_ratio': float(
                width_ratios[index, 1]
            ),
            'shape_gate_passed': int(not shape_gate_failed[index]),
            'shape_minimax_score': float(
                shape_minimax_score[index]
            ),
            'inside_position_window': int(
                not outside_position_window[index]
            ),
            'selected': int(index == selected_index),
        })

    return {
        'selected_index': selected_index,
        'condition_order_code': int(order_codes[selected_index]),
        'shape_gate_relaxed': bool(shape_gate_relaxed),
        'candidate_valid_count': int(np.sum(candidate_invalid == 0)),
        'final_valid_count': int(np.sum(final_invalid == 0)),
        'best_available_position_error_nm': best_position_error,
        'position_quality_window_nm': position_window_nm,
        'selected_shape_gate_passed': int(
            not shape_gate_failed[selected_index]
        ),
        'selected_shape_minimax_score': float(
            shape_minimax_score[selected_index]
        ),
        'ideal_quality': ideal_quality,
        'quality_thresholds': {
            'position_window_slack_nm_diagnostic_only': (
                position_slack_nm
            ),
            'maximum_valley_transmittance': (
                maximum_valley_transmittance
            ),
            'minimum_prominence': minimum_prominence,
            'maximum_valley_imbalance': maximum_valley_imbalance,
            'minimum_width_ratio': minimum_width_ratio,
            'maximum_width_ratio': maximum_width_ratio,
        },
        'candidate_maximum_index_error': float(
            candidate_maximum_index_error[selected_index]
        ),
        'candidate_summed_index_error': float(
            candidate_summed_index_error[selected_index]
        ),
        'candidate_maximum_wavelength_error': float(
            candidate_maximum_wavelength_error[selected_index]
        ),
        'candidate_summed_wavelength_error': float(
            candidate_summed_wavelength_error[selected_index]
        ),
        'final_maximum_index_error': float(
            final_maximum_index_error[selected_index]
        ),
        'final_summed_index_error': float(
            final_summed_index_error[selected_index]
        ),
        'final_maximum_wavelength_error': float(
            final_maximum_wavelength_error[selected_index]
        ),
        'final_summed_wavelength_error': float(
            final_summed_wavelength_error[selected_index]
        ),
        'candidate_spectrum': candidates[selected_index],
        'candidate_detection': selected_detection,
        'structure': structures[selected_index],
        'final_response': responses[selected_index],
        'final_detection': final_detection,
        'final_quality': final_qualities[selected_index],
        'candidate_diagnostics': candidate_diagnostics,
        'ideal_to_candidate_mse': float(
            np.mean(
                (
                    candidates[selected_index]
                    - np.asarray(ideal_target)
                ) ** 2
            )
        ),
        'ideal_to_final_mse': float(response_mse[selected_index]),
        'candidate_to_final_mse': float(
            np.mean(
                (
                    candidates[selected_index]
                    - responses[selected_index]
                ) ** 2
            )
        ),
        'base_latents': base_latents,
    }
