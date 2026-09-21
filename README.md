# Attainability-Aware Inverse Design of Narrowband Nanophotonic Structures

Code and results for *Attainability-Aware Inverse Design of Narrowband
Nanophotonic Structures from Incomplete Spectral Specifications* (Li, Kim,
Zhang, Zhang, Wang, Liu).

A design request here is a notch wavelength, or a pair of them — not a
complete target spectrum. A conditional VAE turns that request into several
candidate target spectra drawn from the distribution the structure family can
actually produce; a tandem inverse network maps each to a structure; a frozen
forward surrogate predicts what that structure does; and a screening and
ranking step picks one. The comparison throughout is against the conventional
route, which feeds an analytically prescribed Lorentzian target straight to
the inverse network.

Two systems are studied: a single-notch filter (2 design parameters) and a
double-notch filter (3 design parameters).

## Layout

```
1peak/                  single-notch system
2peak/                  double-notch system
  <system>/model/           the deployed networks + provenance.json
  <system>/weight_grid_runs*/   loss-weight sweep dumps and training logs
  2peak/<low>_<high>/       one design case: tandem/, cvae/, cst/
  1peak/<centre>/           likewise for the single-notch cases
Filter_CST_DataGen_Share/   CST dataset generation (see "Regenerating the dataset")
figures/                  paper figures and the numbers behind them
*.py                      the analysis and plotting entry points (repo root)
```

`figures/PAPER_NUMBERS.md` records every number quoted in the manuscript
together with the file it came from, including the cases where the value in
an earlier draft no longer holds and why.

## Requirements

Python 3.11 with PyTorch (CUDA optional), NumPy, SciPy, matplotlib, pandas and
`shap`. CST Studio Suite 2026 is needed only to regenerate the datasets or to
re-run the full-wave validation; every figure below is reproducible without
it from the shipped result files.

## Data and trained models

Tracked here, so the repository reproduces the paper on its own:

| | size | contents |
|---|---|---|
| `2peak/data/cstdata_35547.npz` | 79 MB | double-notch training set |
| `1peak/data/` | 16 MB | single-notch training set |
| `<system>/model/` | 261 MB | the six deployed networks and their provenance |

Cloning therefore pulls about 380 MB. What is not here is the 35,547 raw
per-sample exports the double-notch bundle was packed from: the bundle is the
mean over the three incidence angles, so the raw files additionally hold each
angle on its own. Nothing in this repository reads them, and every figure and
table is reproducible without them.

The double-notch set is a full factorial grid: both host refractive indices
from 1.50 to 3.05 in steps of 0.1 (17 values each) and the nanoparticle radius
from 2.5 to 6.5 nm in steps of 0.1 nm (41 values), giving 11,849 structures.
Each was simulated at 0, 30 and 60 degrees and the three spectra averaged, so
35,547 full-wave runs stand behind it. Spectra are sampled at 1001 points from
380 to 799 nm.


## Reproducing the figures

Run from the repository root.

| Figure | Command |
|---|---|
| Fig. 3, Fig. 4 | `python plot_spectral_comparison_3x3.py` |
| Fig. 5 | `python 1peak/plot_notch_filter_comparison.py` |
| Fig. 6, Fig. S7 | `python plot_shap_attribution.py --kind 1peak --output figures/shap_single.pdf`  /  `--kind 2peak --output figures/shap_dual.pdf` |
| Fig. S1, S2 | `python plot_weight_grid.py --dumps "Forward, single=1peak/weight_grid_runs/grid16_1peak_forward.json" "Tandem, single=1peak/weight_grid_runs/grid16_1peak_tandem.json" "CVAE, single=1peak/weight_grid_runs/grid16_1peak_cvae.json" "Forward, double=2peak/weight_grid_runs/grid16_2peak_forward.json" "Tandem, double=2peak/weight_grid_runs/grid16_2peak_tandem.json" "CVAE, double=2peak/weight_grid_runs/grid16_2peak_cvae.json" --layout case --output figures/weight_grid.pdf` |
| Fig. S3 | `python plot_loss_comparison_2x3.py` |
| Fig. S4 | `python plot_design_gallery.py --kind 1peak --limit 12 --output figures/design_gallery_single.pdf` |
| Fig. S5 | `python plot_design_gallery.py --kind 2peak --output figures/design_gallery_dual.pdf` |
| Fig. S6 | `python plot_budget_ablation.py --output figures/budget_ablation.png` |

Both gallery scripts cache their curves next to the figure, so
`--from-cache` replots without re-running the networks.

Tables 1 and 2 are read from the case directories; the supplement's test-set
loss components come from `python evaluate_deployed_losses.py --system 1peak`
(and `--system 2peak`).

Two sweeps feed the statistics: `python sweep_single.py --output
figures/single_bn.json` (25 single-notch targets) and `python sweep_pairs.py
--grid --output figures/pair_bn.json` (446 double-notch pairs).
`python plot_design_error_statistics.py --single figures/single_bn.json --dual
figures/pair_bn.json --output figures/design_error_statistics.pdf` draws the
distribution over both.

## Training

```
python 1peak/train_networks.py                 # forward, tandem, CVAE
python 2peak/train_networks_2p.py
```

The deployed networks are not one run each: every network is the best cell of
its own 4×4 physics-loss weight grid, scored on the held-out test tenth, and
the three cells differ. `<system>/model/provenance.json` records which cell
each checkpoint came from, its γ and δ, and its md5. To rebuild the grids:

```
python sweep_physics_weights.py --system 2peak --split test
python sweep_physics_weights.py --system 2peak --split test --stage tandem-on-best-forward
```

Every tandem cell trains against the same frozen forward checkpoint — the one
the forward grid selected — so the tandem row isolates the loss's effect on
the inverse network.

## Designing a new structure

```
python 2peak/run_dual_peak_case.py --peak1 500 --peak2 550 --n-candidates 20
```

writes `2peak/500_550/` with the direct-route and CVAE-route designs, their
predicted spectra, and the ranked candidate table.

## How a notch is located and judged

Two things are deliberately separate, and mixing them up was the source of
several errors during this work:

**Detection.** Double-notch dip positions come from
`dual_peak_utils.detect_two_resonances`: `find_peaks` at prominence 0.01 with
a 20 nm minimum separation, keeping the two deepest and sorting by
wavelength. The separation constraint is part of what "two notches" means —
without it two samples of one broad notch can be returned as a pair. The
single-notch case uses the spectrum's argmin.

**Admissibility.** A response is a usable double notch only if the shallower
valley reaches T ≤ 0.35, both prominences are ≥ 0.30, the two depths differ
by ≤ 0.10, and both widths are within a factor 0.5–1.5 of the target's.
`dual_peak_utils.shape_gate_verdict` is the single definition, shared by the
candidate selector and by every figure. The single-notch analogue is
`notch_quality.single_notch_verdict`, at the 0.10 depth threshold that
`select_single_notch_candidate` screens with.

Admissibility is **reported, not filtered**: the inverse network always
returns a structure, so a failing design still has dip positions and still
carries its wavelength error. Figures S4 and S5 mark those with a dagger. The
direct route has no selection step at all, so dropping its failures would
compare a screened route against an unscreened one.

## Regenerating the dataset

Only needed to rebuild the training data from scratch; it took about 40 days
of wall-clock CST time on this machine.

```
cd Filter_CST_DataGen_Share
./run_2peak_dataset.sh --jobs 4 --theta-list "0 30 60" --output-dir <dir>
python pack_dataset.py <dir> --average-theta --verify
```

`check_dataset.py` validates a finished directory. CST is driven headless
through `cst.interface` under `xvfb-run`; four shards is the throughput
optimum on 28 cores — eight shards is slower, because each gets too few cores
for the solver's fixed overhead to amortise.

## Citation

Citation details will be added on publication.
