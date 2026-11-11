# Nuisance-AUC

Code and results for *Propensity-Adjusted AUC under Nuisance-Model
Misspecification*.

The package `nuisauc` implements the shortcut-adjusted AUC (propensity-stratified,
pair-weighted), its two self-derived checks, the correlation law, the simulation
designs, and every experiment in the paper. Each experiment is a registered,
config-driven module run through one command-line tool. Every result table in the
paper is produced by `nuisauc run` and can be checked against the shipped reference
tables with `nuisauc verify`.

## Install

```bash
pip install -e .            # core: numpy, pandas, scipy, scikit-learn, scikit-image
pip install -e ".[images]"  # also torch / open_clip / transformers, for image encoders
pip install -e ".[dev]"     # pytest
```

Python 3.9+. Results were produced with scikit-learn 1.6.1; other versions can change
tree-model outputs in late decimals.

## Quickstart

```bash
nuisauc list                         # experiments, where they appear, what data they need
nuisauc show family_i                # description, outputs, default configuration
nuisauc run theory_law family_i      # write tables to runs/ (or --out DIR)
nuisauc verify theory_law family_i   # compare runs/ with the reference tables in results/
nuisauc audit --manuscript PATH/main.tex  # check the paper's numbers against results/
nuisauc figures                      # draw Figure 1 from the tables
```

Override configuration with `--set key=value` (repeatable) or `--config file.yaml`;
the default for every experiment is in `configs/<experiment>.yaml`.

## Layout

```
src/nuisauc/
  cli.py            command-line interface
  estimand.py       adjusted AUC: quantile strata, pair weights, conditional_auc
  checks.py         overlap screen and self-stratified control
  propensity.py     nuisance models C1 (logistic), C2 (boosted trees), C1e, C2r
  theory/           correlation law and quadrature; Monte Carlo checks
  simulate/         Gaussian Families I-III; semi-synthetic labels on real descriptors
  data/             corpora, descriptors, caches, resampling groups, MedMNIST
  detectors/        BiomedCLIP / PLIP / QuiltNet zero-shot scores, self-training, probes
  diagnostics/      balance, calibration, held-out (split-family) tests
  inference/        cluster bootstrap, label-fit selection of the model class
  experiments/      one module per experiment, registered in experiments/__init__.py
  audit.py          manuscript-linked number audit
  figures.py        Figure 1
  paths.py seeds.py io.py
configs/            default configuration of each experiment (YAML)
results/            reference result tables (the tables behind every number in the paper)
tests/              unit, contract and end-to-end tests
```

## Data

Experiments declare what they need (`nuisauc list`, column `needs`):

- `none` — Gaussian simulations and quadrature; run from code alone.
- `cache` — per-tile inputs derived from images (labels, colour/focus/texture
  descriptors, frequency and morphology features, encoder embeddings), read from
  `$NUISAUC_CACHE` (default `data/cache`).
- `raw` — raw images, read from `$NUISAUC_RAW` (default `data/raw`; layout in
  `src/nuisauc/data/corpora.py`). Missing cache files are rebuilt from raw images
  on first use, with encoder weights from `$NUISAUC_MODELS`.

This repository includes aggregate reference tables. Per-tile arrays and
raw images are excluded. With independently obtained inputs, the commands below
run the PCam and NIH malaria comparisons.

~~~bash
nuisauc run ties --set "corpora=[pcam, malaria]"
nuisauc run bins_balance --set "corpora=[pcam, malaria]"
~~~

The AFB corpus is private. The Kaggle and Wellgen inputs were derived from local
crops whose source versions were not retained, so a rebuild from the public releases
gives comparable but not identical inputs.

## Reproducibility

- Seeds are deterministic (`nuisauc.seeds`, scheme `sha256-v1`). Each run writes a
  `<experiment>.provenance.json` with the configuration, package versions and input
  hashes.
- `nuisauc verify` compares new tables with `results/` (numeric tolerance 1e-9).
  Every experiment was rerun through this package and matched its reference table;
  as described in the manuscript.
- `nuisauc audit` recomputes the paper's headline numbers from `results/`, formats
  them as printed, and checks that each sentence or table row occurs in the paper's LaTeX source
  (`--manuscript`, the path to `main.tex`; `\input` files are followed).
- `pytest` runs the unit and contract tests; `pytest -m slow` reruns the fast
  experiments end to end and verifies them.

## Experiments and where they appear

| paper | experiments |
|---|---|
| §3, Table 1, Figure 1 (law vs simulation) | `theory_law` |
| §3, Table 2 (three regimes on real descriptors) | `semisynthetic_regimes`, `oracle_bins` |
| §4 (checks vs misspecification) | `family_i`, `families_ii_iii`, `cutoff_sweep`, `overlap_medmnist`, `semisynthetic_checks` |
| §5, Table 4 (same features, two answers) | `realdata_panel`, `realdata_screen`, `realdata_uncertainty` |
| §5, Table 5 and Table 9 (bins, balance) | `bins_balance` |
| §5 colour shift | `colour_shift` |
| §5 enrichment | `realdata_enrichment` |
| App. A (proofs, Monte Carlo checks) | `theory_montecarlo` |
| App. B (Families II-III, grid, bins) | `law_grid`, `control_bins`, `families_ii_iii` |
| App. D (held-out tests) | `heldout_statistic`, `heldout_specificity`, `heldout_hob`, `heldout_lod`, `heldout_null`, `heldout_attribution` |
| App. E (selection rule) | `selection_validation`, `selection_cluster`, `selection_replication`, `selection_linear` |
| App. F (uncertainty, learners, ties) | `realdata_learners`, `realdata_uncertainty`, `ties` |
| App. H (deterministic reruns) | `semisynthetic_checks`, `semisynthetic_sensitivity` |

`nuisauc list` prints the same mapping per experiment.
