# Kiva Moral-Distance Prediction

Do various distances between a lender and a borrower (geographic, cultural, linguistic,
relational) predict which loans a Kiva lender chooses to fund? This project builds a
lender–loan choice dataset from Kiva snapshot data, joins it against a gravity-model
country-distance dataset (CEPII), other country-level distance sources, and NLP-derived features of loan narratives.
It then fits various models including elastic-net logistic regression, XGBoost, and Random Forest to rank the predictors and test our research question.

## Pipeline

Each stage is a `*_scripts/` directory of code paired with a data directory of the
same stage's inputs/outputs (data directories are gitignored — see
[Data availability](#data-availability)). Run in this order:

| # | Scripts | Reads | Writes | What it does |
|---|---------|-------|--------|---------------|
| 1 | [kiva_cleaning_scripts/](kiva_cleaning_scripts/) | `KivaMatlabData/*.mat` | `raw_csv/`, `cleaned_csv/` | Exports Kiva's MATLAB structs to CSV (`mat_to_csv.py`), builds `borrower_info.csv` / `lender_info.csv` / `lender_loan_funding.csv`, guesses lender gender from first name, converts MATLAB datenums to real dates. `null_analysis.py`, `prefunded_crosstab.py`, `borrower_analysis.py` are diagnostic/QA scripts, not part of the write path. |
| 2 | [distance_scripts/](distance_scripts/) | `distance_data/Gravity_V202211.csv`, `distance_data/Countries_V202211.csv` | `distance_data/` | Filters the CEPII gravity dataset to 2006–2013 and to Kiva-observed countries, matches Kiva's `nation_code` to the gravity dataset's ISO3/`country_id` (`match_countries.py`, with manual overrides for Kiva-nonstandard codes like South Sudan and Gaza). |
| 3 | [pairs_dataset_scripts/](pairs_dataset_scripts/) | `cleaned_csv/`, `distance_data/` | `cleaned_csv/pairs_dataset.parquet`, `cleaned_csv/country_pair_distances.parquet` | `build_pairs_dataset.py` is the core dataset builder: cleans the loan/lender tables, subsamples lenders (≥5 loans), does risk-set negative sampling, assembles the long lender×loan feature matrix, and adds lender-grouped / temporal train-test split columns. See its docstring for the full step-by-step spec. |
| 4 | [nlp_scripts/](nlp_scripts/) | `cleaned_csv/borrower_info.csv` | `nlp_output/` (gitignored) | Scores each loan's English description for sentiment and emotion (`transformers` pipelines), named entities (`spacy`), and readability (`textstat`). Built for an HPC GPU cluster — see [nlp_scripts/requirements.txt](nlp_scripts/requirements.txt) and the `run_*.sh` / `submit_*.sh` job scripts. Supports resuming an interrupted run. |
| 5 | [model_output/](model_output/) | `cleaned_csv/pairs_dataset_fixed.parquet` | `model_output/*.joblib`, `*.json`, `*.csv` | Elastic-net logistic regression on distance + loan + lender features, selected by grouped CV (`run_elasticnet.py`), plus a partner-fixed-effects robustness spec (`step7_partner_fe.py`), a cluster bootstrap for coefficient CIs (`step8_bootstrap.py`), and final output assembly (`step9_outputs.py`, `save_main_coefs.py`). Shared preprocessing lives in `common.py`. |
| 6 | [figures/](figures/) | `model_output/` results | `figures/*.png/.pdf/.svg` | `generate_figures.py` renders the poster figures (distance coefficients, pooled-vs-conditional model comparison, feature category composition, saturation curves, accuracy) purely from already-computed result files — no model fitting happens here. |

## Data availability

The majority of both raw and generated data was not committed to this repository due largely to size. It can be regnerated by downloading the Kiva platfrom data from FODOVA and the distance data.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers steps 1–3, 5, and 6 (data cleaning, distance matching,
modeling, figures). The NLP step (4) was run in a separate HPC/GPU environment —
see [nlp_scripts/requirements.txt](nlp_scripts/requirements.txt) for those
(unpinned) dependencies.