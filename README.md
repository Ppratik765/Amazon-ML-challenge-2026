# Business Entity Resolution Pipeline

Matches noisy business records from Source 2 and Source 3 against the deduplicated
Source 1 reference set (Amazon ML Challenge 2026). Pipeline: TF-IDF + dense-embedding
blocking → LightGBM/XGBoost(/CatBoost) ensemble → per-source threshold + match-cap
decision rule tuned directly against the F0.5 metric on a held-out split.

## Repository layout

```
<repo_root>/
├── dataset/                        # NOT included in this repo -- see "Data layout" below
│   ├── train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
│   └── test/{test_source1,test_source2,test_source3}.tsv
├── output/                         # created by pipeline.py
│   ├── matching_results.tsv        # scored on the leaderboard
│   └── candidate_pairs.tsv         # candidate set actually fed to the model
├── utils/validate_submission.py    # run before every submission
└── business_entity_resolution/
    ├── Documentation_template.md
    └── code/
        ├── requirements.txt
        ├── model.pkl                # created by train.py (gitignored)
        ├── thresholds.json          # created by train.py (gitignored)
        └── src/
            ├── preprocess.py        # name/address/country normalization
            ├── blocking.py          # TF-IDF + MiniLM candidate retrieval
            ├── features.py          # pairwise similarity feature extraction
            ├── model.py             # LightGBM+XGBoost(+CatBoost) ensemble
            ├── evaluate.py          # F0.5 metric + threshold/cap grid search
            ├── train.py             # fit model, tune thresholds, save artifacts
            └── pipeline.py          # inference: test data -> submission files
```

## Setup

Python 3.10+.

```bash
cd business_entity_resolution/code
python3 -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

The dense blocking/feature stage uses `sentence-transformers/all-MiniLM-L6-v2`
(Apache-2.0, ~22M parameters — comfortably inside the challenge's MIT/Apache-2.0,
≤8B-parameter model constraint). **The first run downloads its weights from
Hugging Face** (a few hundred MB) and caches them under `~/.cache/huggingface`;
after that it works offline. This is a one-time weights download, not a lookup of
any business data, so it doesn't conflict with the "no external data lookup" rule
— but it does mean the machine you first run on needs outbound internet access
once. If you're on a fully air-gapped box, pre-download the model elsewhere and
copy `~/.cache/huggingface` over.

## Data layout

This repo does not include `dataset/` (correctly gitignored — it's large and
provided separately by the challenge). Place the official files at:

```
<repo_root>/dataset/train/train_source1.tsv
<repo_root>/dataset/train/train_source2.tsv
<repo_root>/dataset/train/train_source3.tsv
<repo_root>/dataset/train/train_ground_truth.tsv
<repo_root>/dataset/test/test_source1.tsv
<repo_root>/dataset/test/test_source2.tsv
<repo_root>/dataset/test/test_source3.tsv
```

`train.py`/`pipeline.py` resolve `dataset/` and `output/` as `../../dataset/...`
and `../../output/...` **relative to your current working directory** — so run
them from `business_entity_resolution/code/` (as shown below), or override the
paths with the environment variables in the Configuration section.

## Running locally

```bash
cd business_entity_resolution/code

# 1. Train: builds candidates, extracts features, fits the ensemble, tunes
#    thresholds against a held-out split, saves model.pkl + thresholds.json.
python3 -m src.train

# 2. Inference: runs the same pipeline over the test set and writes
#    ../../output/matching_results.tsv and ../../output/candidate_pairs.tsv
python3 -m src.pipeline

# 3. Validate the output format before submitting anything
cd ../..
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

`train.py` prints diagnostics as it runs — read these before trusting the run:

```
Blocking recall ceiling: 0.XXXX (N/M true matches retrieved)   <- upper bound on achievable recall
Recall after per-source cap (30): 0.XXXX                       <- should be ~equal to the line above
Split: X train / Y val / Z holdout S1 entities
Best thresholds: S2>=0.XX, S3>=0.XX, cap_per_source=K -> grid-search F0.5=0.XXXX
Holdout macro F0.5 (reference implementation): 0.XXXX          <- your real, trusted number
```

If the recall ceiling is well below 1.0, no amount of model tuning downstream can
fix it — that's a blocking problem (raise `tfidf_top_k`/`dense_top_k` in
`BlockingEngine`, or loosen `tfidf_min_sim`/`dense_min_sim`), not a modeling one.

### Fast local iteration (no embeddings)

The MiniLM encode step is the slowest part of blocking/feature extraction and
isn't needed to sanity-check the rest of the pipeline (feature engineering,
training, threshold search, output format). Skip it with:

```bash
AMZ_ML_USE_DENSE=0 python3 -m src.train
AMZ_ML_USE_DENSE=0 python3 -m src.pipeline
```

This uses TF-IDF-only blocking, which is much faster but lower recall — use it
for quick correctness checks and debugging, not for your real/final run.

## Running on AWS

**Instance choice:** the boosters (LightGBM/XGBoost/CatBoost) are CPU-only and
parallelize across cores (`n_jobs=-1`), so they benefit from a high-vCPU
compute-optimized instance (e.g. `c6i.4xlarge` or larger). The one step that
meaningfully benefits from a GPU is the MiniLM encoding pass in blocking and
feature extraction — if source2/3 are large (the validator script's own comments
reference a full test set on the order of ~1.7M entities), a `g4dn.xlarge` or
similar will encode that in minutes instead of potentially hours on CPU. If
you're CPU-only, budget accordingly or reduce `dense_batch_size`/consider
disabling the dense stage for an initial pass.

**Memory:** blocking and feature extraction hold the candidate-pair table in
memory as a pandas DataFrame. At large scale, watch memory on the `unknown`/most
populous country bucket in blocking and on the full (pre-cap) candidate table.
`cap_candidates_per_source` (called right after blocking, before feature
extraction) trims this — raise or lower `MAX_CANDIDATES_PER_SOURCE` in
`train.py`/`pipeline.py` to trade recall against memory/compute.

**Steps:**

```bash
# On the AWS instance
git clone <your-repo-url>
cd Amazon-ML-challenge-2026
# upload/mount the dataset/ folder here (S3 sync, EBS volume, etc.)

cd business_entity_resolution/code
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# First run needs outbound internet once, to fetch the MiniLM weights (see Setup).
python3 -m src.train
python3 -m src.pipeline

cd ../..
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Given limited compute time, a reasonable order of operations:

1. Run once with `AMZ_ML_USE_DENSE=0` to confirm the whole pipeline finishes
   cleanly and check the blocking-recall-ceiling diagnostic cheaply.
2. If recall looks reasonable, run the real (dense-enabled) `train.py`, read the
   printed holdout F0.5 and thresholds before spending more compute.
3. Only then run `pipeline.py` over the full test set.

## Configuration reference

All of the following are optional environment variables; sane defaults match the
directory layout above.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `AMZ_ML_DATA_DIR` | `../../dataset/train` | `train.py` | Folder with the four train TSVs |
| `AMZ_ML_TEST_DIR` | `../../dataset/test` | `pipeline.py` | Folder with the three test TSVs |
| `AMZ_ML_OUT_DIR` | `../../output` | `pipeline.py` | Where `matching_results.tsv`/`candidate_pairs.tsv` are written |
| `AMZ_ML_USE_DENSE` | `1` | `train.py`, `pipeline.py` | `0` disables the MiniLM dense-embedding stage (TF-IDF-only, faster, lower recall) |
| `AMZ_ML_MAX_CAND_PER_SOURCE` | `30` | `pipeline.py` | Cap on candidates per (S1 entity, source) kept for scoring — mirrors `MAX_CANDIDATES_PER_SOURCE` in `train.py` (edit that constant directly to change the training-time cap) |

`model.pkl` and `thresholds.json` are written to `business_entity_resolution/code/`
and read from there by `pipeline.py` — no path configuration needed as long as
both scripts are run from that directory.

## Reproducibility note for the final submission zip

`requirements.txt` currently pins minimum versions known to work together, not
exact ones. Before packaging your final submission, freeze your actual working
environment so the grader can reproduce it exactly:

```bash
pip freeze > requirements.txt
```

## Troubleshooting

- **`OSError: We couldn't connect to 'https://huggingface.co'...`** — the machine
  has no internet access and the MiniLM weights aren't cached yet. See Setup.
- **CatBoost not installed** — it's optional; `model.py` skips it automatically
  and trains LightGBM+XGBoost only.
- **Out of memory during blocking** — lower `tfidf_top_k`/`dense_top_k` in
  `BlockingEngine`, or lower `MAX_CANDIDATES_PER_SOURCE`.
- **`RuntimeError: Could not load model/thresholds`** from `pipeline.py` —
  `train.py` hasn't been run yet (or didn't finish) in this directory.