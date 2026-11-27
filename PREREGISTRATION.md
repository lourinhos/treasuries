# Pre-registration: 2024–2025 test

Written 27 November 2026, before any model is scored on 2024–2025. The yields for 2024–2025 have been downloaded and checked only for their shape (row counts, duplicate dates, missing values). `src/experiments.py --score-test` refuses to run until this file is committed, and the results record the commit.

## Primary comparison

**Lasso on the rich feature set against the zero-change forecast, pooled over 2024–2025.**

- Statistic: D = RMSE(lasso) − RMSE(zero change), in basis points, on the 499 daily changes of 2024 (250) and 2025 (249).
- Interval: 95% percentile interval from a paired circular moving-block bootstrap: 20-day blocks, 10,000 resamples, seed 5420. Both forecasts are scored on the same resampled days.
- Reading: the lasso **beats** the random walk if the whole interval is below 0. It **loses** if the whole interval is above 0. Otherwise the two **tie**.

## Secondary comparisons

These are reported, but none of them can overturn or replace the primary result.

- Every other model against zero change, pooled: training mean, OLS, kNN, random forest (10-year lags), ridge, kNN, random forest (rich features). That makes 7 secondary comparisons. Each gets a 95% interval and a Bonferroni interval at level 1 − 0.05/7.
- Each model in 2024 and 2025 separately.
- Each model against the training mean.
- Block-length sensitivity: 10- and 30-day blocks (5,000 resamples each).

## Data

- Treasury daily par yield curve: the 2020–2023 archive file plus the 2024 and 2025 files (`data/`).
- Trading days per year: 251, 251, 249, 250, 250 and 249. The early draft said 249 and 248 for 2024 and 2025; those counts were off by one because the files lack a trailing newline.
- Yields are in whole basis points, and changes are taken between consecutive trading days.

## Inputs

- **10-year lags:** the last *p* daily changes of the 10-year yield, with p ∈ {1, …, 5}.
- **Rich features:** the last *p* daily changes of the 2-, 10- and 30-year yields; the previous day's 2s10s and 10s30s spreads; and the standard deviation of the 10-year change over the previous 20 days.

Every input is known before the day being forecast.

## Models and grids

| Model | Inputs | Grid |
|---|---|---|
| Zero change (benchmark) | — | — |
| Training mean | — | — |
| OLS | 10-year lags | p |
| kNN | 10-year lags | p; k ∈ {3, 5, 10, 20, 40, 80, 160, 320} |
| Random forest | 10-year lags | p; leaf ∈ {10, 40, 160}; features per split ∈ {all, √n} |
| Ridge | rich | p; α ∈ 10^{0, 0.5, …, 5} |
| Lasso | rich | p; α ∈ 10^{−3, −2.75, …, 0.5} |
| kNN | rich | as kNN above |
| Random forest | rich | as random forest above |

Implementation details:
- Ridge and lasso standardize their inputs within each training window.
- kNN standardizes within each training window and averages over days tied at the k-th distance (`src/knn.py`). k is capped at the number of training days, which only affects k = 320 in the earliest folds.
- Each random forest is five forests of 100 trees with seeds 0–4, fixed in advance. Its forecast is the average of the five, and the spread across seeds is reported.

## Training start and validation

Training starts on **2 January 2020**. Validation uses the 12 expanding-window quarters 2021 Q1 – 2023 Q4: each quarter is predicted by a model fitted on every earlier day from 2020 on.

This was decided on validation data only. On the eight quarters 2022 Q1 – 2023 Q4, training from 2020 and from 2021 differ by at most 0.017 bp in mean RMSE for every model (`results/train_start_comparison.json`). Keeping 2020 keeps the draft's planned protocol and four extra validation quarters.

Selection rule: the lowest mean quarterly RMSE. Ties go to smaller p, then the simpler setting: larger penalty, larger leaf, smaller k.

### Settings selected on validation, fixed for the test (`results/validation_2020.json`)

| Model | Selected |
|---|---|
| Training mean | — |
| OLS | p = 1 |
| kNN | p = 2, k = 320 |
| Random forest | p = 2, leaf = 160, √n features |
| Ridge, rich | p = 5, α = 10,000 |
| **Lasso, rich (primary)** | **p = 5, α = 1.778** |
| kNN, rich | p = 3, k = 320 |
| Random forest, rich | p = 3, leaf = 160, √n features |

## Test protocol

- Each model is refitted once on 2020–2023 with the settings above and forecasts every day of 2024–2025 one step ahead.
- Lagged inputs use the observed earlier days, but no model is ever refitted on 2024–2025.
- The test is run once. Nothing above changes after it is scored.
- Any model or feature added later is labelled exploratory and kept out of the conclusions.
- The 2023 results stay in the paper as reported, from the original eight-quarter protocol.
