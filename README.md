# Can Lagged Yield Changes Beat a Random Walk?

ESE 5420 — Final Project  
Jonah Engelmann, Leo Gordon, Malcolm Siljehag Alencar  
November 2026

**Answer: no.** On the pre-registered 2024–2025 test, the lasso on rich cross-maturity inputs ties the zero-change forecast (RMSE difference +0.003 bp, 95% interval [−0.017, 0.021]), and so does every other model. See `paper/main.tex`.

## Layout

```
PREREGISTRATION.md   primary comparison and protocol, committed before the 2024–2025 test was scored
data/                Treasury par yield files: 2020–2023, 2024, 2025
src/                 experiment and rendering scripts
results/             experiment output (JSON)
paper/               the paper: main.tex, NeurIPS style file, generated/ tables, figures/
deliverables/        what has been handed in: baseline.txt, report.pdf, draft.pdf (early draft)
```

- `src/analysis.py`: the 2023 study, as designed in the early draft: OLS, kNN and training-mean forecasts tuned on the eight quarters of 2021–2022 and tested on 2023 → `results/results_verified.json`.
- `src/experiments.py`: the final study: 10-year lags vs rich features × linear vs flexible models, tuned on the twelve quarters of 2021–2023 and tested once on 2024–2025 → `results/validation_*.json`, `results/train_start_comparison.json`, `results/final_2020.json`.
- `src/knn.py`: kNN that averages over days tied at the k-th distance, so results do not depend on row order or library version.
- `src/render_final.py`: the paper's tables (`paper/generated/`) and figures (`paper/figures/`) from the results files.
- `src/render_report.py`, `paper/report_template.tex`, `paper/draft_template.tex`: rebuild the earlier report and early draft.
- Compiled PDFs and the Overleaf zip go to `paper/build/`, which is not tracked.

## Reproduction

Requires Python 3.10+ and the packages in `requirements.txt` (results were produced with Python 3.15, NumPy 2.5, pandas 3.1, scikit-learn 1.9, Matplotlib 3.11). Run from the repository root.

```bash
python -m pip install -r requirements.txt
python src/analysis.py --out results/results_verified.json
python src/experiments.py --compare-starts
python src/experiments.py --train-start 2020 --score-test
python src/render_final.py
```

`experiments.py` runs only the validation unless `--score-test` is given, and refuses to score the test unless `PREREGISTRATION.md` and `src/` are committed.

To build the paper, compile `paper/main.tex` with `pdflatex` (run from `paper/`), or upload `main.tex`, `neurips_2026.sty`, `generated/` and the two figures used (`figures/validation_quarters.pdf`, `figures/test_differences.pdf`) to Overleaf. The earlier documents rebuild with `python src/render_report.py --compile` (report) and `python src/render_report.py --template paper/draft_template.tex --out paper/build/draft.tex --figure validation_rmse_draft --no-knn --compile` (early draft).

## Protocol in brief

Training data start on 2 January 2020. Every model is tuned by expanding-window validation: each quarter is predicted by a model fitted on all earlier days, and the setting with the lowest mean quarterly RMSE is selected. The test years are never used for tuning, and the refitted models use observed test-period lags but are never refitted on them. Test intervals come from a paired circular moving-block bootstrap (20-day blocks, 10,000 resamples, seed 5420). Full details are in `PREREGISTRATION.md` and the paper.

Data: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates, https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve (2020–2023 archive file: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-2020-2023.csv)
