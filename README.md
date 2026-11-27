# Can Lagged Yield Changes Beat a Random Walk?

ESE 5420 — Final Project  
Jonah Engelmann, Leo Gordon, Malcolm Siljehag Alencar  
November 2026

## Layout

```
data/           Treasury input data
src/            analysis.py (experiment) and render_report.py (tables, figure, LaTeX)
results/        experiment output (results_verified.json)
paper/          LaTeX sources: templates, NeurIPS style file, figures/
deliverables/   what has been handed in: baseline.txt, report.pdf, draft.pdf
```

- `PREREGISTRATION.md`: the primary comparison and full protocol for the 2024–2025 test, committed before the test was scored.
- `src/analysis.py`: the 2023 study: expanding-window validation, OLS, kNN and training-mean forecasts, 2023 evaluation, and bootstrap intervals.
- `src/experiments.py`: the final experiments: 10-year lags vs rich features × linear vs flexible models, tuned on 2021–2023 quarters and tested once on 2024–2025.
- `src/knn.py`: kNN that averages over days tied at the k-th distance, so results do not depend on row order or library version.
- `src/render_report.py`: fills a LaTeX template from the results JSON, draws the quarterly validation figure, and optionally compiles the PDF.
- `paper/report_template.tex`, `paper/draft_template.tex`: report and early-draft (NeurIPS 2026 style) sources, with `__PLACEHOLDER__` fields for results.
- `paper/figures/`: quarterly validation figures (`validation_rmse_draft.*` omits kNN).
- Filled `.tex` files and compiled PDFs are written to `paper/build/`, which is not tracked.

## Reproduction

Requires Python 3.10+, `pdflatex`, and standard LaTeX packages (including `newtxtext`, `newtxmath`, `microtype`, `graphicx`, and `booktabs`). Run from the repository root.

```bash
python -m pip install -r requirements.txt
python src/analysis.py --out results/results_verified.json
python src/render_report.py --compile
```

Final experiments (validation by default; `--score-test` scores 2024–2025 and requires `PREREGISTRATION.md` to be committed):

```bash
python src/experiments.py --compare-starts
python src/experiments.py --train-start 2020 --score-test
```

To build the early draft as submitted (it withheld kNN numbers, which `src/knn.py` now makes deterministic):

```bash
python src/render_report.py --template paper/draft_template.tex --out paper/build/draft.tex --figure validation_rmse_draft --no-knn --compile
```

The 2023 data are excluded from hyperparameter selection. Validation consists of eight expanding-window quarterly splits in 2021–2022. For linear regression, lag lengths are 1–5; for kNN, lag lengths are 1–5 and neighbor counts are 3, 5, 10, 20, and 40. kNN scaling is fitted separately in each training window. Confidence intervals use 10,000 paired circular moving-block bootstrap samples of length 20, with seed 5420. Observed 2023 lag values enter next-day forecasts but are never used to refit the models.

Data: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates, 2020–2023, https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-2020-2023.csv
