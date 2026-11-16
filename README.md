# Can Lagged Yield Changes Beat a Random Walk?

ESE 5420 — Final Project Report  
Jonah Engelmann, Leo Gordon, Malcolm Siljehag Alencar  
November 2026

## Files

- `draft.pdf`: early paper draft (NeurIPS 2026 style), generated from `draft_template.tex`.
- `draft_template.tex`, `draft.tex`: draft source template and its filled version.
- `neurips_2026.sty`: official NeurIPS 2026 style file.
- `validation_rmse_draft.pdf`, `validation_rmse_draft.png`: validation figure for the draft (zero change and OLS only).
- `report.pdf`: typeset report.
- `report.tex`: compiled LaTeX source.
- `report_template.tex`: source template with experiment-result placeholders.
- `analysis.py`: expanding-window validation, OLS and kNN fitting, 2023 evaluation, and bootstrap intervals.
- `render_report.py`: creates the results tables, quarterly validation figure, and report from `results_verified.json`.
- `results_verified.json`: detailed experiment results.
- `validation_rmse.pdf`, `validation_rmse.png`: quarterly validation results.
- `par-yield-curve-rates-2020-2023.csv`: Treasury input data.
- `requirements.txt`: Python dependencies.

## Reproduction

Requires Python 3.10+, `pdflatex`, and standard LaTeX packages (including `newtxtext`, `newtxmath`, `microtype`, `graphicx`, and `booktabs`).

```bash
python -m pip install -r requirements.txt
python analysis.py --csv par-yield-curve-rates-2020-2023.csv --out results_verified.json
python render_report.py --results results_verified.json --compile
```

To build the early draft (kNN numbers are withheld pending deterministic tie handling):

```bash
python render_report.py --results results_verified.json --template draft_template.tex --out draft.tex --figure validation_rmse_draft --no-knn --compile
```

The 2023 data are excluded from hyperparameter selection. Validation consists of eight expanding-window quarterly splits in 2021–2022. For linear regression, lag lengths are 1–5; for kNN, lag lengths are 1–5 and neighbor counts are 3, 5, 10, 20, and 40. kNN scaling is fitted separately in each training window. Confidence intervals use 10,000 paired circular moving-block bootstrap samples of length 20, with seed 5420. Observed 2023 lag values enter next-day forecasts but are never used to refit the models.

Data: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates, 2020–2023, https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-2020-2023.csv
