#!/usr/bin/env python3
"""Final experiments: does more information or more flexibility beat the random walk?

A two-by-two design, tuned on expanding-window quarters and tested once on 2024-2025:

                     linear              flexible
  10-year lags only  OLS                 kNN, random forest
  rich features      ridge, lasso        kNN, random forest

plus two references: the zero-change forecast (the benchmark) and the training mean.

By default only the validation runs. The 2024-2025 test is scored with --score-test,
which refuses to run until PREREGISTRATION.md is committed.

Usage (from the repository root):
  python src/experiments.py --compare-starts            # validation: train from 2020 vs 2021
  python src/experiments.py --train-start 2020          # validation for the chosen protocol
  python src/experiments.py --train-start 2020 --score-test
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from knn import TieAwareKNN

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = ('par-yield-curve-rates-2020-2023.csv', 'par-yield-curve-rates-2024.csv',
         'par-yield-curve-rates-2025.csv')
EXPECTED = {2020: 251, 2021: 251, 2022: 249, 2023: 250, 2024: 250, 2025: 249}
LAST_TRAIN_YEAR = 2023
TEST_YEARS = (2024, 2025)
PRIMARY = 'Lasso, rich'
SEEDS = (0, 1, 2, 3, 4)              # random-forest seeds, fixed in advance
BLOCK, B, BOOT_SEED = 20, 10_000, 5420
SENS_BLOCKS, SENS_B = (10, 30), 5_000

LAGS = (1, 2, 3, 4, 5)
KS = (3, 5, 10, 20, 40, 80, 160, 320)
RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(0, 5, 11))
LASSO_ALPHAS = tuple(float(a) for a in np.logspace(-3, 0.5, 15))
LEAVES = (10, 40, 160)
RF_TREES = 100                       # per seed; the forecast averages all five seeds


# ---------------------------------------------------------------- data and features
def load(data_dir):
    frames = []
    for f in FILES:
        d = pd.read_csv(data_dir / f)
        d.columns = d.columns.str.strip().str.lower()
        d['date'] = pd.to_datetime(d['date'], format='%m/%d/%Y')
        frames.append(d[['date', '2 yr', '10 yr', '30 yr']])
    df = pd.concat(frames).sort_values('date').reset_index(drop=True)
    assert not df.date.duplicated().any(), 'duplicate dates'
    for col, name in (('2 yr', '2'), ('10 yr', '10'), ('30 yr', '30')):
        # Quoted to two decimals, so yields are whole basis points; round away float noise.
        df[f'r{name}'] = (100 * pd.to_numeric(df[col])).round()
        df[f'd{name}'] = df[f'r{name}'].diff()
    assert df[['r2', 'r10', 'r30']].notna().all().all(), 'missing yields'
    df['year'], df['quarter'] = df.date.dt.year, df.date.dt.quarter
    counts = df.groupby('year').size().to_dict()
    assert counts == EXPECTED, counts
    return df


def features(df, kind, p):
    """Inputs known before day t's yield is published. Returns X, y, row metadata."""
    X = pd.DataFrame(index=df.index)
    for j in range(1, p + 1):
        X[f'd10_lag{j}'] = df.d10.shift(j)
        if kind == 'rich':
            X[f'd2_lag{j}'] = df.d2.shift(j)
            X[f'd30_lag{j}'] = df.d30.shift(j)
    if kind == 'rich':
        X['spread_2s10s_lag1'] = (df.r10 - df.r2).shift(1)
        X['spread_10s30s_lag1'] = (df.r30 - df.r10).shift(1)
        X['vol20_lag1'] = df.d10.rolling(20).std().shift(1)   # days t-20 .. t-1
    ok = X.notna().all(axis=1) & df.d10.notna()
    return X[ok].to_numpy(float), df.d10[ok].to_numpy(float), df.loc[ok, ['date', 'year', 'quarter']]


# ---------------------------------------------------------------- models
def _rf_grid(kind):
    grid = []
    for p in LAGS:
        n = p if kind == '10y' else 3 * p + 3
        for leaf in LEAVES:
            for mf in ('all', 'sqrt'):
                if mf == 'sqrt' and max(1, int(np.sqrt(n))) == n:
                    continue                      # same as 'all'
                grid.append({'p': p, 'leaf': leaf, 'mf': mf})
    return grid

# name: family, feature set, cell of the design, grid, tie-break (smaller preferred)
MODELS = {
    'Training mean':       ('mean', '10y', 'reference', [{'p': 1}], lambda q: (q['p'],)),
    'OLS':                 ('ols', '10y', 'linear', [{'p': p} for p in LAGS], lambda q: (q['p'],)),
    'kNN':                 ('knn', '10y', 'flexible', [{'p': p, 'k': k} for p in LAGS for k in KS],
                            lambda q: (q['p'], q['k'])),
    'Random forest':       ('rf', '10y', 'flexible', _rf_grid('10y'),
                            lambda q: (q['p'], -q['leaf'], q['mf'] == 'all')),
    'Ridge, rich':         ('ridge', 'rich', 'linear', [{'p': p, 'alpha': a} for p in LAGS for a in RIDGE_ALPHAS],
                            lambda q: (q['p'], -q['alpha'])),
    'Lasso, rich':         ('lasso', 'rich', 'linear', [{'p': p, 'alpha': a} for p in LAGS for a in LASSO_ALPHAS],
                            lambda q: (q['p'], -q['alpha'])),
    'kNN, rich':           ('knn', 'rich', 'flexible', [{'p': p, 'k': k} for p in LAGS for k in KS],
                            lambda q: (q['p'], q['k'])),
    'Random forest, rich': ('rf', 'rich', 'flexible', _rf_grid('rich'),
                            lambda q: (q['p'], -q['leaf'], q['mf'] == 'all')),
}


def make(family, prm, seed=0):
    if family == 'mean':  return DummyRegressor(strategy='mean')
    if family == 'ols':   return LinearRegression()
    # Standardization is part of each pipeline, so it is fitted inside each training window.
    if family == 'ridge': return make_pipeline(StandardScaler(), Ridge(alpha=prm['alpha']))
    if family == 'lasso': return make_pipeline(StandardScaler(), Lasso(alpha=prm['alpha'], max_iter=200_000))
    if family == 'knn':   return TieAwareKNN(prm['k'])
    if family == 'rf':    return RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=prm['leaf'],
                                                       max_features=1.0 if prm['mf'] == 'all' else 'sqrt',
                                                       random_state=seed, n_jobs=1)
    raise ValueError(family)


def fit_predict(family, prm, Xtr, ytr, Xte):
    """Forecasts for Xte; a random forest returns the mean over the five fixed seeds,
    plus each seed's own forecasts."""
    if family == 'knn':
        # Early folds can have fewer training days than k; k = n is the training mean.
        prm = {**prm, 'k': min(prm['k'], len(ytr))}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        if family != 'rf':
            return make(family, prm).fit(Xtr, ytr).predict(Xte), None
        per_seed = np.array([make(family, prm, s).fit(Xtr, ytr).predict(Xte) for s in SEEDS])
        return per_seed.mean(axis=0), per_seed


def key(prm):
    return ','.join(f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}' for k, v in prm.items())


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


# ---------------------------------------------------------------- validation
def quarters_for(train_start):
    first = train_start + 1          # one full year of training before the first fold
    return [(y, q) for y in range(first, LAST_TRAIN_YEAR + 1) for q in (1, 2, 3, 4)]


def _fold_scores(family, prm, X, y, year, quarter, quarters, train_start):
    """Expanding window: each quarter is predicted by a model fitted on every earlier day."""
    out, sizes = [], []
    for yr, q in quarters:
        tr = (year >= train_start) & ((year < yr) | ((year == yr) & (quarter < q)))
        va = (year == yr) & (quarter == q)
        assert year[tr].max() <= LAST_TRAIN_YEAR and year[va].max() <= LAST_TRAIN_YEAR
        pred, _ = fit_predict(family, prm, X[tr], y[tr], X[va])
        out.append(rmse(y[va], pred))
        sizes.append(int(va.sum()))
    return out, sizes


def validate(df, train_start, n_jobs=-1):
    quarters = quarters_for(train_start)
    cache = {}
    for kind in ('10y', 'rich'):
        for p in LAGS:
            X, y, meta = features(df, kind, p)
            cache[(kind, p)] = (X, y, meta.year.to_numpy(), meta.quarter.to_numpy())
    jobs = [(name, prm) for name, spec in MODELS.items() for prm in spec[3]]
    t0 = time.time()
    done = Parallel(n_jobs=n_jobs)(
        delayed(_fold_scores)(MODELS[n][0], p, *cache[(MODELS[n][1], p['p'])], quarters, train_start)
        for n, p in jobs)
    X, y, year, quarter = cache[('10y', 1)]
    zero = [rmse(y[(year == yr) & (quarter == q)], 0.0) for yr, q in quarters]
    sizes = [int(((year == yr) & (quarter == q)).sum()) for yr, q in quarters]
    val = {name: {} for name in MODELS}
    for (name, prm), (out, n) in zip(jobs, done):
        assert n == sizes, 'validation days differ across candidates'
        val[name][key(prm)] = {'params': prm, 'quarters': out}
    return {'train_start': train_start, 'quarters': [f'{y}Q{q}' for y, q in quarters],
            'zero_quarters': zero, 'validation': val, 'seconds': round(time.time() - t0, 1)}


def select(v, which=None):
    """Lowest mean quarterly RMSE (optionally over a subset of quarters); ties go to the
    simpler candidate. Adds paired per-quarter differences against zero change."""
    idx = list(range(len(v['quarters']))) if which is None else [v['quarters'].index(q) for q in which]
    zero = np.array(v['zero_quarters'])[idx]
    best = {}
    for name, cands in v['validation'].items():
        tie = MODELS[name][4]
        c = min(cands.values(), key=lambda c: (round(float(np.mean(np.array(c['quarters'])[idx])), 12), tie(c['params'])))
        q = np.array(c['quarters'])[idx]
        d = q - zero
        best[name] = {'params': c['params'], 'mean': float(q.mean()), 'sd': float(q.std(ddof=1)),
                      'paired_mean': float(d.mean()), 'paired_sd': float(d.std(ddof=1)),
                      'wins': int((d < 0).sum()), 'n_quarters': len(idx), 'n_candidates': len(cands)}
    return best, {'mean': float(zero.mean()), 'sd': float(zero.std(ddof=1))}


# ---------------------------------------------------------------- test (2024-2025)
def boot_index(n, block, b, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(b, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, -1)[:, :n]) % n


def paired(y, a, ref, block=BLOCK, b=B, seed=BOOT_SEED, bonferroni=None):
    """D = RMSE(a) - RMSE(ref) with circular block-bootstrap percentile intervals.
    The same seed gives the same resampled days for every model (paired across models)."""
    ea, er = (y - a) ** 2, (y - ref) ** 2
    idx = boot_index(len(y), block, b, seed)
    dist = np.sqrt(ea[idx].mean(axis=1)) - np.sqrt(er[idx].mean(axis=1))
    out = {'D': float(np.sqrt(ea.mean()) - np.sqrt(er.mean())), 'n': int(len(y)), 'block': block, 'B': b,
           'ci95': np.quantile(dist, [0.025, 0.975]).tolist()}
    if bonferroni:
        out['ci_bonf'] = np.quantile(dist, [(1 - bonferroni) / 2, (1 + bonferroni) / 2]).tolist()
    return out


def require_preregistration():
    prereg = ROOT / 'PREREGISTRATION.md'
    if not prereg.exists():
        raise SystemExit('PREREGISTRATION.md is missing: write and commit it before scoring the test.')
    def git(*a):
        return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    if not git('log', '-1', '--format=%H %cI', '--', 'PREREGISTRATION.md'):
        raise SystemExit('PREREGISTRATION.md is not committed: commit it before scoring the test.')
    if git('status', '--porcelain', '--', 'PREREGISTRATION.md', 'src'):
        raise SystemExit('PREREGISTRATION.md or src/ has uncommitted changes: commit them before scoring.')
    return git('log', '-1', '--format=%H %cI', '--', 'PREREGISTRATION.md')


def score_test(df, train_start, best):
    n_secondary = len(MODELS) - 1
    bonf = 1 - 0.05 / n_secondary
    years = {}
    preds = {}
    for name, sel in best.items():
        family, kind = MODELS[name][:2]
        X, y, meta = features(df, kind, sel['params']['p'])
        tr = ((meta.year >= train_start) & (meta.year <= LAST_TRAIN_YEAR)).to_numpy()
        te = meta.year.isin(TEST_YEARS).to_numpy()
        pred, per_seed = fit_predict(family, sel['params'], X[tr], y[tr], X[te])
        preds[name] = (pred, per_seed)
        years.setdefault('y', y[te]); years.setdefault('year', meta.year[te].to_numpy())
        assert np.array_equal(years['y'], y[te]), 'test days differ across models'
        if name == 'OLS' or name == PRIMARY:
            m = make(family, sel['params']).fit(X[tr], y[tr])
            lin = m[-1] if hasattr(m, 'steps') else m
            best[name]['fit'] = {'coef': lin.coef_.tolist(), 'intercept': float(lin.intercept_),
                                 'n_nonzero': int(np.sum(lin.coef_ != 0)), 'n_features': int(len(lin.coef_))}
        if name == 'Training mean':
            best[name]['fit'] = {'constant': float(pred[0])}
    y, yr = years['y'], years['year']
    zero = np.zeros_like(y)
    mean_pred = preds['Training mean'][0]
    test = {}
    for name, (pred, per_seed) in preds.items():
        r = {'pooled': paired(y, pred, zero, bonferroni=None if name == PRIMARY else bonf),
             'rmse': rmse(y, pred), 'zero_rmse': rmse(y, zero),
             'vs_training_mean': paired(y, pred, mean_pred)}
        for year in TEST_YEARS:
            m = yr == year
            r[str(year)] = paired(y[m], pred[m], zero[m]) | {'rmse': rmse(y[m], pred[m]), 'zero_rmse': rmse(y[m], 0.0)}
        r['block_sensitivity'] = {str(bl): paired(y, pred, zero, block=bl, b=SENS_B)['ci95'] for bl in SENS_BLOCKS}
        if per_seed is not None:
            ds = [rmse(y, s) - rmse(y, zero) for s in per_seed]
            r['per_seed_D'] = ds
        test[name] = r
    return test, {'primary': PRIMARY, 'n_secondary': n_secondary, 'bonferroni_level': bonf}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=ROOT / 'data', type=pathlib.Path)
    ap.add_argument('--train-start', type=int, choices=(2020, 2021), default=2020)
    ap.add_argument('--compare-starts', action='store_true', help='validation only: train from 2020 vs 2021')
    ap.add_argument('--score-test', action='store_true', help='score 2024-2025 once (needs committed PREREGISTRATION.md)')
    ap.add_argument('--jobs', type=int, default=-1)
    args = ap.parse_args()
    df = load(args.data)
    (ROOT / 'results').mkdir(exist_ok=True)

    if args.compare_starts:
        common = [f'{y}Q{q}' for y, q in quarters_for(2021)]
        rows = {}
        for start in (2020, 2021):
            v = validate(df, start, args.jobs)
            best, zero = select(v, common)
            rows[str(start)] = {'best': best, 'zero': zero, 'seconds': v['seconds']}
        out = ROOT / 'results' / 'train_start_comparison.json'
        out.write_text(json.dumps({'quarters': common, 'by_start': rows}, indent=2))
        print(f'Validation over {common[0]}-{common[-1]} ({len(common)} quarters); D = model - zero, mean of quarterly RMSE differences')
        print(f'{"model":22s} {"train from 2020":>26s} {"train from 2021":>26s}')
        for name in MODELS:
            a, b = rows['2020']['best'][name], rows['2021']['best'][name]
            fmt = lambda r: f'{r["mean"]:.3f} (D {r["paired_mean"]:+.3f}, wins {r["wins"]}/{r["n_quarters"]})'
            print(f'{name:22s} {fmt(a):>26s} {fmt(b):>26s}')
        print(f'{"Zero change":22s} {rows["2020"]["zero"]["mean"]:>26.3f} {rows["2021"]["zero"]["mean"]:>26.3f}')
        print('Wrote', out)
        return

    stamp = require_preregistration() if args.score_test else None
    v = validate(df, args.train_start, args.jobs)
    best, zero = select(v)
    result = {'protocol': {'train_start': args.train_start, 'validation_quarters': v['quarters'],
                           'test_years': list(TEST_YEARS), 'primary': PRIMARY, 'block': BLOCK, 'B': B,
                           'bootstrap_seed': BOOT_SEED, 'rf_seeds': list(SEEDS), 'rf_trees_per_seed': RF_TREES},
              'counts': {str(k): v_ for k, v_ in EXPECTED.items()},
              'validation': v, 'selected': best, 'zero_validation': zero}
    if args.score_test:
        result['preregistration_commit'] = stamp
        result['test'], result['multiplicity'] = score_test(df, args.train_start, best)
    out = ROOT / 'results' / (f'final_{args.train_start}.json' if args.score_test else f'validation_{args.train_start}.json')
    out.write_text(json.dumps(result, indent=2))
    print(f'Validation {v["quarters"][0]}-{v["quarters"][-1]}, train from {args.train_start} ({v["seconds"]}s)')
    print(f'{"Zero change":22s} {zero["mean"]:.3f} +/- {zero["sd"]:.3f}')
    for name, s in best.items():
        print(f'{name:22s} {s["mean"]:.3f} +/- {s["sd"]:.3f}  paired D {s["paired_mean"]:+.3f} +/- {s["paired_sd"]:.3f}'
              f'  wins {s["wins"]}/{s["n_quarters"]}  [{key(s["params"])}]')
    if args.score_test:
        print('\n2024-2025 test, D = RMSE(model) - RMSE(zero) in bp, 95% block-bootstrap interval')
        for name, r in result['test'].items():
            pl = r['pooled']
            print(f'{name:22s} pooled {pl["D"]:+.3f} [{pl["ci95"][0]:+.3f}, {pl["ci95"][1]:+.3f}]'
                  f'  2024 {r["2024"]["D"]:+.3f}  2025 {r["2025"]["D"]:+.3f}' + ('  <- primary' if name == PRIMARY else ''))
    print('Wrote', out)


if __name__ == '__main__':
    main()
