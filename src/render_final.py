#!/usr/bin/env python3
"""Tables and figures for the final draft, built only from the results files.

Reads results/results_verified.json (2023 study) and results/final_2020.json (final
study, scored once on 2024-2025). Writes LaTeX tabulars to paper/generated/ and figures
to paper/figures/. The paper's text and captions live in paper/main.tex.

Usage (from the repository root):
  python src/render_final.py
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experiments as E

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN, FIG = ROOT / 'paper' / 'generated', ROOT / 'paper' / 'figures'

INK, INK2, GRID = '#0b0b0b', '#52514e', '#dddddd'
C2024, C2025 = '#2a78d6', '#eb6834'          # categorical slots 1-2 (validated)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'stix'})

ORDER = ['Lasso, rich', 'Ridge, rich', 'OLS', 'kNN', 'kNN, rich', 'Random forest', 'Random forest, rich',
         'Training mean']
LABEL = {'Lasso, rich': 'Lasso', 'Ridge, rich': 'Ridge', 'OLS': 'OLS', 'kNN': '$k$NN', 'kNN, rich': '$k$NN',
         'Random forest': 'Random forest', 'Random forest, rich': 'Random forest', 'Training mean': 'Training mean'}
PLOT_LABEL = {'Lasso, rich': 'Lasso, rich (primary)', 'Ridge, rich': 'Ridge, rich', 'OLS': 'OLS, 10-yr lags',
              'kNN': 'kNN, 10-yr lags', 'kNN, rich': 'kNN, rich', 'Random forest': 'Random forest, 10-yr lags',
              'Random forest, rich': 'Random forest, rich', 'Training mean': 'Training mean'}


def inputs(name):
    return '---' if name == 'Training mean' else ('rich' if 'rich' in name else '10-yr lags')


def num(x, sign=False):
    if round(x, 3) == 0:
        return '$0.000$'                     # no sign on values that round to zero
    return f'${x:+.3f}$' if sign else f'${x:.3f}$'


def interval(ci):
    return f'$[{ci[0]:.3f}, {ci[1]:.3f}]$'


def pm(m, s, sign=False):
    return f'${m:+.3f} \\pm {s:.3f}$' if sign else f'${m:.3f} \\pm {s:.3f}$'


def setting(prm):
    parts = [f'$p={prm["p"]}$'] if 'p' in prm else []
    if 'k' in prm: parts.append(f'$k={prm["k"]}$')
    if 'alpha' in prm:
        a = prm['alpha']
        e = np.log10(a)
        parts.append(f'$\\alpha=10^{{{e:.0f}}}$' if abs(e - round(e)) < 1e-9 and a >= 10 else f'$\\alpha={a:.3g}$')
    if 'leaf' in prm: parts.append(f'leaf {prm["leaf"]}')
    if 'mf' in prm: parts.append('$\\sqrt{n}$ feat.' if prm['mf'] == 'sqrt' else 'all feat.')
    return ', '.join(parts)


def tabular(spec, header, rows):
    body = '\n'.join(r if r == '\\midrule' else r + r' \\' for r in rows)
    return f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n'


def main():
    old = json.loads((ROOT / 'results' / 'results_verified.json').read_text())
    new = json.loads((ROOT / 'results' / 'final_2020.json').read_text())
    GEN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    tables = {}

    # Table 1: data by year
    df = E.load(ROOT / 'data')
    roles = {2020: ('train', 'train'), 2021: ('validate', 'train, validate'), 2022: ('validate', 'train, validate'),
             2023: ('test', 'train, validate'), 2024: ('---', 'test'), 2025: ('---', 'test')}
    rows = []
    for y, g in df.groupby('year'):
        d = g.d10.dropna()
        rows.append(f'{y} & {len(g)} & {len(d)} & {num(np.sqrt(np.mean(d**2)))} & {roles[y][0]} & {roles[y][1]}')
    tables['tab_data'] = tabular('lrrrll', 'Year & Yield obs. & Changes & Zero-change RMSE (bp) & 2023 study & Final study', rows)

    # Table 2: 2023 study, validation over 8 quarters of 2021-2022
    zero = np.array([q['zero_rmse'] for q in old['validation_quarters']])
    def vrow(label, sett, quarters):
        q = np.array(quarters); d = q - zero
        return f'{label} & {sett} & {pm(q.mean(), q.std(ddof=1))} & {pm(d.mean(), d.std(ddof=1), True)} & {(d < 0).sum()}/8'
    rows = [f'Zero change (benchmark) & --- & {pm(zero.mean(), zero.std(ddof=1))} & --- & ---',
            vrow('Training mean', '---', old['validation']['Mean:p=1']['quarters'])]
    for p in range(1, 6):
        lab = 'OLS (selected)' if p == old['selected']['OLS']['p'] else 'OLS'
        rows.append(vrow(lab, f'$p={p}$', old['validation'][f'OLS:p={p}']['quarters']))
    k = old['selected']['kNN']
    rows.append(vrow('$k$NN (selected)', f'$p={k["p"]}$, $k={k["k"]}$', k['quarters']))
    tables['tab_cv2023'] = tabular('llccc', 'Forecast & Setting & RMSE (bp) & $D$ per quarter (bp) & Quarters won', rows)

    # Table 3: 2023 test
    t = old['test']
    rows = [f'Zero change (benchmark) & {num(t["OLS"]["zero_rmse"])} & --- & ---']
    for name, lab in (('Mean', 'Training mean'), ('OLS', f'OLS ($p={t["OLS"]["p"]}$)'),
                      ('kNN', f'$k$NN ($p={t["kNN"]["p"]}$, $k={t["kNN"]["k"]}$)')):
        rows.append(f'{lab} & {num(t[name]["model_rmse"])} & {num(t[name]["delta"], True)} & {interval(t[name]["ci"])}')
    tables['tab_test2023'] = tabular('lccc', 'Forecast & RMSE (bp) & $D$ (bp) & 95\\% interval for $D$ (bp)', rows)

    # Table 4: final study, validation over 12 quarters of 2021-2023
    sel, zv = new['selected'], new['zero_validation']
    nq = len(new['protocol']['validation_quarters'])
    def frow(name):
        s = sel[name]
        lab = LABEL[name] + (' (primary)' if name == E.PRIMARY else '')
        sett = '---' if name == 'Training mean' else setting(s['params'])
        return (f'{lab} & {inputs(name)} & {sett} & {s["n_candidates"]} & {pm(s["mean"], s["sd"])} & '
                f'{pm(s["paired_mean"], s["paired_sd"], True)} & {s["wins"]}/{nq}')
    rows = [f'Zero change (benchmark) & --- & --- & --- & {pm(zv["mean"], zv["sd"])} & --- & ---',
            frow('Training mean'), '\\midrule',
            frow('OLS'), frow('Ridge, rich'), frow('Lasso, rich'), '\\midrule',
            frow('kNN'), frow('Random forest'), frow('kNN, rich'), frow('Random forest, rich')]
    tables['tab_cvfinal'] = tabular('lllcccc', 'Model & Inputs & Selected setting & Cand. & RMSE (bp) & $D$ per quarter (bp) & Won', rows)

    # Table 5: final study, 2024-2025 test
    T = new['test']
    z = T['OLS']
    rows = [f'Zero change (benchmark) & --- & {num(z["zero_rmse"])} & --- & --- & --- & --- & ---']
    for name in ORDER:
        r, pl = T[name], T[name]['pooled']
        lab = LABEL[name] + (' (primary)' if name == E.PRIMARY else '')
        bonf = interval(pl['ci_bonf']) if 'ci_bonf' in pl else '---'
        row = (f'{lab} & {inputs(name)} & {num(r["rmse"])} & {num(pl["D"], True)} & {interval(pl["ci95"])} & {bonf} & '
               f'{num(r["2024"]["D"], True)} & {num(r["2025"]["D"], True)}')
        rows.append(f'\\textbf{{{lab}}}' + row[len(lab):] if name == E.PRIMARY else row)
        if name in ('Lasso, rich', 'OLS', 'Random forest, rich'):
            rows.append('\\midrule')
    tables['tab_testfinal'] = tabular('llcccccc', 'Model & Inputs & RMSE & $D$ & 95\\% interval & Bonferroni interval & $D_{2024}$ & $D_{2025}$', rows)

    # Appendix table: training from 2020 vs 2021, same eight validation quarters (2022-2023)
    cmp = json.loads((ROOT / 'results' / 'train_start_comparison.json').read_text())
    a, b = cmp['by_start']['2020'], cmp['by_start']['2021']
    rows = [f'Zero change (benchmark) & --- & {num(a["zero"]["mean"])} & {num(b["zero"]["mean"])} & --- & ---']
    for name in ['Training mean'] + ORDER[:-1]:
        x, y = a['best'][name], b['best'][name]
        rows.append(f'{LABEL[name]} & {inputs(name)} & {num(x["mean"])} & {num(y["mean"])} & {num(y["mean"] - x["mean"], True)} '
                    f'& {num(x["paired_mean"], True)} / {num(y["paired_mean"], True)}')
    tables['tab_trainstart'] = tabular('llcccc', 'Model & Inputs & From 2020 & From 2021 & Difference & $D$ vs zero (2020 / 2021)', rows)

    for name, body in tables.items():
        (GEN / f'{name}.tex').write_text(f'% Generated by src/render_final.py from results/*.json. Do not edit by hand.\n{body}')

    # Figure 1: validation by quarter (12 quarters, 2021-2023)
    v = new['validation']
    qs = new['protocol']['validation_quarters']
    zq = np.array(v['zero_quarters'])
    dq = {n: np.array(v['validation'][n][E.key(sel[n]['params'])]['quarters']) - zq for n in sel}
    allq = np.array(list(dq.values()))
    x = np.arange(len(qs))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.5, 3.5), sharex=True, gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.12})
    a1.plot(x, zq, color=INK, lw=1.4, marker='o', ms=4)
    a1.set_ylabel('Zero-change\nRMSE (bp)')
    a1.set_ylim(0, 9.5)
    a2.fill_between(x, allq.min(0), allq.max(0), color='#c8c7c2', alpha=0.6, lw=0, label='Range of all eight models')
    a2.plot(x, dq['OLS'], color=C2025, lw=1.4, marker='s', ms=4, label='OLS, 10-yr lags')
    a2.plot(x, dq['Lasso, rich'], color=C2024, lw=1.4, marker='o', ms=4, label='Lasso, rich (primary)')
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_ylabel('$D$ per quarter (bp)\nmodel $-$ zero')
    a2.legend(loc='upper left', fontsize=7.5, frameon=False, ncol=3, handlelength=1.6)
    lim = np.abs(allq).max() * 1.35
    a2.set_ylim(-lim, lim * 1.25)
    a2.set_xticks(x, [q.replace('Q', '\nQ') for q in qs])
    for a in (a1, a2):
        a.spines[['top', 'right']].set_visible(False)
        a.grid(axis='y', color=GRID, lw=0.5)
        a.set_axisbelow(True)
    fig.align_ylabels((a1, a2))
    fig.savefig(FIG / 'validation_quarters.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'validation_quarters.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 2 (key figure): D with 95% intervals, per year and pooled
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ys = np.arange(len(ORDER))[::-1].astype(float)
    ys[1:] -= 0.5                                   # gap below the primary comparison
    styles = [('2024', C2024, 'o', 4.5, 1.2, +0.22, '2024 ($n=250$)'),
              ('2025', C2025, 's', 4.5, 1.2, 0.0, '2025 ($n=249$)'),
              ('pooled', INK, 'D', 6.0, 2.0, -0.24, 'Pooled 2024–2025 ($n=499$)')]
    for yr, col, mk, ms, lw, off, lab in styles:
        for i, name in enumerate(ORDER):
            r = T[name][yr] if yr != 'pooled' else T[name]['pooled']
            ax.plot(r['ci95'], [ys[i] + off] * 2, color=col, lw=lw, solid_capstyle='round')
            ax.plot(r['D'], ys[i] + off, marker=mk, color=col, ms=ms, mec='white', mew=0.6, ls='none',
                    label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8, zorder=0)
    ax.set_yticks(ys, [PLOT_LABEL[n] for n in ORDER])
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.set_xlabel('$D$ = RMSE(model) $-$ RMSE(zero change), bp   (left of 0: model beats the random walk)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5, frameon=False, handlelength=1.2)
    fig.savefig(FIG / 'test_differences.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'test_differences.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Wrote', ', '.join(sorted(tables)), 'and figures validation_quarters, test_differences')


if __name__ == '__main__':
    main()
