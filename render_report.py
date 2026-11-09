#!/usr/bin/env python3
"""Render the report template and validation graph from analysis.py results JSON."""
import argparse,json,pathlib, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser()
ap.add_argument('--results',default='results_verified.json')
ap.add_argument('--compile',action='store_true')
args=ap.parse_args()
r=json.loads(pathlib.Path(args.results).read_text())
T=pathlib.Path('report_template.tex').read_text()
zero=[v['zero_rmse'] for v in r['validation_quarters']]
a=r['test']['OLS'];b=r['test']['kNN'];  va=r['selected']['OLS'];vb=r['selected']['kNN']
f=lambda x: f'{x:.3f}'
vars={
 'ZERO_TEST':f(a['zero_rmse']), 'OLS_TEST':f(a['model_rmse']), 'KNN_TEST':f(b['model_rmse']),
 'OLS_DELTA':f(a['delta']), 'KNN_DELTA':f(b['delta']),
 'OLS_LO':f(a['ci'][0]),'OLS_HI':f(a['ci'][1]),'KNN_LO':f(b['ci'][0]),'KNN_HI':f(b['ci'][1]),
 'ZERO_CV':f(np.mean(zero)),'ZERO_CV_SD':f(np.std(zero,ddof=1)),
 'OLS_P':str(va['p']), 'KNN_P':str(vb['p']), 'KNN_K':str(vb['k']),
 'KNN_CV':f(vb['mean']),'KNN_CV_SD':f(vb['sd']),
 'OLS10LO':f(a['sensitivity']['10'][0]),'OLS10HI':f(a['sensitivity']['10'][1]),
 'OLS30LO':f(a['sensitivity']['30'][0]),'OLS30HI':f(a['sensitivity']['30'][1]),
 'KNN10LO':f(b['sensitivity']['10'][0]),'KNN10HI':f(b['sensitivity']['10'][1]),
 'KNN30LO':f(b['sensitivity']['30'][0]),'KNN30HI':f(b['sensitivity']['30'][1]),
}
for p in range(1,6):
 z=r['validation'][f'OLS:p={p}'];vars[f'OLS{p}']=f(z['mean']);vars[f'OLS{p}SD']=f(z['sd'])
for year in range(2020,2024):vars[f'Y{year}']=f(r['yearly_change'][str(year)]['rmse_zero'])
for key,val in vars.items():T=T.replace('__'+key+'__',val)
assert '__' not in T, [x for x in T.split() if '__' in x]
pathlib.Path('report.tex').write_text(T)
fig,ax=plt.subplots(figsize=(6.8,2.55))
xs=np.arange(8)
ax.plot(xs, zero, color='#111111', linewidth=1.7, linestyle='--', marker='o', markersize=4, label='Zero change')
ax.plot(xs, va['quarters'], color='#666666', linewidth=1.4, marker='s', markersize=4, label=f'OLS, p={va["p"]}')
ax.plot(xs, vb['quarters'], color='#999999', linewidth=1.4, marker='^', markersize=4, label=f'kNN, p={vb["p"]}, k={vb["k"]}')
ax.set_xticks(xs, [v['quarter'].replace('202','2').replace('Q',' Q') for v in r['validation_quarters']],rotation=0)
ax.set_ylabel('RMSE (bp)')
ax.set_ylim(2.75,9.3)
ax.spines[['right','top']].set_visible(False)
ax.grid(axis='y',color='#dddddd',linewidth=.55)
ax.legend(loc='upper left',fontsize=8,frameon=False,ncol=3)
fig.tight_layout(pad=0.7)
fig.savefig('validation_rmse.pdf',bbox_inches='tight')
fig.savefig('validation_rmse.png',dpi=160,bbox_inches='tight')
plt.close(fig)
if args.compile:
 for i in range(2):
  result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex'],text=True,capture_output=True)
  if result.returncode: print(result.stdout[-5000:]);raise SystemExit(result.returncode)
 print('Created report.pdf')
