#!/usr/bin/env python3
"""Reproducible next-day Treasury yield-change forecast experiment.

Usage:
  python analysis.py --csv par-yield-curve-rates-2020-2023.csv --out results_verified.json

The 2023 test year is not used in model tuning; only earlier observations are
used as lagged features for later 2023 forecasts.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-2020-2023.csv'
def from_csv(path):
    df=pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns or '10 yr' not in df.columns:
        raise ValueError(f'Missing expected Treasury columns: {df.columns.to_list()}')
    df['date']=pd.to_datetime(df['date'],format='%m/%d/%Y')
    assert not df['date'].duplicated().any(), 'Duplicate Treasury dates'
    df['yield_bp'] = 100*pd.to_numeric(df['10 yr'], errors='coerce')
    df=df.sort_values('date').dropna(subset=['yield_bp'])
    df=df.loc[df['date'].dt.year.between(2020,2023),['date','yield_bp']].reset_index(drop=True)
    df['year']=df.date.dt.year
    df['quarter']=df.date.dt.quarter
    return df

def design(df,p):
    z=df.copy()
    z['dy'] = z['yield_bp'].diff()
    for j in range(1,p+1):
        z[f'lag{j}']=z.dy.shift(j)
    z=z.dropna(subset=['dy']+[f'lag{j}' for j in range(1,p+1)]).copy()
    return z, [f'lag{j}' for j in range(1,p+1)]

def rmse(y,yhat):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(yhat))**2)))

def model_fit(name,k=None):
    if name=='OLS': return LinearRegression()
    if name=='kNN': return make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=k, weights='uniform'))
    raise ValueError(name)

def paired_block_ci(y,yhat,block=20,B=10000,seed=5420):
    """Delta=RMSE(model)-RMSE(zero); same circular blocks for paired errors."""
    y=np.asarray(y,float); yh=np.asarray(yhat,float)
    n=len(y); base=rmse(y,0.); diff=rmse(y,yh)-base
    rng=np.random.default_rng(seed)
    offset=np.arange(block,dtype=int)
    results=np.empty(B)
    for i in range(B):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+offset[None,:]) % n).ravel()[:n]
        results[i]=rmse(y[idx],yh[idx])-rmse(y[idx],0.)
    return {'delta':diff,'ci':np.quantile(results,[.025,.975]).tolist(),
            'model_rmse':rmse(y,yh),'zero_rmse':base,'block':block,'B':B}

def evaluate(df):
    counts=df.groupby('year').size().to_dict()
    assert counts == {2020:251,2021:251,2022:249,2023:250}, counts
    quarters=[(y,q) for y in (2021,2022) for q in (1,2,3,4)]
    z1,cols=design(df,1)
    quarters_result=[]
    for yr,q in quarters:
        z=z1[(z1.year==yr)&(z1.quarter==q)]
        quarters_result.append({'quarter':f'{yr}Q{q}','n':len(z),'zero_rmse':rmse(z.dy,0)})
    params={}
    # Tune each lag p (OLS and kNN) strictly on the eight within-2020--22 quarters.
    for p in range(1,6):
        z,cols=design(df,p)
        for name,k in [('OLS',None)]+[('kNN',j) for j in (3,5,10,20,40)]:
            vals=[]
            for yr,q in quarters:
                first=(z.year < yr)|((z.year==yr)&(z.quarter<q))
                nextq=(z.year==yr)&(z.quarter==q)
                tr,va=z.loc[first],z.loc[nextq]
                assert tr.year.max()<=2022 and va.year.max()<=2022
                model=model_fit(name,k)
                model.fit(tr[cols],tr.dy)
                vals.append(rmse(va.dy,model.predict(va[cols])))
            params[f'{name}:p={p}'+(f':k={k}' if k else '')] = {
                'name':name,'p':p,'k':k,'mean':np.mean(vals).item(),
                'sd':np.std(vals,ddof=1).item(),'quarters':vals}
    # select mean quarterly RMSE; break ties in favor of smaller p and smaller k.
    best={}
    for name in ('OLS','kNN'):
        candidates=[v for v in params.values() if v['name']==name]
        best[name]=min(candidates,key=lambda x:(x['mean'],x['p'],x['k'] or 0))
    # Only after all tuning is fixed, fit all 2020--22 and evaluate 2023 once.
    test={}
    for name in ('OLS','kNN'):
        p=best[name]['p']; z,cols=design(df,p)
        tr=z[z.year<=2022]; te=z[z.year==2023]
        m=model_fit(name,best[name]['k'])
        m.fit(tr[cols],tr.dy)
        pred=m.predict(te[cols])
        res=paired_block_ci(te.dy,pred)
        res['p']=p; res['k']=best[name]['k']; res['n_test']=len(te)
        res['sensitivity']={str(b):paired_block_ci(te.dy,pred,block=b,B=5000)['ci'] for b in (10,30)}
        if name=='OLS':
            res['coef']=m.coef_.tolist(); res['intercept']=float(m.intercept_)
        test[name]=res
    # Include volatility by year of the actual changes.
    d=df.copy();d['dy']=d.yield_bp.diff()
    volatility={str(y):{'sd':float(s.dy.std()),'rmse_zero':rmse(s.dy.dropna(),0)}
                for y,s in d.groupby('year')}
    return {'N':counts,'validation_quarters':quarters_result,'validation':params,
            'selected':best,'test':test,'yearly_change':volatility,
            'source':'10-year UST daily par yield; 2020--2023'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',required=True,help='Original Treasury CSV file path')
    ap.add_argument('--out',default='results.json')
    args=ap.parse_args()
    df=from_csv(args.csv)
    result=evaluate(df)
    result['input_is_transcription']=False
    pathlib.Path(args.out).write_text(json.dumps(result,indent=2))
    print('By year:',result['N'])
    print('Validation zero:', [(r['quarter'],round(r['zero_rmse'],2)) for r in result['validation_quarters']])
    for p in range(1,6):
        r=result['validation'][f'OLS:p={p}']
        print(f'OLS p={p}: {r["mean"]:.3f} +/- {r["sd"]:.3f}')
    for name in ('OLS','kNN'):
        p=result['selected'][name]
        r=result['test'][name]
        print(f'{name} selected p={p["p"]} k={p["k"]} CV={p["mean"]:.3f}; '
              f'2023 RMSE={r["model_rmse"]:.3f}, zero={r["zero_rmse"]:.3f}, '
              f'Delta={r["delta"]:.3f}, CI={r["ci"]}')

if __name__=='__main__':main()
