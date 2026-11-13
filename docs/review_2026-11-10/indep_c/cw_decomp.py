import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/tmp/e2e')
from engine.backtest import clark_west, diebold_mariano
def one(mode, stem, win):
    r=json.load(open(f'results/{mode}/{stem}.json')); d=pd.read_parquet(f'results/{mode}/{stem}_daily.parquet')
    ev=r.get('evaluation_period'); k=r['config']['target']['k']
    full=d.copy()
    # prevailing-mean benchmark: mean of the last `win` targets that were already realised at decision time t (target at s realised by s+1+k in trading mode)
    lagn = (1+k) if mode=='trading' else k
    pm = full['target'].shift(lagn).rolling(win, min_periods=min(win,30)).mean()
    e=full.loc[ev[0]:ev[1]].dropna(subset=['prediction','target']); y=e['target'].values; f=e['prediction'].values; b=pm.reindex(e.index).values
    ok=np.isfinite(b)
    cw0=clark_west(y,f,0.0,h=k); cwm=clark_west(y[ok],f[ok],b[ok],h=k)
    a=2*y*f; n=len(y)
    print(f"{stem[:34]:34s} n={n} mean(y)={y.mean():+.5f} mean(f)={f.mean():+.5f} sd(y)={y.std():.4f} sd(f)={f.std():.4f} corr(y,f)={np.corrcoef(y,f)[0,1]:+.4f}")
    print(f"    CW vs 0: t={cw0[0]:.2f} p={cw0[1]:.4f} | share of mean(2yf) from 2*ybar*fbar = {2*y.mean()*f.mean()/a.mean():+.2f} | CW vs prevailing mean({win}): t={cwm[0]:.2f} p={cwm[1]:.4f} (n={ok.sum()}) | CW(bench=mean) for a pure prevailing-mean 'model' vs 0: t={clark_west(y[ok],b[ok],0.0,h=k)[0]:.2f}")
for s in ('04_ridge_A_v1107_6e2191440f16','07_lgbm_ABC_v1107_d7656ba811bf','08_lgbm_ABCD_v1107_3f5151af48a3','09_lstm_ABC_v1107_6c9f1ba66aa1','lstm_v1107_ccfc51bdb577'):
    one('trading',s,756)
for s in ('f11_arima_k1_v1107_29387d348071','f11_sarima_k1_v1107_fbc71152d943','f11_lstm_macro_k1_v1107_6bcd24db12da','f11_biva_macro_k1_v1107_5e664224b16d'):
    one('forecast',s,60)
