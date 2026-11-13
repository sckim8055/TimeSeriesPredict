import sys, numpy as np, time
sys.path.insert(0,'/tmp/e2e')
from engine.backtest import clark_west, diebold_mariano
from scipy.stats import norm
def cw_var(adj, h, kind):
    n=len(adj); m=adj.mean(); dc=adj-m; var=dc@dc/n
    if kind=='bartlett_h-1':
        for lag in range(1,h): var+=2*(1-lag/h)*(dc[lag:]@dc[:-lag])/n
    elif kind=='rect_h-1':
        for lag in range(1,h): var+=2*(dc[lag:]@dc[:-lag])/n
    elif kind=='bartlett_1.5h':
        L=int(round(1.5*h))
        for lag in range(1,L+1): var+=2*(1-lag/(L+1))*(dc[lag:]@dc[:-lag])/n
    return m/np.sqrt(var/n) if var>0 else np.nan
def sim(rng, n_eval, W, k, h, rho, mu, intercept, refit):
    T=W+h+n_eval+h+5
    e=rng.normal(size=(T,k))
    if rho:
        X=np.empty_like(e); X[0]=e[0]/np.sqrt(1-rho**2)
        for t in range(1,T): X[t]=rho*X[t-1]+e[t]
    else: X=e
    r=mu+rng.normal(size=T+h+1)
    cs=np.cumsum(r); y=np.array([cs[t+h]-cs[t] for t in range(T)])   # y_t = r_{t+1}+...+r_{t+h}
    Z=np.column_stack([np.ones(T),X]) if intercept else X
    P=np.einsum('ti,tj->tij',Z,Z).cumsum(0); Q=(Z*y[:,None]).cumsum(0)
    ts=np.arange(W+h, W+h+n_eval)                 # decision rows
    if refit>1: fit_at=ts[0]+((ts-ts[0])//refit)*refit
    else: fit_at=ts
    hi=fit_at-h; lo=hi-W                           # train rows (lo, hi]: targets realised by fit time
    A=P[hi]-P[lo]; b=Q[hi]-Q[lo]
    beta=np.linalg.solve(A,b[:,:,None])[:,:,0]
    f=(Z[ts]*beta).sum(1)
    return y[ts], f
def run(label, R=1500, **kw):
    rng=np.random.default_rng(12345); h=kw['h']
    out={'code':[], 'rect_h-1':[], 'bartlett_1.5h':[]}; dmbad=[]
    for _ in range(R):
        y,f=sim(rng,**kw)
        out['code'].append(clark_west(y,f,0.0,h=h)[0])
        adj=2*y*f
        for kind in ('rect_h-1','bartlett_1.5h'): out[kind].append(cw_var(adj,h,kind))
    s=f"{label:58s}"
    for kind,v in out.items():
        v=np.array(v); s+=f" | {kind}: 5%={np.nanmean(v>1.645):.3f} 10%={np.nanmean(v>1.282):.3f}"
    print(s, flush=True)
base=dict(n_eval=1500, W=250, k=5)
t0=time.time()
run('h=1 iid X, mu=0, no intercept, refit each step', h=1, rho=0, mu=0, intercept=False, refit=1, **base)
run('h=1 iid X, mu=0, intercept, refit every 21', h=1, rho=0, mu=0, intercept=True, refit=21, **base)
run('h=1 AR(.95) X, mu=0, intercept, refit every 21', h=1, rho=0.95, mu=0, intercept=True, refit=21, **base)
run('h=5 iid X, mu=0, intercept, refit each step', h=5, rho=0, mu=0, intercept=True, refit=1, **base)
run('h=5 AR(.95) X, mu=0, intercept, refit each step', h=5, rho=0.95, mu=0, intercept=True, refit=1, **base)
run('h=5 AR(.95) X, mu=0, intercept, refit every 21', h=5, rho=0.95, mu=0, intercept=True, refit=21, **base)
run('h=20 AR(.95) X, mu=0, intercept, refit every 21', h=20, rho=0.95, mu=0, intercept=True, refit=21, **base)
print('--- non-zero mean target, X useless (the "variables have information" reading is false here) ---')
run('h=1 iid X, mu=0.05sd, intercept, refit every 21', h=1, rho=0, mu=0.05, intercept=True, refit=21, **base)
run('h=1 iid X, mu=0.10sd, intercept, refit every 21', h=1, rho=0, mu=0.10, intercept=True, refit=21, **base)
run('h=1 iid X, mu=0.6sd (F11-like drift), W=60,n=120', h=1, rho=0, mu=0.6, intercept=True, refit=1, n_eval=120, W=60, k=5)
print('secs', round(time.time()-t0))
