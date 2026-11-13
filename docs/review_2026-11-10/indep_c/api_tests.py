import sys, json, shutil, threading
sys.path.insert(0,'/tmp/e2e')
from pathlib import Path
import api.main as m
from fastapi import HTTPException
class FakeJobs:
    def __init__(s): s.j=[]
    def submit(s, config, *, sealed=False, kind="run", sealed_note="", sealed_source=None):
        job={"id":f"job{len(s.j)}","status":"queued","sealed":sealed,"sealed_source":sealed_source,"submitted_at":str(len(s.j)),"config":config}
        s.j.append(job); return job
    def list(s, limit=100): return list(s.j)
m.jobs=FakeJobs()
T=Path('/tmp/e2e/results/trading')
A='04_ridge_A_v1107_6e2191440f16'
r=json.loads((T/f'{A}.json').read_text()); r['config']['name']='ridge_copy'
B='ridge_copy_6e2191440f16'
(T/f'{B}.json').write_text(json.dumps(r)); shutil.copy(T/f'{A}_daily.parquet', T/f'{B}_daily.parquet')
pv=m.sealed_preview(A); print('preview blockers',pv['blockers'],'hash',pv['config_hash'],'hist',{k:(len(v) if isinstance(v,list) else v) for k,v in pv['history'].items()})
print('warnings:', [w[:60] for w in pv['warnings']])
def run(stem,**kw):
    try:
        out=m.sealed_run(m.SealedRunBody(stem=stem,confirm=True,**kw)); return 'ACCEPTED '+out['job']['id']+' hash '+out['config_hash']
    except HTTPException as e: return f'REJECTED {e.status_code} {e.detail[:70]}'
print('1 run A        ->', run(A))
print('2 run A again  ->', run(A))
print('3 run B (same config hash, other stem) ->', run(B))
try: m.submit(m.SubmitBody(config=r['config'], sealed=True)); print('submit sealed accepted')
except HTTPException as e: print('4 submit sealed=True ->', e.status_code, e.detail)
c=dict(r['config']); c['validation']=dict(c['validation'], sealed_from=None)
try: out=m.submit(m.SubmitBody(config=c)); print('5 submit dev run with sealed_from=None -> ACCEPTED', out['config_hash'])
except HTTPException as e: print('5 ->', e.status_code, e.detail)
m.jobs=FakeJobs()
res=[]
def call(): res.append(run(A))
ts=[threading.Thread(target=call) for _ in range(2)]; [t.start() for t in ts]; [t.join() for t in ts]
print('6 two simultaneous requests for A ->', res)
for s in ('trial_ledger','run_log','sealed_log','experiment_counter','.x','history/x', A):
    print('find', s, '->', m._find_result(s) is not None)
(T/f'{B}.json').unlink(); (T/f'{B}_daily.parquet').unlink()
