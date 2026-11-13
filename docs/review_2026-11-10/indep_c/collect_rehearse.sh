#!/bin/bash
# usage: collect_rehearse.sh <tag> <collect args...>
tag=$1; shift
cd /tmp/e2e
rm -rf /tmp/e2e/data && cp -a /tmp/review_c/pristine/data /tmp/e2e/data
timeout 600 python collect.py "$@" > /tmp/review_c/collect_$tag.log 2>&1; rc=$?
echo "=== [$tag] collect.py $* -> exit code $rc"
grep -n "경고\|이어받음\|fail\|건너뜀\|수집된 것이\|완료\|superseded\|유지\|Traceback\|Error" /tmp/review_c/collect_$tag.log | cut -c1-260 | head -30
python - <<'PY'
import sys, json, hashlib; sys.path.insert(0,'/tmp/e2e')
from pathlib import Path
import pandas as pd
V=Path('/tmp/e2e/data/versions'); P=Path('/tmp/review_c/pristine/data/versions')
vs=sorted(p.name for p in V.iterdir() if (p/'manifest.json').exists())
old='20261103T141116Z'; new=vs[-1]
print('versions with manifest:', vs, '| dirs without manifest:', [p.name for p in V.iterdir() if not (p/'manifest.json').exists()])
def ids(d): 
    out={}
    for f in sorted(d.glob('variables_*.parquet')):
        for v in pd.read_parquet(f,columns=['variable_id'])['variable_id'].unique(): out.setdefault(v,[]).append(f.name)
    return out
o=ids(P/old); n=ids(V/new)
print('latest =',new, '(new)' if new!=old else '(same as before)')
print('  files:', sorted(p.name for p in (V/new).iterdir()))
print('  variables: before', len(o), 'now', len(n), '| LOST:', sorted(set(o)-set(n)), '| gained:', sorted(set(n)-set(o)))
for t in ('091160','091230'):
    a=pd.read_parquet(P/old/f'prices_{t}.parquet'); 
    try:
        b=pd.read_parquet(V/new/f'prices_{t}.parquet'); print(f'  prices_{t}: before {len(a)} rows {a.index.min().date()}~{a.index.max().date()} | now {len(b)} rows {b.index.min().date()}~{b.index.max().date()} | identical={a.equals(b)}')
    except Exception as e: print(f'  prices_{t}: MISSING in latest ({type(e).__name__})')
for v in ('variables_xlsx_macro_seoul34.parquet','variables_xlsx_master.parquet'):
    if (V/new/v).exists() and (P/old/'variables_xlsx_macro_seoul34.parquet').exists():
        a=pd.read_parquet(P/old/'variables_xlsx_macro_seoul34.parquet').sort_values(['variable_id','date']).reset_index(drop=True); b=pd.read_parquet(V/new/v).sort_values(['variable_id','date']).reset_index(drop=True)
        print('  xlsx data', v, 'equal to previous xlsx data:', a.equals(b))
m=json.loads((V/new/'manifest.json').read_text()); run=(m.get('collect_runs') or [{}])[-1]
print('  manifest: variables_collected', m.get('variables_collected'), 'variables_missing', m.get('variables_missing'), 'tickers', m.get('tickers'), '| run.sources', run.get('sources'), '| carried_from', (run.get('carried_from') or {}).get('version'), list((run.get('carried_from') or {}).get('files',{}))[:12], '| superseded', run.get('superseded'), '| multi', run.get('multi_source_ids'), '| log', {k:v[:40] for k,v in (run.get('log') or {}).items()})
if new==old:
    pm=json.loads((P/old/'manifest.json').read_text())
    print('  existing version manifest changed:', {k:(pm.get(k), m.get(k)) for k in ('created_utc','updated_utc','variables_collected','master_sha256')}, '| files removed from manifest:', sorted(set(pm['files'])-set(m['files'])), 'added:', sorted(set(m['files'])-set(pm['files'])), '| changed sha:', [f for f in pm['files'] if f in m['files'] and pm['files'][f]!=m['files'][f]])
print('  master file changed:', hashlib.sha256(Path('/tmp/e2e/data/variable_master.xlsx').read_bytes()).hexdigest()!=hashlib.sha256(Path('/tmp/review_c/pristine/data/variable_master.xlsx').read_bytes()).hexdigest(), '| master_history:', [p.name for p in Path('/tmp/e2e/data/master_history').glob('*')] if Path('/tmp/e2e/data/master_history').exists() else None, '| registry.csv changed:', Path('/tmp/e2e/data/registry.csv').read_bytes()!=Path('/tmp/review_c/pristine/data/registry.csv').read_bytes())
PY
