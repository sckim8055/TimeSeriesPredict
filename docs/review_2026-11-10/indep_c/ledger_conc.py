import sys, shutil, multiprocessing as mp
sys.path.insert(0,'/tmp/e2e')
from pathlib import Path
from engine import ledger
d=Path('/tmp/review_c/led/conc')
def work(tag):
    for i in range(200): ledger.record(d,'trial',config_hash=f'{tag}{i}')
if __name__=='__main__':
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); (d/'experiment_counter.json').write_text('{"count": 0}')
    ledger.ensure(d)
    ps=[mp.Process(target=work,args=(t,)) for t in 'ab']; [p.start() for p in ps]; [p.join() for p in ps]
    try:
        es=ledger.entries(d); print('lines',len(es),'count',ledger.count(d),'(expected 400)'); print('verify',ledger.verify(d))
        seqs=[e['seq'] for e in es]; print('duplicate seq values:', len(seqs)-len(set(seqs)))
    except Exception as e: print('entries raises', type(e).__name__, e)
    import json; print('counter copy', json.loads((d/'experiment_counter.json').read_text()))
    # genesis race: two processes ensure() at once on fresh dir
    d2=Path('/tmp/review_c/led/conc2')
    cnt=0
    for rep in range(30):
        shutil.rmtree(d2, ignore_errors=True); d2.mkdir(); (d2/'experiment_counter.json').write_text('{"count": 98}')
        ps=[mp.Process(target=ledger.record,args=(d2,'trial')) for _ in range(2)]; [p.start() for p in ps]; [p.join() for p in ps]
        es=ledger.entries(d2); g=sum(1 for e in es if e['kind']=='genesis')
        if g>1 or ledger.count(d2)!=100: cnt+=1; last=(g, ledger.count(d2), [e['kind'] for e in es])
    print('genesis race: anomalies in 30 reps:', cnt, last if cnt else '')
