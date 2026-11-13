# S4: failure AFTER sealed metrics were printed -> status "failed" (= not counted)
import sys, runpy
sys.path.insert(0,'/tmp/e2e')
import engine.runlog as rl
def boom(**kw): raise RuntimeError("simulated failure after metrics were logged (e.g. Ctrl-C / manifest unreadable)")
rl.build_run_record = boom
sys.argv=['cli.py','run','configs/04_ridge_A.json','--sealed']
runpy.run_path('/tmp/e2e/cli.py', run_name='__main__')
