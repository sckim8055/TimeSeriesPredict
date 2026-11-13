import json, sys, os, shutil, subprocess, multiprocessing as mp
sys.path.insert(0,'/tmp/e2e')
from pathlib import Path
from engine import ledger
base=Path('/tmp/review_c/led'); shutil.rmtree(base, ignore_errors=True); base.mkdir()

# 1 count creates nothing
d=base/'nodir'/'trading'
print('1a count on missing dir ->', ledger.count(d), 'dir exists after:', d.exists(), 'verify:', ledger.verify(d), 'distinct', ledger.distinct_configs(d), 'exists', d.exists())
d=base/'t1'; d.mkdir(); (d/'experiment_counter.json').write_text('{"count": 98}')
print('1b', ledger.count(d), sorted(p.name for p in d.iterdir()))
(d/'experiment_counter.json').write_text('garbage')
print('1c corrupt counter count ->', ledger.count(d)); ledger.ensure(d); print('   genesis after corrupt counter:', ledger.entries(d)[0]['count'], '(note says?)', ledger.entries(d)[0]['note'][:40])

# 2 CRLF
d=base/'t2'; d.mkdir(); (d/'experiment_counter.json').write_text('{"count": 5}')
for i in range(3): ledger.record(d,'trial',config_hash=f'h{i}',name='이름')
raw=(d/ledger.LEDGER).read_bytes()
(d/ledger.LEDGER).write_bytes(raw.replace(b'\n',b'\r\n'))
print('2 CRLF verify', ledger.verify(d), 'count', ledger.count(d))
ledger.record(d,'trial',config_hash='h9'); print('2 CRLF + append (mixed) verify', ledger.verify(d))
# BOM
(d/ledger.LEDGER).write_bytes(b'\xef\xbb\xbf'+raw)
try: print('2b BOM count', ledger.count(d))
except Exception as e: print('2b BOM count raises', type(e).__name__, str(e)[:80])
print('2b BOM verify', ledger.verify(d))

# 3 partial last line
d=base/'t3'; d.mkdir(); (d/'experiment_counter.json').write_text('{"count": 5}')
for i in range(3): ledger.record(d,'trial',config_hash=f'h{i}')
raw=(d/ledger.LEDGER).read_bytes(); (d/ledger.LEDGER).write_bytes(raw[:-40])
try: print('3 partial count', ledger.count(d))
except Exception as e: print('3 partial-last-line: count raises', type(e).__name__, str(e)[:80])
print('3 verify', ledger.verify(d))
try: ledger.record(d,'deleted',result='x')
except Exception as e: print('3 record raises', type(e).__name__)
print('3 file tail after record attempt:', (d/ledger.LEDGER).read_text()[-200:].replace('\n','\\n'))
# 3b: blank/garbage middle line
d=base/'t3b'; d.mkdir(); (d/'experiment_counter.json').write_text('{"count": 5}')
for i in range(3): ledger.record(d,'trial',config_hash=f'h{i}')
L=(d/ledger.LEDGER).read_text().splitlines(); (d/ledger.LEDGER).write_text('\n'.join(L[:2]+['']+L[2:])+'\n')
print('3b blank line inserted verify', ledger.verify(d)['ok'])
# 3c: delete LAST line (truncate) -> verify?
(d/ledger.LEDGER).write_text('\n'.join(L[:-1])+'\n'); print('3c last line removed: verify', ledger.verify(d), 'count', ledger.count(d))
# 3d delete whole ledger, counter remains
os.remove(d/ledger.LEDGER); print('3d ledger deleted: count', ledger.count(d)); (d/'experiment_counter.json').write_text('{"count": 1}'); ledger.ensure(d); print('3d counter edited to 1 then ensure -> count', ledger.count(d), ledger.verify(d)['ok'])

# 5 U+2028 in name
d=base/'t5'; d.mkdir()
ledger.record(d,'trial',config_hash='h', name='a b')
try: print('5 count', ledger.count(d))
except Exception as e: print('5 U+2028 in name: count raises', type(e).__name__)
d=base/'t5b'; d.mkdir()
ledger.record(d,'trial',config_hash='h', name='a\x85b')
try: print('5b count', ledger.count(d))
except Exception as e: print('5b NEL in name: count raises', type(e).__name__)
