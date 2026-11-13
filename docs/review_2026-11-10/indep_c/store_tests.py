import sys, datetime as dt, time, warnings
sys.path.insert(0,'/tmp/e2e')
import numpy as np, pandas as pd, openpyxl
from engine.store import parse_date_cell as pdc, parse_value_cell as pvc, read_data_sheets, excel_to_long, DataVersion
warnings.simplefilter('ignore')
print('== parse_date_cell ==')
cases=[2020.1, 202001.0, 2020.0, "2020-1", "20201", "20-01-31", "20/1/31", "1/2/2020", "31/01/2020", "01-02-20", "2020.1", "2020.10", 2020.10, dt.datetime(2020,1,31,15,30), dt.time(12,0), "12:00", 43831, 43831.0, 43831.5, "43831", "2020H1", "2020-H1", "2020 H2", "2020S1", "2020년 상반기", "2020Q1", "2020-Q4", "Q1 2020", "1Q20", "2020M01", "2020-01-31 00:00:00", "2020-01-31T00:00:00", "Jan 2020", "2020 Jan", "January", "3.5", "1.5", "5", "may", "now", "today", "1e5", " 2020-01-31 ", "2020/13", "2020-02-30", "２０２０-01-31", "2020.1.5.", "200101", 200101, "99", 1999, "2020-00", "nan", "", " ", "-", np.float64(2020.0), np.int64(202001), pd.Timestamp('2020-01-31 10:00', tz='Asia/Seoul'), "2020-W05", "2020w5", "20200131.0", "2020-1-1 ~ 2020-1-31", "T", "A", "D", "M"]
for c in cases:
    try: r=pdc(c)
    except Exception as e: r=f'EXC {type(e).__name__}: {e}'
    print(f'  {c!r:42} -> {r}')
print('== parse_value_cell ==')
for c in ["1.234,5","1,5","(1,234)","1e5","1E-3","inf","-inf","Infinity","NaN","nan","1_000","１２３","3.5%","1 234.5","+5","--","#DIV/0!","#N/A","#VALUE!","TRUE",True,"0x10","1,234,567.8",",","1.","  ","na","N.A.","..","…","x","1 e5", "１，２３４", "−5", "5-", "12:00", dt.datetime(2020,1,1), np.float64('inf'), float('inf'), np.int64(3), np.float32(1.5), np.bool_(True)]:
    try: r=pvc(c)
    except Exception as e: r=f'EXC {type(e).__name__}: {e}'
    print(f'  {c!r:28} -> {r}')
