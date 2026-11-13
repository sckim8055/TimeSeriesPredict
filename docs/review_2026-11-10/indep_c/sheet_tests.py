import sys, datetime as dt, time, warnings, zipfile, re, shutil
sys.path.insert(0,'/tmp/e2e')
import numpy as np, pandas as pd, openpyxl
from engine.store import read_data_sheets, excel_to_long
from engine.registry import Registry
from engine.master_check import check_master
warnings.simplefilter('ignore')
D='/tmp/review_c/xl/'
def mk(path, var_rows, sheets, write_only=False):
    wb=openpyxl.Workbook(write_only=write_only)
    ws=wb.create_sheet('variables') if write_only else wb.active
    if not write_only: ws.title='variables'
    cols=["id","name","freq","source","source_code","lag_days","duplicate_of","active"]; ws.append(cols)
    for r in var_rows: ws.append([r.get(c,"") for c in cols])
    for sh,header,data in sheets:
        s=wb.create_sheet(sh); s.append(header)
        for row in data: s.append(row)
    wb.save(path)
V=lambda i,f='M':{"id":i,"freq":f,"source":"XLSX","source_code":f"df_{f}!{i}","lag_days":30}
def show(tag, path, reg=True):
    try:
        r,_=Registry.load_master(path); chk=check_master(r,path)
        long,iss=read_data_sheets(path)
        print(f'[{tag}] check errors={chk.errors} notices={chk.notices[:3]} new={[x["id"] for x in chk.new_vars]}')
        print('    rows:', [(a,str(c.date()),d) for _,a,c,d in long.itertuples(index=False)][:8], 'issues:', [(x['kind'],x.get('cell'),x.get('text')) for x in iss[:-1]])
        try: el=excel_to_long(path,r); print('    excel_to_long rows', len(el))
        except Exception as e: print('    excel_to_long raises:', str(e)[:160])
    except Exception as e: print(f'[{tag}] EXC {type(e).__name__}: {e}')
# 1 date column not in A
mk(D+'a.xlsx',[V('X1')],[('df_M',['X1','date'],[[1.0,'2020-01'],[2.0,'2020-02']])]); show('date col in B',D+'a.xlsx')
# 2 blank date with values, and date-only row
mk(D+'b.xlsx',[V('X1')],[('df_M',['date','X1'],[['2020-01',1.0],[None,2.0],['2020-03',None],[None,None],['2020-04',4.0]])]); show('blank date with value',D+'b.xlsx')
# 3 two-digit year / time / today strings in date col
mk(D+'c.xlsx',[V('X1')],[('df_M',['date','X1'],[['20-01-31',1.0],[dt.time(12,0),2.0],['1Q20',3.0],['1/2/2020',4.0]])]); show('ambiguous dates accepted',D+'c.xlsx')
# 4 values
mk(D+'d.xlsx',[V('X1')],[('df_M',['date','X1'],[['2020-01','1.234,5'],['2020-02','1,5'],['2020-03','inf'],['2020-04','1e5']])]); show('odd values accepted',D+'d.xlsx')
# 5 half-year
mk(D+'e.xlsx',[V('H1','H')],[('df_H',['date','H1'],[['2020H1',1.0],['2020H2',2.0]])]); show('2020H1',D+'e.xlsx')
# 6 duplicate header same sheet / two date columns
mk(D+'f.xlsx',[V('X1')],[('df_M',['date','X1','Date'],[['2020-01',1.0,'2021-01']])]); show('two date columns',D+'f.xlsx')
# 7 same variable in two sheets, registered M
mk(D+'g.xlsx',[V('X1')],[('df_M',['date','X1'],[['2020-01',1.0]]),('df_Q',['date','X1'],[['2020-01',9.0]])]); show('two sheets',D+'g.xlsx')
# 8 header not in row 1 (blank first row)
mk(D+'h.xlsx',[V('X1')],[('df_M',[None,None],[['date','X1'],['2020-01',1.0]])]); show('header in row 2',D+'h.xlsx')
# 9 stale/wrong <dimension> tag: data beyond the declared range
mk(D+'i.xlsx',[V('X1')],[('df_M',['date','X1'],[[f'{2000+i//12}-{i%12+1:02d}',float(i)] for i in range(120)])])
src=D+'i.xlsx'; dst=D+'i_dim.xlsx'
zin=zipfile.ZipFile(src); zout=zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED)
names={s.title:i for i,s in enumerate(openpyxl.load_workbook(src).worksheets,1)}
for it in zin.infolist():
    data=zin.read(it.filename)
    if it.filename==f"xl/worksheets/sheet{names['df_M']}.xml":
        before=re.search(rb'<dimension ref="[^"]*"',data).group(0); data=data.replace(before,b'<dimension ref="A1:B50"'); print('   dimension tag', before, '-> A1:B50')
    zout.writestr(it,data)
zout.close()
r,_=Registry.load_master(dst)
long,iss=read_data_sheets(dst); print('[wrong dimension tag] cell-reader rows:', len(long), '(file has 120) | pandas read_excel rows:', len(pd.read_excel(dst,sheet_name='df_M')), '| check errors:', check_master(r,dst).errors, '| excel_to_long:', len(excel_to_long(dst,r)))
# 10 formula with cached value lost
mk(D+'j.xlsx',[V('X1')],[('df_M',['date','X1'],[['2020-01',1.0],['2020-02','=B2*2']])]); show('formula without cached value',D+'j.xlsx')
# 11 timing 5000 x 300
t=time.time()
wb=openpyxl.Workbook(write_only=True); ws=wb.create_sheet('variables'); ws.append(["id","name","freq","source","source_code","lag_days","duplicate_of","active"])
for i in range(300): ws.append([f'V{i}','', 'D','XLSX',f'df_D!V{i}',1,'',''])
s=wb.create_sheet('df_D'); s.append(['date']+[f'V{i}' for i in range(300)])
days=pd.bdate_range('2005-01-03',periods=5000); rng=np.random.default_rng(0); M=rng.normal(size=(5000,300))
for d,row in zip(days,M): s.append([d.to_pydatetime()]+row.tolist())
wb.save(D+'big.xlsx'); print('big file written', round(time.time()-t,1),'s')
t=time.time(); r,_=Registry.load_master(D+'big.xlsx'); t1=time.time()-t
t=time.time(); long,iss=read_data_sheets(D+'big.xlsx'); t2=time.time()-t
t=time.time(); chk=check_master(r,D+'big.xlsx'); t3=time.time()-t
t=time.time(); el=excel_to_long(D+'big.xlsx',r); t4=time.time()-t
t=time.time(); pd.read_excel(D+'big.xlsx',sheet_name='df_D'); t5=time.time()-t
print(f'[5000x300] load_master {t1:.1f}s | read_data_sheets {t2:.1f}s rows={len(long)} | check_master {t3:.1f}s errors={len(chk.errors)} | excel_to_long {t4:.1f}s rows={len(el)} | (pandas read_excel one sheet {t5:.1f}s)')
