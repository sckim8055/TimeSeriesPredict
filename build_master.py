"""현재 registry.csv(55개) → data/variable_master.xlsx 초안 (그룹·도메인·중복 제안 포함). 사용자가 엑셀을 고쳐 확정한다."""
import pandas as pd
r = pd.read_csv('data/registry.csv', dtype=str, keep_default_na=False)
rows = []
def add(id, name, group, domain, role, freq, lag, source, code, alt='', dup='', note='', desc='', unit=''):
    rows.append(dict(id=id, name=name, description=desc, group=group, domain=domain, role=role, freq=freq, lag_days=lag,
                     source=source, source_code=code, alt_sources=alt, duplicate_of=dup, default_fill='ffill', unit=unit, active='N' if dup else 'Y', note=note))
add('USDKRW','원/달러 환율','위험선호','공통','설명변수','D',0,'YF','KRW=X',alt='XLSX:df_M!L1(월말)',note='일간을 대표로. 엑셀 L1(월말)은 대안')
add('KOSPI','코스피 지수','국내시장','공통','설명변수','D',0,'YF','^KS11')
add('SOX','필라델피아 반도체지수','반도체 글로벌','반도체','설명변수','D',1,'YF','^SOX',note='미국 마감이 한국 다음날 새벽 → lag 1')
add('NVDA','엔비디아 주가','반도체 글로벌','반도체','설명변수','D',1,'YF','NVDA')
add('MU','마이크론 주가','반도체 글로벌','반도체','설명변수','D',1,'YF','MU')
add('TSM','TSMC(ADR) 주가','반도체 글로벌','반도체','설명변수','D',1,'YF','TSM')
add('SAMSUNG','삼성전자 주가','반도체 국내','반도체','설명변수','D',0,'YF','005930.KS',note='ETF 구성종목 — ETF와 거의 같은 정보. 넣는 의미 자체가 실험 대상')
add('HYNIX','SK하이닉스 주가','반도체 국내','반도체','설명변수','D',0,'YF','000660.KS',note='위와 같음')
add('US10Y','미 국채 10년 금리','금리·유동성','공통','설명변수','D',1,'FRED','DGS10',alt='YF:^TNX')
add('US2Y','미 국채 2년 금리','금리·유동성','공통','설명변수','D',1,'FRED','DGS2')
add('VIX','VIX 변동성지수','위험선호','공통','설명변수','D',1,'FRED','VIXCLS',alt='YF:^VIX')
add('FEDFUNDS','미 연방기금금리(월평균)','금리·유동성','공통','설명변수','M',1,'FRED','FEDFUNDS',note='월평균값, 다음달 초 공개')
add('BOK_RATE','한은 기준금리','금리·유동성','공통','설명변수','M',0,'ECOS','722Y001/0101000/M',alt='XLSX:df_M!N1',note='결정일 당일 공개. 엑셀 N1과 동일')
add('KR10Y','국고채 10년 금리','금리·유동성','공통','설명변수','D',1,'ECOS','817Y002/010210000/D')
add('KR3Y','국고채 3년 금리','금리·유동성','공통','설명변수','D',1,'ECOS','817Y002/010200000/D',alt='XLSX:df_M!N12')
add('TERM_US','미 장단기 금리차(10y-2y)','금리·유동성','공통','설명변수','D',1,'DERIVED','US10Y - US2Y',note='파생 변수 예시')
gmap = {
 'A1':('국내경기','공통','',''), 'B1':('국내경기','공통','',''), 'B2':('국내경기','공통','',''),
 'F1':('물가','공통','',''), 'F7':('물가','공통','',''), 'F10':('물가','공통','','F7과 상관 매우 높음 — 하나만 쓰는 것 검토'),
 'F11':('부동산 가격','부동산','타겟후보',''), 'F12':('부동산 가격','부동산','둘다',''),
 'H1':('부동산 공급','부동산','',''), 'H2':('부동산 공급','부동산','',''), 'H3':('부동산 가격','부동산','','DUP:O3'),
 'K1':('국내경기','공통','',''), 'K2':('국내경기','공통','',''), 'K3':('국내경기','공통','',''), 'K4':('국내경기','공통','','K3과 상관 높음'),
 'L1':('위험선호','공통','','DUP:USDKRW'), 'L2':('위험선호','공통','',''), 'L3':('위험선호','공통','',''), 'L9':('대외','공통','',''), 'M3':('금리·유동성','공통','',''),
 'O1':('부동산 가격','부동산','','F12와 유사(전세)'), 'O2':('부동산 거래','부동산','',''), 'O3':('부동산 가격','부동산','',''), 'O4':('부동산 금융','부동산','','DUP:N20'),
 'N1':('금리·유동성','공통','','DUP:BOK_RATE'), 'N2':('가계금융','부동산','',''), 'N4':('가계금융','부동산','',''), 'N5':('부동산 금융','부동산','',''), 'N6':('부동산 금융','부동산','',''),
 'N7':('가계금융','부동산','',''), 'N8':('금리·유동성','공통','',''), 'N10':('금리·유동성','공통','',''), 'N11':('금리·유동성','공통','',''), 'N12':('금리·유동성','공통','','DUP:KR3Y'),
 'N19':('가계금융','부동산','',''), 'N20':('부동산 금융','부동산','',''), 'N21':('가계금융','부동산','',''), 'N24':('가계금융','부동산','',''),
}
for _, v in r[r.source=='XLSX'].iterrows():
    g, dom, role, tag = gmap.get(v['id'], ('', '', '', ''))
    dup = tag[4:] if tag.startswith('DUP:') else ''
    note = f'{dup} 와 같은 경제량 → 실험에서 자동 제외(대안 출처로만 보관)' if dup else tag
    add(v['id'], v['name'].strip(), g, dom, role or '설명변수', v['freq'], v['lag_days'], 'XLSX', v['source_code'], dup=dup, note=note, desc=v['description'], unit=v['unit'])
cols = ['id','name','description','group','domain','role','freq','lag_days','source','source_code','alt_sources','duplicate_of','default_fill','unit','active','note']
df = pd.DataFrame(rows)[cols]
with pd.ExcelWriter('data/variable_master.xlsx', engine='openpyxl') as w:
    df.to_excel(w, sheet_name='variables', index=False)
    pd.DataFrame({'열': cols, '설명': [
        '표준 ID. 실험 설정·코드에서 쓰는 이름. 영문·숫자·_ 만', '표준명(한글)', '설명', '가설 그룹 — 사용자가 정함. ablation 의 단위', '적용 연구: 공통 / 반도체 / 부동산',
        '설명변수 / 타겟후보 / 둘다', '원래 주기 D/W/M/Q/H/Y', '기준일→공개일 일수 (모르면 비우고 note 에 "확인 필요")', '대표 출처 YF/FRED/ECOS/KRX/XLSX/DERIVED',
        '출처 조회 코드. XLSX 는 시트!열, ECOS 는 통계/항목/주기, DERIVED 는 식', '같은 지표의 다른 출처(교차검증용). 출처:코드 를 ; 로 구분', '다른 변수와 같은 경제량이면 그 id → 실험에서 자동 제외',
        '저주기→일간 채움 기본값 ffill/bfill/interp', '단위', 'Y/N 사용 여부', '근거·주의·미확인 사항']}).to_excel(w, sheet_name='설명', index=False)
    pd.DataFrame({'group': sorted(df.group.unique()), '가설(사용자 작성)': ''}).to_excel(w, sheet_name='groups', index=False)
    # 열 너비
    ws = w.sheets['variables']
    for col, wdt in zip('ABCDEFGHIJKLMNOP', [12,26,30,14,8,10,6,9,9,26,24,12,10,8,7,50]):
        ws.column_dimensions[col].width = wdt
print(len(df), '행'); print(df.groupby(['domain','group']).size().to_string()); print('\n중복:', df[df.duplicate_of!=''][['id','duplicate_of']].to_string(index=False))
