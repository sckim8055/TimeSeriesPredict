import TechnicalInfo from './TechnicalInfo'
import { Fragment, useMemo, useState } from 'react'

/** 데이터 받는 법(engine/api_spec.py): 같은 값을 다른 사람이 다시 받을 수 있게 주소·매개변수·응답에서 쓰는 열·코드 위치. API 키는 자리표시자만. */
export function ApiSpec({ a }: { a: any }) {
  if (!a) return null
  const copy = (t: string) => navigator.clipboard?.writeText(t)
  return <div className="apispec">
    <div><b>{a.source}</b>{a.role ? ` · ${a.role}` : ''} · {a.method}{a.library ? ` · ${a.library}` : ''}</div>
    {a.full_url && <div className="apiurl"><code>{a.full_url}</code> <button type="button" className="link" onClick={() => copy(a.full_url)}>복사</button></div>}
    {!a.full_url && a.url && <div className="hint">{a.url}</div>}
    {a.params && <table><tbody>{Object.entries(a.params).map(([k, v]) => <tr key={k}><td>{k}</td><td><code>{v === null ? 'null(없음)' : String(v)}</code></td></tr>)}</tbody></table>}
    {a.python && <div className="apiurl"><span className="hint">파이썬:</span> <code>{a.python}</code> <button type="button" className="link" onClick={() => copy(a.python)}>복사</button></div>}
    <div className="hint">기간: {a.start} ~ {a.end}</div>
    {a.key_env && <div className="hint">키: <code>.env</code> 의 <code>{a.key_env}</code> (본인 키로 바꿔 넣기 — 화면에는 절대 표시하지 않음)</div>}
    <div className="hint">응답에서 쓰는 값: {a.response}</div>
    <div className="hint">코드: <code>{a.code}</code>{a.doc ? <> · 공식 문서: {a.doc}</> : null}</div>
    {a.note && <div className="hint" style={{ color: 'var(--warn)' }}>{a.note}</div>}
  </div>
}
import type { Meta } from './api'
import { SourceBadge, SOURCE_INFO, KIND_SYMBOL, KIND_LABEL } from './source'
import AddVariable from './AddVariable'
import VariableValues, { DataCheckPanel, TickerValues } from './VariableValues'

export default function Variables({ meta, onReload }: { meta: Meta; onReload?: () => void }) {
  const [q, setQ] = useState('')
  const [openApi, setOpenApi] = useState<string | null>(null)
  const [openVal, setOpenVal] = useState<string | null>(null)
  const [openTk, setOpenTk] = useState<string | null>(null)
  const rows = useMemo(() => meta.variables.filter(v => !q || `${v.id} ${v.name} ${v.description} ${v.group} ${v.domain}`.toLowerCase().includes(q.toLowerCase())), [meta, q])
  const unconfirmed = meta.variables.filter(v => v.usable && !v.lag_confirmed).length
  return (
    <div>
      <h2>변수 사전</h2>
      <p className="lead"><code>data/variable_master.xlsx</code> 가 유일한 기준입니다. 인터넷 출처의 변수는 아래 '변수 추가'로 넣을 수 있고, 각 줄의 [값 보기]에서 저장된 값·점검을 보고 발표시차·변환을 고칠 수 있습니다(그 밖의 수정은 엑셀을 고친 뒤 새로고침). lag_days 는 "기준일(기간 시작일) → 공개일" 일수이며 변수별로 다릅니다. 발표시차는 2026-11-05 에 기관 공표일정·발표 보도로 검증했습니다: <b>✓ 확인</b> 49개, <b>? 근거 부족</b> 16개. 오른쪽 "발표시차 근거" 열에 관측 공표일 범위와 계산(최장+여유)이, 마우스를 올리면 출처 전문이 보입니다. 상세는 도움말 「발표시차(lag) 검증 보고서」.</p>
      <AddVariable onChanged={() => onReload?.()} transformOptions={meta.transform_options} />
      <div className="toolbar">
        <input placeholder="id·이름·그룹으로 찾기" value={q} onChange={e => setQ(e.target.value)} style={{ width: 260 }} />
        <span className="hint">{(['file', 'api', 'derived'] as const).map(k => { const n = meta.variables.filter(v => (SOURCE_INFO[v.source]?.kind ?? 'api') === k).length; return <span key={k} style={{ marginRight: 10 }}>{KIND_SYMBOL[k]} {KIND_LABEL[k]} {n}개</span> })}</span>
        <span className="hint">{meta.variables.length}개 · 실험 가능 {meta.variables.filter(v => v.usable && v.has_data).length}개 · lag 미확인 {unconfirmed}개</span>
        {meta.master_warnings.length > 0 && <span className="badge warn">마스터 경고 {meta.master_warnings.length}</span>}
      </div>
      <div className="panel" style={{ padding: 0, overflow: 'auto', maxHeight: '75vh' }}>
        <table>
          <thead><tr><th>값</th><th>id</th><th>이름</th><th>설명</th><th>그룹</th><th>도메인</th><th>주기</th><th>발표시차</th><th>출처</th><th>위치(파일·코드) · 데이터 받는 법</th><th>데이터 기간</th><th>변환</th><th>상태</th><th>발표시차 근거</th></tr></thead>
          <tbody>{rows.map(v => (
            <Fragment key={v.id}><tr>
              <td><button type="button" className="link" onClick={() => setOpenVal(openVal === v.id ? null : v.id)}>{openVal === v.id ? '▼ 닫기' : '▶ 값 보기'}</button></td><td>{v.id}</td><td>{v.name}</td><td style={{ whiteSpace: 'normal', maxWidth: 360 }}>{v.description}</td><td>{v.group}</td><td>{v.domain}</td><td>{v.freq}</td><td><span className={'badge lagb ' + (v.lag_confirmed ? 'pass' : 'warn')} title={v.lag_confirmed ? '공표일정으로 확인' : '근거 부족(미확인)'}>{v.lag_confirmed ? '✓' : '?'} {v.lag_days}일</span></td>
              <td><SourceBadge v={v} /></td>
              <td style={{ fontSize: 12, whiteSpace: 'normal', maxWidth: 260 }}>{v.source === 'XLSX' ? <>{(v.stored?.upload ?? 'data/variable_master.xlsx').replace('data/', '')}<span className="hint"> · {String(v.source_code).replace('!', ' · 열 ')}</span></> : <><code style={{ fontSize: 11.5 }}>{v.source_code || '-'}</code>{v.api && <div><button type="button" className="link apibtn" onClick={() => setOpenApi(openApi === v.id ? null : v.id)}>{openApi === v.id ? '▼' : '▶'} 데이터 받는 법(주소·값)</button></div>}</>}</td>
              <td className="num" style={{ fontSize: 12 }}>{v.stored ? <>{v.stored.first.slice(0, 7)} ~ {v.stored.last.slice(0, 7)}<div className="hint">{v.stored.n.toLocaleString()}개</div></> : <span className="badge fail">없음</span>}</td>
              <td>{v.transform || 'none'}</td>
              <td>{!v.active && <span className="badge muted">active=N</span>}{v.duplicate_of && <span className="badge muted">중복 → {v.duplicate_of}</span>}{!v.has_data && <span className="badge fail">데이터 없음</span>}{v.usable && v.has_data && <span className="badge pass">사용 가능</span>}</td>
            <td style={{ whiteSpace: 'normal', maxWidth: 460, fontSize: 12, color: 'var(--ink-2)' }} title={v.note}>{(() => { const ev = (v.note || '').split(' | ').find(x => x.startsWith('[lag 검증'))?.replace(/^\[lag 검증 [0-9-]+\] /, '') ?? ''; const [head, rest] = ev.split('. 출처: '); return <>{head}{rest && <div style={{ color: 'var(--ink-3, #8a94a6)', marginTop: 2 }}>출처·수정: {rest}</div>}</> })()}</td></tr>
            {openApi === v.id && v.api && <tr><td colSpan={14} style={{ background: '#fafbfc' }}><ApiSpec a={v.api} /></td></tr>}
            {openVal === v.id && <tr><td colSpan={14} style={{ background: '#fafbfc', whiteSpace: 'normal' }}><VariableValues id={v.id} onChanged={() => onReload?.()} transformOptions={meta.transform_options} /></td></tr>}</Fragment>))}</tbody>
        </table>
      </div>
      <DataCheckPanel />
      <TechnicalInfo />
      <div className="panel"><h3>종목 가격</h3>
        <p className="hint">실험이 읽는 가격(시가·고가·저가·종가·거래량)입니다. [값 보기]로 그래프·점검·CSV 를 봅니다. 종목은 위 '종목 추가'로 넣고, [데이터 가져오기]가 끝나면 가격이 생깁니다.</p>
        <table><tbody>{meta.tickers.map(t => <Fragment key={t.id}><tr>
          <td><button type="button" className="link" onClick={() => setOpenTk(openTk === t.id ? null : t.id)}>{openTk === t.id ? '▼ 닫기' : '▶ 값 보기'}</button></td>
          <td>{t.id}</td><td>{t.name}</td><td>{t.kind === 'stock' ? '주식' : 'ETF'}</td></tr>
          {openTk === t.id && <tr><td colSpan={4} style={{ background: '#fafbfc', whiteSpace: 'normal' }}><TickerValues id={t.id} /></td></tr>}</Fragment>)}</tbody></table>
      </div>
      {meta.price_api && <div className="panel"><h3>종목 가격 데이터 받는 법</h3>
        <p className="hint">실험의 기본 데이터(시가·종가 등)는 세 출처를 받아 조립합니다: 공식(공공데이터포털, 2020~)을 기준으로 그 이전은 Yahoo, 빠진 거래일은 네이버 수정주가로 비율 보정. 어느 날을 어느 출처로 채웠는지는 데이터 버전 폴더의 <code>repair_log_&lt;티커&gt;.csv</code>. 이 데이터 버전의 수집: {meta.data_version_collect?.start} ~ {meta.data_version_collect?.end ?? `수집일(${String(meta.data_version_collect?.created_utc ?? '').slice(0, 10)})`}.</p>
        {Object.entries(meta.price_api).map(([t, specs]) => <details key={t} className="apidet"><summary>{t} — {specs.length}개 출처</summary>{specs.map((a: any, i: number) => <ApiSpec key={i} a={a} />)}</details>)}
      </div>}
      {meta.master_warnings.length > 0 && <div className="panel"><h3>마스터 경고</h3><ul>{meta.master_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></div>}
    </div>
  )
}
