import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import TransformSelect, { type TransformOption } from './TransformSelect'

/** 변수 사전의 [값 보기] (2026-11-13): 데이터 버전에 저장된 값 그대로 + 점검 + CSV 내려받기 + 발표시차·변환 고치기. */
export default function VariableValues({ id, onChanged, transformOptions = [] }: { id: string; onChanged?: () => void; transformOptions?: TransformOption[] }) {
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)
  const [showTr, setShowTr] = useState(false)
  const [rev, setRev] = useState(0)
  useEffect(() => { setD(null); setErr(null) }, [id])
  useEffect(() => { api.varValues(id).then(x => { setD(x); setErr(null) }).catch(e => setErr(e.message)) }, [id, rev])   // 고친 뒤 다시 읽을 때는 화면을 비우지 않는다(저장 결과 문구가 남도록)
  if (err) return <p className="hint" style={{ color: 'var(--fail)' }}>{err}</p>
  if (!d) return <p className="hint">값을 읽는 중…</p>
  const c = d.checks
  return <div className="varvals">
    <p><b>{d.id}</b> {d.name} · 데이터 버전 <code>{d.version}</code> · {d.n > 0 ? <>{d.n.toLocaleString()}개 ({d.first} ~ {d.last})</> : null} · 주기 {d.freq} · 모델에 넣는 꼴 <code>{d.transform}</code>
      {d.n > 0 && <> · <a href={`/api/variables/${encodeURIComponent(d.id)}/values.csv?version=${encodeURIComponent(d.version)}`} download>엑셀로 내려받기(CSV)</a></>}</p>
    {d.message && <p className="hint" style={{ color: 'var(--warn)' }}>{d.message}</p>}
    {d.n > 0 && <>
      <Checks c={c} freq={d.freq} />
      <DataCheckLine dc={d.datacheck} />
      {d.cross?.length > 0 && <div className="checks"><p><b>다른 출처와 대조</b> <span className="hint">— {d.cross_rule}</span></p><CrossSourceTable rows={d.cross} /></div>}
      <div className="toolbar">
        <label><input type="checkbox" checked={showTr} onChange={e => setShowTr(e.target.checked)} /> 모델에 넣는 꼴({d.transform})로 바꾼 값 보기</label>
        {d.transform_note && <span className="hint"> — {d.transform_note}</span>}
      </div>
      <Line values={d.values} useTr={showTr} marks={(c?.reversals ?? []).map((x: any) => x.date)} />
      <details><summary className="hint">최근 값 30개 (전체는 CSV)</summary>
        <table className="mini"><thead><tr><th>날짜</th><th>저장된 값</th><th>바꾼 값({d.transform})</th></tr></thead>
          <tbody>{d.values.slice(-30).reverse().map((r: any) => <tr key={r.date}><td>{r.date}</td><td className="num">{fmt(r.value)}</td><td className="num">{r.transformed == null ? '-' : fmt(r.transformed)}</td></tr>)}</tbody></table>
      </details>
    </>}
    <Edit options={transformOptions} d={d} onSaved={() => { setRev(rev + 1); onChanged?.() }} />
  </div>
}

export function fmt(x: number) {
  if (x == null || !isFinite(x)) return '-'
  const a = Math.abs(x)
  return a !== 0 && (a < 0.001 || a >= 1e7) ? x.toExponential(3) : x.toLocaleString(undefined, { maximumFractionDigits: a < 1 ? 6 : 3 })
}

export function Checks({ c, freq }: { c: any; freq: string }) {
  if (!c) return null
  const ok = c.gaps.length === 0 && c.duplicate_dates === 0 && c.reversals.length === 0 && c.nonpositive === 0
  return <div className="checks">
    <p><b>점검</b> <span className="hint">— 알림일 뿐 값을 바꾸지 않습니다. 실제 시장 급변도 걸릴 수 있으니 날짜를 보고 판단하세요.</span></p>
    <ul>
      {ok && <li><span className="badge pass">문제 없음</span> 빈 기간·중복·되돌아온 날·0 이하 값이 없습니다.</li>}
      {c.gaps.length > 0 && <li><span className="badge warn">빈 기간 {c.gaps.length}곳</span> {freq} 주기 기준 {c.gap_days_limit}일보다 긴 간격: {c.gaps.slice(0, 6).map((g: any) => `${g.from} → ${g.to} (${g.days}일)`).join(' · ')}{c.gaps.length > 6 ? ' …' : ''}</li>}
      {c.duplicate_dates > 0 && <li><span className="badge warn">같은 날짜 {c.duplicate_dates}번 더</span> 같은 날짜가 두 번 이상 있습니다(실험에는 마지막 값만 쓰임).</li>}
      {c.reversals.length > 0 && <li><span className="badge warn">확인해 볼 날 {c.reversals.length}곳</span> 하루 크게 움직였다가 다음 날 되돌아온 날(자료 오류의 흔한 모양): {c.reversals.slice(0, 6).map((r: any) => `${r.date} (${fmt(r.before)} → ${fmt(r.value)} → ${fmt(r.after)})`).join(' · ')}{c.reversals.length > 6 ? ' …' : ''}</li>}
      {c.reversal_note && <li className="hint">{c.reversal_note}</li>}
      {c.nonpositive > 0 && <li><span className="badge fail">0 이하 {c.nonpositive}개</span> 변화율·로그 변환인데 0 이하 값이 있어 그 날은 계산되지 않습니다.</li>}
      {c.largest_moves?.length > 0 && <li className="hint">참고 — 하루 변화가 가장 컸던 날({c.largest_moves[0].kind}): {c.largest_moves.map((m: any) => `${m.date} (${m.change > 0 ? '+' : ''}${fmt(m.change)})`).join(' · ')}</li>}
    </ul>
  </div>
}

export function Line({ values, useTr, marks, field }: { values: any[]; useTr: boolean; marks: string[]; field?: string }) {
  const W = 900, H = 200, P = 36
  const pts = useMemo(() => {
    const raw = values.map((r: any) => ({ t: r.date, y: field ? r[field] : useTr ? r.transformed : r.value })).filter((p: any) => p.y != null && isFinite(p.y))
    const step = Math.max(1, Math.ceil(raw.length / 1500))
    const keep = new Set(marks)
    return raw.filter((p: any, i: number) => i % step === 0 || keep.has(p.t) || i === raw.length - 1)
  }, [values, useTr, marks, field])
  if (pts.length < 2) return <p className="hint">그릴 값이 부족합니다.</p>
  const ys = pts.map((p: any) => p.y), lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1
  const x = (i: number) => P + (i / (pts.length - 1)) * (W - P - 8)
  const y = (v: number) => 8 + (1 - (v - lo) / span) * (H - 28)
  const d = pts.map((p: any, i: number) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.y).toFixed(1)}`).join('')
  const mk = new Set(marks)
  return <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, height: 'auto' }} role="img" aria-label="값 그래프">
    <line x1={P} x2={W - 8} y1={H - 20} y2={H - 20} stroke="var(--line)" />
    <text x={2} y={y(hi) + 4} fontSize="11" fill="var(--ink-2)">{fmt(hi)}</text>
    <text x={2} y={y(lo) + 4} fontSize="11" fill="var(--ink-2)">{fmt(lo)}</text>
    <text x={P} y={H - 4} fontSize="11" fill="var(--ink-2)">{pts[0].t}</text>
    <text x={W - 8} y={H - 4} fontSize="11" fill="var(--ink-2)" textAnchor="end">{pts[pts.length - 1].t}</text>
    <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.2" />
    {pts.map((p: any, i: number) => mk.has(p.t) ? <circle key={p.t} cx={x(i)} cy={y(p.y)} r="4" fill="none" stroke="var(--fail)" strokeWidth="1.5"><title>{p.t}: {fmt(p.y)}</title></circle> : null)}
  </svg>
}

function Edit({ d, onSaved, options }: { d: any; onSaved: () => void; options: TransformOption[] }) {
  const [lag, setLag] = useState(String(d.lag_days))
  const [conf, setConf] = useState<boolean>(d.lag_confirmed)
  const [tr, setTr] = useState<string>(d.transform)
  const [basis, setBasis] = useState('')
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const save = async () => {
    setBusy(true); setMsg(null)
    try {
      const r = await api.varEdit(d.id, { lag_days: lag === '' ? null : Number(lag), lag_confirmed: conf, transform: tr.trim(), basis })
      setMsg({ ok: true, text: r.note }); setBasis(''); onSaved()
    } catch (e: any) { setMsg({ ok: false, text: e.message }) }
    setBusy(false)
  }
  return <details className="varedit"><summary><b>발표시차·변환 고치기</b></summary>
    <p className="hint">변수 사전 엑셀(<code>data/variable_master.xlsx</code>)의 이 변수 줄을 고칩니다. 고치기 전 사전 사본이 <code>data/master_history/</code> 에 남고, 비고에 날짜·바꾼 내용·근거가 적힙니다. 다음 실험부터 <b>기본값</b>으로 쓰이며, 지난 결과는 그때 쓴 값이 기록돼 있어 바뀌지 않습니다. 받아 둔 데이터(발표된 원래 값)도 바뀌지 않습니다. 한 실험에서만 바꿔 보려면 여기가 아니라 실험 설정 화면의 변환·발표시차 칸을 쓰세요(도움말 20장 4-5).</p>
    <div className="grid">
      <div className="field"><label>발표시차(일) <span className="hint">기준일 → 공개일</span></label><input value={lag} onChange={e => setLag(e.target.value.replace(/[^0-9]/g, ''))} /></div>
      <div className="field"><label>확인 여부</label><label style={{ fontWeight: 'normal' }}><input type="checkbox" checked={conf} onChange={e => setConf(e.target.checked)} /> 공표일정으로 확인함(✓)</label></div>
      <div className="field"><label>모델에 넣는 꼴</label>
        <TransformSelect value={tr} onChange={setTr} options={options} /></div>
      <div className="field" style={{ gridColumn: 'span 2' }}><label>근거 <span className="hint">예: 공표일정 주소·보도자료</span></label><input value={basis} onChange={e => setBasis(e.target.value)} /></div>
    </div>
    <button className="primary" onClick={save} disabled={busy}>저장</button>
    {msg && <p className="hint" style={{ color: msg.ok ? 'var(--pass)' : 'var(--fail)' }}>{msg.text}</p>}
  </details>
}


/** 종목 가격 [값 보기] (2026-11-13): 실험이 읽는 조립된 가격 그대로 + 점검 + CSV. 고치기는 없음(가격은 출처에서만 온다). */
export function TickerValues({ id }: { id: string }) {
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { setD(null); setErr(null); api.tickerValues(id).then(setD).catch(e => setErr(e.message)) }, [id])
  if (err) return <p className="hint" style={{ color: 'var(--fail)' }}>{err}</p>
  if (!d) return <p className="hint">값을 읽는 중…</p>
  const c = d.checks
  return <div className="varvals">
    <p><b>{d.id}</b> {d.name} · 데이터 버전 <code>{d.version}</code>{d.n > 0 && <> · {d.n.toLocaleString()}일 ({d.first} ~ {d.last}) · <a href={`/api/tickers/${encodeURIComponent(d.id)}/values.csv?version=${encodeURIComponent(d.version)}`} download>엑셀로 내려받기(CSV)</a></>}</p>
    {d.message && <p className="hint" style={{ color: 'var(--warn)' }}>{d.message}</p>}
    {d.n > 0 && <>
      <Checks c={c} freq="D" />
      <ul style={{ margin: '-4px 0 8px 18px' }}>
        {c.ohlc_bad > 0 ? <li><span className="badge fail">고가·저가 앞뒤 안 맞음 {c.ohlc_bad}일</span> {c.ohlc_bad_dates.join(' · ')}</li> : <li className="hint">고가·저가는 모든 날 시가·종가를 감쌉니다.</li>}
        <li className="hint">빠진 거래일을 네이버 수정주가로 메운 날: {c.repaired}일 (데이터 버전 폴더의 repair_log 에 날짜별로 있음)</li>
      </ul>
      <Line values={d.values} useTr={false} field="close" marks={(c.reversals ?? []).map((x: any) => x.date)} />
      <details><summary className="hint">최근 30일 (전체는 CSV)</summary>
        <table className="mini"><thead><tr><th>날짜</th><th>시가</th><th>고가</th><th>저가</th><th>종가</th><th>거래량</th></tr></thead>
          <tbody>{d.values.slice(-30).reverse().map((r: any) => <tr key={r.date}><td>{r.date}</td>{['open', 'high', 'low', 'close', 'volume'].map(k => <td key={k} className="num">{fmt(r[k])}</td>)}</tr>)}</tbody></table>
      </details>
    </>}
  </div>
}



// ---------------------------------------------------------------------------
// 자료 점검표(2026-11-14) — 정상성·계절성·계절조정 표시. 값을 바꾸지 않는 검사(engine/datacheck.py 머리말과 같은 규칙).
// ---------------------------------------------------------------------------
const SIG: Record<string, string> = { '초록': 'pass', '노랑': 'warn', '회색': 'muted' }
const ST_HINT = 'ADF(귀무 = 추세 있음)와 KPSS(귀무 = 정상)를 함께 봄: ADF p<0.05 이고 KPSS p>0.05 → 정상, 반대 → 추세 남음, 그 밖 → 판단 어려움'

export function DataCheckLine({ dc }: { dc: any }) {
  if (!dc) return null
  const st = dc.stationarity || {}, se = dc.seasonality || {}
  return <div className="checks">
    <p><b>자료 점검</b> <span className={'badge ' + (SIG[dc.signal] || 'muted')}>{dc.signal}</span> <span className="hint">— 모델이 받는 꼴({dc.transform})로 검사합니다. 알림일 뿐 값·설정을 바꾸지 않습니다.</span></p>
    <ul>
      <li title={ST_HINT}>정상성(추세): <b>{st.verdict}</b>{st.adf_p_text != null && <span className="hint"> · ADF p {st.adf_p_text} · KPSS p {st.kpss_p_text} · 값 {st.n}개</span>} <span className="hint">— {st.why}</span></li>
      <li>계절성: <b>{se.verdict}</b>{se.p_text && <span className="hint"> · F 검정 p {se.p_text}</span>} <span className="hint">— {se.why}</span></li>
      <li>계절조정 표시: <b>{dc.seasonal_adj}</b> <span className="hint">— {dc.seasonal_adj_basis}</span></li>
      {dc.advice?.length > 0 && <li>권고: {dc.advice.join(' · ')} <span className="hint">(바꾸지 않았습니다 — 바꾸려면 아래 '고치기'의 변환)</span></li>}
    </ul>
  </div>
}

export function DataCheckTable({ rows, upto }: { rows: any[]; upto?: string | null }) {
  const order: Record<string, number> = { '노랑': 0, '회색': 1, '초록': 2 }
  const sorted = [...rows].sort((a, b) => (order[a.signal] ?? 3) - (order[b.signal] ?? 3) || String(a.id).localeCompare(String(b.id)))
  return <div style={{ overflow: 'auto', maxHeight: '60vh' }}>
    {upto && <p className="hint">봉인 시작일 {upto} 앞의 값만으로 검사했습니다(봉인 구간 값이 섞이지 않게).</p>}
    <table>
      <thead><tr><th>신호</th><th>변수</th><th>주기</th><th>변환</th><th title={ST_HINT}>정상성(추세)</th><th>계절성</th><th>계절조정 표시</th><th>빈 기간</th><th>확인해 볼 날</th><th>이유 · 권고</th></tr></thead>
      <tbody>{sorted.map(r => <tr key={r.id}>
        <td><span className={'badge ' + (SIG[r.signal] || 'muted')}>{r.signal}</span></td>
        <td><b>{r.id}</b> <span className="hint">{r.name}</span></td>
        <td>{r.freq}</td><td><code>{r.transform}</code></td>
        <td title={r.stationarity?.why}>{r.stationarity?.verdict}{r.stationarity?.adf_p_text != null && <span className="hint"> (ADF {r.stationarity.adf_p_text}, KPSS {r.stationarity.kpss_p_text})</span>}</td>
        <td title={r.seasonality?.why}>{r.seasonality?.verdict}{r.seasonality?.p_text && <span className="hint"> (p {r.seasonality.p_text})</span>}</td>
        <td title={r.seasonal_adj_basis}>{r.seasonal_adj}</td>
        <td className="num">{r.gaps || ''}</td><td className="num">{r.reversals || ''}</td>
        <td style={{ whiteSpace: 'normal', minWidth: 260 }}>{r.reasons?.join(' · ')}{r.advice?.length > 0 && <div className="hint">권고: {r.advice.join(' · ')}</div>}</td>
      </tr>)}</tbody>
    </table>
  </div>
}

/** 다른 출처와 대조(2026-11-14, 표시만) — engine/datacheck.cross_pair 와 같은 규칙. */
export function CrossSourceTable({ rows }: { rows: any[] }) {
  if (!rows?.length) return null
  const pct = (x: number, kind: string) => kind?.startsWith('로그') ? `${(x * 100).toFixed(2)}%` : fmt(x)
  return <div style={{ overflow: 'auto', maxHeight: '50vh' }}>
    <table>
      <thead><tr><th>신호</th><th>변수 (출처)</th><th>대조한 변수 (출처)</th><th>맞춘 기간</th><th>겹친 기간</th><th>변화 상관</th><th>이유</th></tr></thead>
      <tbody>{rows.map(r => <tr key={r.id + r.ref_id}>
        <td><span className={'badge ' + (SIG[r.signal] || 'muted')}>{r.signal}</span></td>
        <td><b>{r.id}</b> <span className="hint">{r.name} ({r.source}, {r.freq})</span></td>
        <td><b>{r.ref_id}</b> <span className="hint">{r.ref_name} ({r.ref_source}, {r.ref_freq})</span></td>
        <td>{r.period ? (r.period === 'D' ? '같은 날' : (r.period === 'M' ? '월' : '분기') + (r.definition ? ` · 일간은 ${r.definition}` : '')) : ''}</td>
        <td className="num">{r.n || ''}</td>
        <td className="num">{r.change_corr != null ? r.change_corr.toFixed(3) : ''}</td>
        <td style={{ whiteSpace: 'normal', minWidth: 260 }}>{r.reasons?.join(' · ') || '걸린 것 없음'}
          {r.flags?.length > 0 && <details><summary className="hint">확인해 볼 기간 {r.n_flags}개 보기 (차이가 큰 순)</summary>
            <table><thead><tr><th>기간</th><th>{r.id}</th><th>{r.ref_id}</th><th>{r.diff_kind}</th><th>단서</th></tr></thead>
              <tbody>{r.flags.map((f: any) => <tr key={f.period}><td>{f.period}</td><td className="num">{fmt(f.a)}</td><td className="num">{fmt(f.b)}</td><td className="num">{pct(f.diff, r.diff_kind)}</td><td style={{ whiteSpace: 'normal' }}>{f.clue?.join(' · ')}</td></tr>)}</tbody></table>
            <p className="hint">평소 차이(중앙값) {pct(r.median_diff, r.diff_kind)}, 기준 ±{pct(r.tolerance, r.diff_kind)}. 두 출처만으로는 어느 쪽이 맞는지 알 수 없습니다 — 원 발표처를 확인하세요. 값은 바꾸지 않았습니다.</p>
          </details>}</td>
      </tr>)}</tbody>
    </table>
  </div>
}

export function DataCheckPanel() {
  const [d, setD] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const load = () => { setBusy(true); setErr(''); api.dataChecks().then(setD).catch(e => setErr(String(e))).finally(() => setBusy(false)) }
  return <div className="panel"><h3>자료 점검표</h3>
    <p className="hint">변수마다 추세(정상성)·계절성·계절조정 표시·빈 기간·확인해 볼 날을 검사합니다. 모델이 받는 꼴(변환 후)로 보며, 값과 설정은 바꾸지 않습니다. 노랑은 '그래프를 보고 판단할 것'이라는 뜻이지 쓰면 안 된다는 뜻이 아닙니다.</p>
    {!d && <button onClick={load} disabled={busy}>{busy ? '검사 중… (변수 수에 따라 10초쯤)' : '자료 점검표 보기'}</button>}
    {err && <p className="err">{err}</p>}
    {d && <>
      <p>데이터 버전 <code>{d.version}</code> · <span className="badge warn">노랑 {d.counts['노랑']}</span><span className="badge muted">회색 {d.counts['회색']}</span><span className="badge pass">초록 {d.counts['초록']}</span> <button onClick={load} disabled={busy}>다시 검사</button></p>
      <p className="hint">{d.rule}</p>
      <DataCheckTable rows={d.rows} />
      {d.cross?.length > 0 && <>
        <h4 style={{ marginTop: 16 }}>다른 출처와 대조 <span className="badge warn">노랑 {d.cross_counts['노랑']}</span><span className="badge muted">회색 {d.cross_counts['회색']}</span><span className="badge pass">초록 {d.cross_counts['초록']}</span></h4>
        <p className="hint">{d.cross_rule}</p>
        <CrossSourceTable rows={d.cross} />
      </>}
    </>}
  </div>
}
