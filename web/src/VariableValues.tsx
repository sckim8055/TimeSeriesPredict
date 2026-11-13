import { useEffect, useMemo, useState } from 'react'
import { api } from './api'

/** 변수 사전의 [값 보기] (2026-11-13): 데이터 버전에 저장된 값 그대로 + 점검 + CSV 내려받기 + 발표시차·변환 고치기. */
export default function VariableValues({ id, onChanged }: { id: string; onChanged?: () => void }) {
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
    <Edit d={d} onSaved={() => { setRev(rev + 1); onChanged?.() }} />
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

function Edit({ d, onSaved }: { d: any; onSaved: () => void }) {
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
    <p className="hint">변수 사전 엑셀(<code>data/variable_master.xlsx</code>)의 이 변수 줄을 고칩니다. 고치기 전 사전 사본이 <code>data/master_history/</code> 에 남고, 비고에 날짜·바꾼 내용·근거가 적힙니다. 다음 실험부터 적용되며, 지난 결과는 그때 쓴 값이 기록돼 있어 바뀌지 않습니다.</p>
    <div className="grid">
      <div className="field"><label>발표시차(일) <span className="hint">기준일 → 공개일</span></label><input value={lag} onChange={e => setLag(e.target.value.replace(/[^0-9]/g, ''))} /></div>
      <div className="field"><label>확인 여부</label><label style={{ fontWeight: 'normal' }}><input type="checkbox" checked={conf} onChange={e => setConf(e.target.checked)} /> 공표일정으로 확인함(✓)</label></div>
      <div className="field"><label>모델에 넣는 꼴</label>
        <select value={['none', 'pct', 'diff', 'log', 'logdiff'].includes(tr) ? tr : 'other'} onChange={e => setTr(e.target.value === 'other' ? 'pct5' : e.target.value)}>
          <option value="pct">변화율 (pct) — 가격·지수</option><option value="diff">차이 (diff) — 금리처럼 % 단위</option><option value="none">그대로 (none)</option>
          <option value="log">로그 (log)</option><option value="logdiff">로그 차이 (logdiff)</option><option value="other">직접 입력 (예: pct5)</option>
        </select>
        {!['none', 'pct', 'diff', 'log', 'logdiff'].includes(tr) && <input value={tr} onChange={e => setTr(e.target.value)} placeholder="pct5, diff3 …" />}</div>
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

