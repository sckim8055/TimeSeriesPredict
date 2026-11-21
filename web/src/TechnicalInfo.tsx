/** 기술지표(자동 계산) 판 — 변수 사전 화면(2026-11-14). 변수 사전(엑셀)에 없는 A_ 입력 18개가 무엇이고 왜 있는지.
 *  내용은 서버 /api/technical-info = engine/features.py 의 TECH_GROUP_DOCS·PRESETS·technical_example(도움말 20장 5절과 같은 글). */
import { useEffect, useState } from 'react'
import { api } from './api'

export default function TechnicalInfo() {
  const [freq, setFreq] = useState('D')
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState('')
  useEffect(() => { setErr(''); api.technicalInfo(freq).then(setD).catch(e => setErr(String(e))) }, [freq])
  const ex = d?.example
  return <div className="panel"><h3>기술지표(자동 계산 — 고칠 수 없음)</h3>
    <p><b>A_ 로 시작하는 입력은 예측 대상 종목(091160) 자신의 종가·거래량으로, 실험을 돌릴 때 계산하는 값</b>입니다. 밖에서 받아 오는 자료가 아니어서 위 변수 사전(엑셀)에 없고, 발표시차·변환도 없습니다. 모델에게는 변수와 똑같은 입력 한 칸입니다.</p>
    <p className="hint">왜 넣나: 변수 사전의 변수는 금리·환율·경기 같은 <b>바깥 정보</b>, 기술지표는 <b>그 종목 가격 자신의 정보</b>(최근 흐름·평균 대비 위치·출렁임·거래 활기)입니다. 둘을 함께 또는 따로 넣어 보면 "바깥 정보가 가격 자신의 정보보다 나은가?"를 같은 잣대로 비교할 수 있습니다. 미래를 맞힌다는 보장은 없습니다 — 그것은 실험으로 확인할 대상입니다. 모든 지표는 그날 장 마감 가격까지만 씁니다. 실험 설정의 '기술지표 블록'에서 묶음별로 켜고 끕니다. 자세한 설명은 도움말 20장 5절.</p>
    {err && <p className="err">{err}</p>}
    {d && <>
      <p>모델 주기: {[['D', '일간'], ['W', '주간'], ['M', '월간']].map(([k, l]) => <label key={k} style={{ marginRight: 12, fontWeight: 'normal' }}><input type="radio" checked={freq === k} onChange={() => setFreq(k)} /> {l}</label>)}
        <span className="hint">— 이름 끝 숫자는 창 길이(몇 개 관측을 보나). 주기마다 다른 창을 쓰지만 같은 달력 길이는 아닙니다(아래 '누가 정했나'). 예측 연구 모드는 거래량 묶음이 빠집니다.</span></p>
      <table><thead><tr><th>열 이름</th><th>묶음</th><th>계산</th></tr></thead>
        <tbody>{d.columns.map((c: any) => <tr key={c.name}><td><code>{c.name}</code></td><td>{c.group}</td><td style={{ whiteSpace: 'normal' }}>{c.desc}</td></tr>)}</tbody></table>
      <h4 style={{ marginTop: 16 }}>묶음별 설명</h4>
      {d.groups.map((g: any) => <details key={g.group} className="apidet"><summary><b>{g.group}</b> <code>{g.prefix}…</code> — {g.why.split('. ')[0]}.</summary>
        <ul>
          <li><b>왜 있나</b>: {g.why}</li>
          <li><b>쉽게 말하면</b>: {g.analogy}</li>
          <li><b>읽는 법</b>: {g.read}</li>
          <li><b>계산</b>: {g.formula}</li>
          <li><b>주의</b>: {g.caution}</li>
          <li><b>누가 만들었나·기간은 어디서</b>: {g.origin}</li>
        </ul></details>)}
      {d.origin && <details className="apidet"><summary><b>이 18개와 창 길이는 누가 정했나</b> — 관례값과 Claude 가 정한 값, 데이터로 고르지 않음</summary>
        <ul><li>{d.origin.summary}</li><li>{d.origin.not_standard}</li><li>{d.origin.weekly_monthly}</li></ul>
        <table><thead><tr><th>묶음</th><th>일간(D)</th><th>주간(W)</th><th>월간(M)</th></tr></thead>
          <tbody>{d.calendar_weeks.map((r: any) => <tr key={r.key}><td>{r.label}</td>{['D', 'W', 'M'].map(f => <td key={f} style={{ whiteSpace: 'normal' }}>{r[f].periods.map((p: number, i: number) => `${p}(≈${r[f].weeks[i]}주)`).join(' · ')}</td>)}</tr>)}</tbody></table>
        <p className="hint">창 길이를 대략의 주(週) 수로 바꾼 값(일 ÷ 5, 월 × 52/12). 자세한 출처는 도움말 20장 5-9절.</p></details>}
      {ex && <details className="apidet"><summary><b>예시 — 열흘 치 가격으로 직접 계산해 보면</b> (엔진이 계산한 값)</summary>
        <p className="hint">창 길이: {ex.windows}.</p>
        <table><tbody>
          <tr><th>날</th>{ex.close.map((_: number, i: number) => <td key={i}>{i + 1}</td>)}</tr>
          <tr><th>종가</th>{ex.close.map((x: number, i: number) => <td key={i} className="num">{x}</td>)}</tr>
          <tr><th>거래량</th>{ex.volume.map((x: number, i: number) => <td key={i} className="num">{x}</td>)}</tr>
        </tbody></table>
        <table style={{ marginTop: 8 }}><thead><tr><th>10일째 지표</th><th>값</th></tr></thead>
          <tbody>{Object.entries(ex.values).map(([k, v]: any) => <tr key={k}><td><code>{k}</code></td><td className="num">{k.startsWith('A_rsi') ? v.toFixed(1) : v.toFixed(4)}</td></tr>)}</tbody></table>
        {(() => { const c: number[] = ex.close, n = c.length, m5 = c.slice(-5).reduce((a, b) => a + b, 0) / 5; return <p className="hint">
          예: A_ret_1 = {c[n - 1]} ÷ {c[n - 2]} − 1 = {ex.values.A_ret_1.toFixed(4)}(하루 {(ex.values.A_ret_1 * 100).toFixed(2)}% 상승) ·
          A_ma_ratio_5 = {c[n - 1]} ÷ 최근 5일 평균 {+m5.toFixed(4)} − 1 = {ex.values.A_ma_ratio_5.toFixed(4)}(평균보다 {(ex.values.A_ma_ratio_5 * 100).toFixed(2)}% 위). 나머지 계산은 도움말 20장 5-7.</p> })()}
      </details>}
    </>}
  </div>
}
