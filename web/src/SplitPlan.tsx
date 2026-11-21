/** 검증 계획 표 — 워크포워드·1회 분할에서 실제로 몇 개씩, 어느 날짜부터 어느 날짜까지 쓰이는지,
 *  학습 창(구간)이 어디서 몇 개씩 잘려 '학습'이 남는지, 날짜 맞춤·비움·빈 칸 처리를 엔진(engine/splitplan.py)이
 *  실행과 같은 규칙으로 센 값 그대로 보여 준다(학습 없이 미리 계산, tests/test_splitplan.py 가 실행 기록과 같은지 확인). */
type Row = { part: string; first?: string | null; last?: string | null; n: number; unit: string }
type Step = { what: string; n: number; minus?: number; final?: boolean }

function Steps({ steps }: { steps: Step[] }) {
  return <table className="tbl">
    <thead><tr><th>자르는 순서</th><th>빼는 수</th><th>남는 수</th></tr></thead>
    <tbody>{steps.map((s, i) => <tr key={i} style={s.final ? { fontWeight: 600 } : undefined}>
      <td>{s.what}</td><td>{s.minus != null ? `−${s.minus}` : ''}</td><td>{s.n}</td></tr>)}</tbody>
  </table>
}

function Rows({ rows }: { rows: Row[] }) {
  return <table className="tbl">
    <thead><tr><th>구간</th><th>시작</th><th>끝</th><th>개수</th></tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}><td>{r.part}</td><td>{r.first ?? ''}</td><td>{r.last ?? ''}</td><td>{r.n} {r.unit}</td></tr>)}</tbody>
  </table>
}

function Blanks({ blanks }: { blanks: any }) {
  if (!blanks) return null
  return <details className="techlist">
    <summary><b>빈 칸 처리</b> — {blanks.how}{blanks.vars.length ? ` · 빈 칸이 있는 변수 ${blanks.vars.length}개` : ' · 이 구간에는 빈 칸 없음'}</summary>
    {blanks.vars.length > 0 && <table className="tbl">
      <thead><tr><th>변수</th><th>첫 값이 있는 날</th><th>학습 구간 빈 칸</th><th>테스트 구간 빈 칸</th></tr></thead>
      <tbody>{blanks.vars.map((b: any) => <tr key={b.var}><td><code>{b.var}</code></td><td>{b.first_valid ?? '없음'}</td><td>{b.blank_train}</td><td>{b.blank_test}</td></tr>)}</tbody>
    </table>}
  </details>
}

function List({ title, items }: { title: string; items?: string[] }) {
  if (!items || !items.length) return null
  return <div className="hint"><b>{title}</b><ul>{items.map((s, i) => <li key={i}>{s}</li>)}</ul></div>
}

export default function SplitPlan({ plan, busy, err }: { plan: any; busy: boolean; err: string | null }) {
  if (err) return <div className="verdict warn">검증 계획을 계산하지 못했습니다: {err}</div>
  if (!plan) return busy ? <p className="hint">검증 계획 계산 중…</p> : null
  const head = <h4 className="sub">실제로 쓰이는 구간 <span className="hint">— 엔진이 실행할 때와 같은 규칙으로 센 값 (학습 없이 미리 계산)</span></h4>
  if (plan.mode === 'walkforward') {
    const ts = plan.test_span
    return <div className="split-plan" style={{ opacity: busy ? 0.6 : 1 }}>
      {head}
      {(plan.blocking ?? []).map((b: string, i: number) => <div key={i} className="verdict warn">{b}</div>)}
      <p className="hint">
        폴드 {plan.n_folds}개{plan.skipped_folds?.length ? ` 중 ${ts.n_folds_run}개 실행(${plan.skipped_folds.length}개 건너뜀)` : ''} · {plan.window.kind}.
        {ts.first && <> 성적에 쓰는 테스트 = {ts.first} ~ {ts.last} ({ts.n}개 결정일, 폴드마다 {plan.window.test_len}개씩 이어 붙임).</>}
      </p>
      {(plan.folds ?? []).map((f: any) => <div key={f.name}>
        <h4 className="sub">{f.name}</h4>
        <Steps steps={f.steps} />
        <Rows rows={f.rows} />
      </div>)}
      {(plan.folds ?? []).length > 1 && <p className="hint">가운데 폴드들도 같은 규칙입니다 — 창이 {plan.window.test_len}개씩 밀릴 뿐, 자르는 순서·개수는 빈 칸·정답 없는 행이 없으면 첫 폴드와 같습니다.</p>}
      <List title="빠지는 것·비우는 것" items={plan.notes} />
      <Blanks blanks={plan.blanks} />
    </div>
  }
  return <div className="split-plan" style={{ opacity: busy ? 0.6 : 1 }}>
    {head}
    {(plan.blocking ?? []).map((b: string, i: number) => <div key={i} className="verdict warn">{b}</div>)}
    {plan.steps && <Steps steps={plan.steps} />}
    <Rows rows={plan.rows ?? []} />
    <p className="hint">
      바탕: {plan.base?.what} {plan.base?.n}개({plan.base?.first} ~ {plan.base?.last}).
      {plan.ratio && <> 비율 {plan.ratio.requested.join(' : ')} → 개수 {plan.ratio.counts.join(' : ')} ({plan.ratio.rule}).</>}
      {plan.used_pct && <> 비움·창 앞부분·빈 칸을 뺀 뒤 실제 사용 비율: 학습 {plan.used_pct['학습']}% · 검증 {plan.used_pct['검증']}% · 테스트 {plan.used_pct['테스트']}%.</>}
    </p>
    <List title="날짜 맞춤 — 입력한 날짜가 결정일이 아니면 가장 가까운 결정일로 옮깁니다" items={plan.snapped} />
    <List title="빠지는 것·비우는 것" items={plan.notes} />
    <Blanks blanks={plan.blanks} />
  </div>
}
