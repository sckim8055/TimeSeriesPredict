/** 봉인 평가 — 후보(개발 결과) 하나를 '처음 보는 기간'에서 한 번만 확인한다.
 *  SealedPanel: 개발 결과 상세 화면의 '봉인 평가' 칸(미리보기 → 사전 기록 → 확인 → 실행)
 *  SealedInfo:  봉인 결과 상세 화면의 '봉인 평가 기록'(후보와 나란히 비교)
 *  SealedLog:   결과 화면 위의 '봉인 기록'(지금까지 몇 번 열었나, 지울 수 없음)
 *  실행은 서버 POST /api/sealed/run → 작업 큐 → python cli.py run 설정 --sealed (명령줄과 같은 경로). */
import { useEffect, useState } from 'react'
import { api, num, pct, type Mode } from './api'

const T = (s?: string | null) => (s ?? '').slice(0, 16).replace('T', ' ')

/** 개발 결과와 봉인 결과를 같은 줄에 놓는 지표들(둘 다 값이 없는 줄은 뺀다) */
function metricRows(isF: boolean, a: any, b?: any): [string, string, string?][] {
  const P = (x: any) => x?.prediction_metrics ?? {}, F = (x: any) => x?.performance ?? {}
  const both = (f: (x: any) => string) => [f(a), b ? f(b) : undefined] as [string, string | undefined]
  const has = (f: (x: any) => any) => f(a) != null || (b && f(b) != null)
  const rows: [string, string, string?][] = []
  const add = (label: string, test: (x: any) => any, f: (x: any) => string) => { if (has(test)) rows.push([label, ...both(f)]) }
  add('평가 기간', x => x?.evaluation_period, x => (x?.evaluation_period ?? []).join(' ~ '))
  add('폴드 · 예측 수', x => x?.n_folds, x => `${x?.n_folds ?? ''} · ${x?.n_predictions ?? ''}`)
  add('방향 맞힘', x => P(x).dir_acc, x => pct(P(x).dir_acc) + (P(x).dir_acc_naive != null ? ` (naive ${pct(P(x).dir_acc_naive)})` : ''))
  add('RMSE', x => P(x).rmse, x => num(P(x).rmse, 5) + (P(x).rmse_naive != null ? ` (naive ${num(P(x).rmse_naive, 5)})` : ''))
  add('DM vs 0예측 (p)', x => P(x).dm_vs_zero, x => `${num(P(x).dm_vs_zero, 2)} (p ${num(P(x).dm_vs_zero_p)})`)
  add('DM vs naive (p)', x => P(x).dm_vs_naive, x => `${num(P(x).dm_vs_naive, 2)} (p ${num(P(x).dm_vs_naive_p)})`)
  add('AUC', x => P(x).auc, x => num(P(x).auc))
  if (!isF) {
    add('샤프', x => F(x).sharpe, x => num(F(x).sharpe))
    add('CAGR (같은 구간 B&H)', x => F(x).cagr, x => `${pct(F(x).cagr)} (${pct(F(x).benchmark_cagr)})`)
    add('MDD (같은 구간 B&H)', x => F(x).mdd, x => `${pct(F(x).mdd)} (${pct(F(x).benchmark_mdd)})`)
    add('섀넌 50:50 기준 (같은 구간) 샤프 · CAGR', x => F(x).shannon?.sharpe, x => F(x).shannon ? `${num(F(x).shannon.sharpe)} · ${pct(F(x).shannon.cagr)}` : '-')
    add('초과 총수익 (전략 − B&H)', x => F(x).excess_total_return, x => pct(F(x).excess_total_return))
    add('초과수익 p (블록 부트스트랩)', x => F(x).bootstrap_p_excess, x => num(F(x).bootstrap_p_excess))
    add('Deflated Sharpe p (시도 수)', x => F(x).deflated_sharpe_p, x => `${num(F(x).deflated_sharpe_p)} (${F(x).n_trials_so_far ?? ''})`)
    add('전환 횟수 · 노출', x => F(x).n_trades, x => `${F(x).n_trades ?? ''} · ${pct(F(x).exposure, 0)}`)
  }
  return rows
}

export function SealedExplain() {
  return <details className="sealed-explain"><summary><b>봉인 평가가 하는 일</b> — 순서대로 자세히</summary>
    <ol>
      <li><b>무엇을 확인하나</b>: 개발 구간(봉인 시작일 전)에서 본 성적이 <b>한 번도 보지 않은 기간</b>(봉인 시작일 ~ 데이터 끝)에서도 나오는지. 개발 중에는 이 기간의 예측·성과를 만들지 않았습니다(엔진이 지움).</li>
      <li><b>설정은 후보 그대로</b>: 이 결과 파일에 기록된 설정(변수·변환·발표시차·모델 파라미터·검증 규칙·비용·시드·데이터 버전)을 그대로 씁니다. 바꾸는 것은 없습니다. 단, 그 뒤에 기본값이 바뀐 파라미터는 그 결과를 낼 때의 값으로 고정합니다(아래 '옛 기본값 고정' 줄에 표시).</li>
      <li><b>학습 방법도 개발 때와 같은 규칙</b>: 워크포워드면 봉인 첫날부터 '학습 창 → 비움 → 테스트'를 재학습 주기마다 밀어 가며 데이터 끝까지(실전에서 매달 다시 학습하는 것과 같음). 1회 분할이면 '봉인 시작 전 전부 − 비움'으로 한 번 학습해 봉인 구간 전체를 예측. 규칙 모델은 학습 없이 봉인 구간의 포지션만 채점.</li>
      <li><b>개발 구간은 다시 계산하지 않습니다</b>: 결과에는 봉인 구간의 예측·성과만 들어갑니다(학습 창은 봉인 전 데이터를 씀 — 정상).</li>
      <li><b>기록</b>: 끝나면 결과가 '…_sealed' 이름으로 따로 저장되고(개발 결과를 덮어쓰지 않음), <code>results/{'{trading|forecast}'}/sealed_log.json</code> 에 언제·어떤 설정(해시)·몇 번째·사전 기록·결과 요약이 남습니다. 이 기록은 화면에서 지울 수 없습니다.</li>
      <li><b>실험 카운터는 올리지 않습니다</b>: 봉인 평가는 새 시도가 아니라 확인이므로, Deflated Sharpe 의 시도 수에는 지금까지의 개발 시도 수를 씁니다.</li>
      <li><b>한 번만</b>: 같은 설정을 다시 열면(재개봉) 그 결과를 본 뒤 고른 것이 되어 봉인의 의미가 없어집니다. 서버는 재개봉을 따로 확인받고, 결과에 '몇 번째'를 남깁니다. 결과를 보고 후보를 고쳐 다시 여는 것도 같은 문제입니다 — <b>봉인을 연 가설은 그 결과로 끝(통과 또는 폐기)</b>.</li>
      <li><b>여러 후보를 열면</b>: 같은 봉인 시작일로 후보 여러 개를 열고 그중 좋은 것을 고르면 봉인 구간에서도 다중검정이 생깁니다. 열어 본 후보 수를 논문에 함께 보고하세요(아래 기록에 그대로 남음).</li>
    </ol>
  </details>
}

export function SealedPanel({ stem, r }: { stem: string; r: any }) {
  const [pv, setPv] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [note, setNote] = useState('')
  const [ack1, setAck1] = useState(false), [ack2, setAck2] = useState(false), [ack3, setAck3] = useState(false)
  const [done, setDone] = useState<any>(null)
  useEffect(() => { setPv(null); setErr(''); setDone(null); setNote(''); setAck1(false); setAck2(false); setAck3(false) }, [stem])
  const isF = r?.config?.mode === 'forecast'
  const sf = r?.config?.validation?.sealed_from
  async function load() {
    setBusy(true); setErr('')
    try { setPv(await api.sealedPreview(stem)) } catch (e: any) { setErr(/not found|404|method not allowed|405/i.test(String(e.message)) ? '서버가 예전 코드로 돌고 있습니다 — run_server.bat 창을 닫고 다시 실행한 뒤 새로고침하세요.' : String(e.message ?? e)) }
    finally { setBusy(false) }
  }
  async function run() {
    setBusy(true); setErr('')
    try { setDone(await api.sealedRun(stem, note, reopen)) } catch (e: any) { setErr(String(e.message ?? e)) } finally { setBusy(false) }
  }
  const reopen = !!pv?.history?.same_config?.length
  const can = pv && !pv.blockers.length && ack1 && ack2 && (!reopen || ack3) && !done
  const pl = pv?.plan
  return <div className="panel sealed-panel">
    <h3>봉인 평가 — 이 후보를 처음 보는 기간에서 한 번 확인</h3>
    <p className="hint">봉인 시작일 <b>{sf ?? '없음'}</b>{sf ? ' 부터 데이터 끝까지가 봉인 구간입니다. 봉인 시작일은 새 실험 화면의 "봉인 시작일" 칸에서, 개발 실험을 돌리기 전에 정합니다(결과를 본 뒤 옮기면 봉인이 아닙니다).' : ' — 이 후보는 봉인 구간 없이 개발했습니다.'}</p>
    <SealedExplain />
    {!pv && <div style={{ marginTop: 10 }}><button onClick={load} disabled={busy}>{busy ? '계산 중…' : '봉인 평가 준비 (미리보기 — 학습·저장 없음)'}</button></div>}
    {err && <p className="badge fail" style={{ whiteSpace: 'normal' }}>{err}</p>}
    {pv && <>
      {pv.blockers.length > 0 && <div className="verdict fail"><b>실행할 수 없습니다</b><ul>{pv.blockers.map((b: string) => <li key={b}>{b}</li>)}</ul></div>}
      {pl && <>
        <h4>① 봉인 구간과 학습 방법 (실행과 같은 함수로 계산)</h4>
        <dl className="kv"><dt>봉인 구간</dt><dd><b>{pl.first_test} ~ {pl.last_test}</b> · {pl.n_days}{isF ? '기간' : '거래일'} (봉인 시작일 {pl.sealed_from}, 데이터 끝 {pl.data_end})</dd>
          <dt>방법</dt><dd>{pl.how}</dd>
          {pl.n_folds > 0 && <><dt>폴드</dt><dd>{pl.n_folds}개 · 첫 폴드 학습 {pl.folds[0].train.join(' ~ ')} ({pl.folds[0].n_train}) → 테스트 {pl.folds[0].test.join(' ~ ')} ({pl.folds[0].n_test})
            {pl.n_folds > 1 && <><br />마지막 폴드 학습 {pl.folds[pl.n_folds - 1].train.join(' ~ ')} → 테스트 {pl.folds[pl.n_folds - 1].test.join(' ~ ')} ({pl.folds[pl.n_folds - 1].n_test})</>}</dd></>}
          <dt>비움(embargo)</dt><dd>{pl.embargo}행 — 학습 정답이 테스트 기간의 가격을 쓰지 않도록 학습 창 끝에서 뺌</dd>
          {pv.estimate_seconds != null && <><dt>예상 시간</dt><dd>약 {pv.estimate_seconds < 90 ? `${pv.estimate_seconds}초` : `${(pv.estimate_seconds / 60).toFixed(1)}분`} (개발 결과의 폴드당 시간 × 봉인 폴드 수, 장치에 따라 다름)</dd></>}
        </dl>
        {pl.folds?.length > 0 && <details><summary className="hint">폴드 전체 ({pl.folds.length}개)</summary>
          <table className="sealed-folds"><thead><tr><th>#</th><th>학습</th><th className="num">학습 수</th><th>테스트</th><th className="num">테스트 수</th></tr></thead>
            <tbody>{pl.folds.map((f: any) => <tr key={f.no}><td>{f.no}</td><td>{f.train.join(' ~ ')}</td><td className="num">{f.n_train}</td><td>{f.test.join(' ~ ')}</td><td className="num">{f.n_test}</td></tr>)}</tbody></table></details>}
      </>}
      <h4>② 후보의 개발 구간 성적 (봉인 전)</h4>
      <table className="sealed-cmp"><thead><tr><th>지표</th><th>개발 구간</th></tr></thead><tbody>{metricRows(isF, { ...pv.dev, n_predictions: r.n_predictions }).map(([k, a]) => <tr key={k}><td>{k}</td><td>{a}</td></tr>)}</tbody></table>
      <p className="hint">{pv.criteria}</p>
      <h4>③ 봉인 기록</h4>
      <p className="hint">이 설정(해시 {pv.config_hash}) {pv.history.same_config.length}회 · 같은 봉인 시작일 {pv.history.same_seal_date.length}회 · 이 화면({isF ? '예측 연구' : '매매'}) 전체 {pv.history.total}회 열었습니다.</p>
      {pv.warnings.length > 0 && <><h4>④ 확인할 점</h4><ul className="sealed-warn">{pv.warnings.map((w: string) => <li key={w}>{w}</li>)}</ul></>}
      {!pv.blockers.length && !done && <>
        <h4>⑤ 사전 기록 (열기 전에 적기)</h4>
        <p className="hint">봉인 결과를 보기 전에 '무엇이 나오면 통과·실패로 볼지'를 적어 두면 결과를 본 뒤 기준을 바꾸지 않았다는 증거가 됩니다. 기록과 결과 파일에 그대로 남습니다(2000자까지).</p>
        <textarea className="sealed-note" rows={4} value={note} onChange={e => setNote(e.target.value)} placeholder={isF ? "예) 합격: 봉인 구간에서 naive 대비 DM < 0 (오차가 naive 보다 작음), p 는 참고. 실패면 이 가설은 폐기." : "예) 합격: 봉인 구간 초과수익(전략 − B&H) > 0 이고 방향정확도 > 50%. 실패면 이 가설은 폐기하고 다시 열지 않음."} />
        <label className="sealed-ack"><input type="checkbox" checked={ack1} onChange={e => setAck1(e.target.checked)} /> 결과가 좋든 나쁘든 이 후보의 최종 결과로 보고합니다.</label>
        <label className="sealed-ack"><input type="checkbox" checked={ack2} onChange={e => setAck2(e.target.checked)} /> 봉인 결과를 보고 이 후보(변수·파라미터·기간)를 고쳐 다시 열지 않습니다.</label>
        {reopen && <label className="sealed-ack warnline"><input type="checkbox" checked={ack3} onChange={e => setAck3(e.target.checked)} /> 이 설정은 이미 {pv.history.same_config.length}회 열었습니다. 재개봉이라는 사실이 기록·결과에 남는 것을 압니다.</label>}
        <div style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="primary" disabled={!can || busy} onClick={run}>{busy ? '보내는 중…' : reopen ? '재개봉 실행' : '봉인 열기 (실행)'}</button>
          <button onClick={load} disabled={busy}>미리보기 다시 계산</button>
          {!can && !busy && <span className="hint">{pv.blockers.length ? '막는 이유를 먼저 해결하세요' : '위 확인 칸을 모두 체크하면 실행할 수 있습니다'}</span>}
        </div>
      </>}
      {done && <div className="verdict pass">작업 {done.job.id} 를 큐에 넣었습니다(설정 해시 {done.config_hash}). '작업' 화면에서 진행과 로그를 볼 수 있고, 끝나면 결과 목록에 <b>봉인</b> 표시로 따로 나타납니다(이 개발 결과는 그대로).</div>}
    </>}
  </div>
}

export function SealedInfo({ r }: { r: any }) {
  const si = r.sealed_info
  const [src, setSrc] = useState<any>(null)
  useEffect(() => { setSrc(null); if (si?.source_result) api.result(si.source_result).then(setSrc).catch(() => setSrc(false)) }, [si?.source_result])
  const isF = r.config?.mode === 'forecast'
  if (!si) return <div className="panel"><h3>봉인 평가 결과</h3><p className="hint">이 결과는 봉인 구간만 평가한 것입니다(명령줄 --sealed 로 실행, 화면 기록 없음).</p></div>
  return <div className="panel sealed-panel">
    <h3>봉인 평가 결과 — {si.opening_no_same_config > 1 ? <span className="badge fail">재개봉 {si.opening_no_same_config}회째</span> : <span className="badge pass">이 설정의 첫 개봉</span>}</h3>
    <dl className="kv"><dt>연 시각(UTC)</dt><dd>{T(si.opened_utc)}{si.job_id ? ` · 작업 ${si.job_id}` : ''}</dd>
      <dt>후보(개발 결과)</dt><dd>{si.source_result ?? '(명령줄 실행 — 기록 없음)'}</dd>
      <dt>몇 번째</dt><dd>이 설정 {si.opening_no_same_config}회째 · 같은 봉인 시작일 {si.opening_no_same_seal_date}회째 · 이 화면 전체 {si.opening_no_mode}회째</dd>
      <dt>개발 시도 수</dt><dd>{si.dev_trials} (Deflated Sharpe 보정에 쓴 시도 수 — 봉인 평가는 시도로 세지 않음)</dd>
      <dt>사전 기록</dt><dd style={{ whiteSpace: 'pre-wrap' }}>{si.note || '(적지 않음)'}</dd></dl>
    <h4>개발 구간 vs 봉인 구간</h4>
    {src === null && si.source_result && <p className="hint">후보 결과를 불러오는 중…</p>}
    {src === false && <p className="hint">후보 결과 파일({si.source_result})을 찾지 못했습니다(지워졌을 수 있음).</p>}
    <table className="sealed-cmp"><thead><tr><th>지표</th><th>개발 구간 (후보)</th><th>봉인 구간 (이 결과)</th></tr></thead>
      <tbody>{metricRows(isF, src || {}, r).map(([k, a, b]) => <tr key={k}><td>{k}</td><td>{src ? a : ''}</td><td><b>{b}</b></td></tr>)}</tbody></table>
    <p className="hint">판정은 사전 기록에 적은 기준으로 하세요. 봉인 구간은 개발 구간보다 짧아 p값이 커지기 쉽습니다 — 짧은 기간의 '유의하지 않음'은 '효과 없음'과 같지 않고, 기간 길이({r.n_predictions}개)를 함께 보고해야 합니다.</p>
  </div>
}

export function SealedLog({ mode }: { mode: Mode }) {
  const [log, setLog] = useState<any[] | null>(null)
  useEffect(() => { api.sealedLog(mode).then(setLog).catch(() => setLog([])) }, [mode])
  if (!log) return null
  // 실패한 시도(봉인 수치를 보기 전에 오류로 멈춘 것)는 개봉으로 세지 않는다. 시작만 하고 끝나지 않은 것(실행 중·강제 종료)은 센다 — 엔진의 규칙(engine/sealed.history)과 같다
  const opened = log.filter(e => (e.status ?? 'done') !== 'failed')
  const stTxt = (e: any) => ({ done: '끝남', started: '끝나지 않음(실행 중이거나 강제 종료 — 개봉으로 셈)', failed: '실패(수치를 보기 전에 멈춤 — 세지 않음)' } as any)[e.status ?? 'done'] ?? e.status
  return <details className="panel sealed-log"><summary><b>봉인 기록</b> — 지금까지 {opened.length}회 열었습니다{log.length > opened.length ? ` (실패한 시도 ${log.length - opened.length}건은 세지 않음)` : ''} (지울 수 없음 · {mode === 'forecast' ? '예측 연구' : '매매'})</summary>
    {log.length === 0 ? <p className="hint">아직 연 적이 없습니다. 결과 하나를 골라 상세 화면 아래 '봉인 평가'에서 엽니다.</p> :
      <table><thead><tr><th>#</th><th>연 시각(UTC)</th><th>상태</th><th>설정</th><th>모델</th><th>봉인 시작</th><th>봉인 결과</th><th>후보</th><th>사전 기록</th></tr></thead>
        <tbody>{log.map((e, i) => <tr key={i}><td>{i + 1}</td><td>{T(e.when)}</td><td>{stTxt(e)}</td><td>{e.config} <span className="hint">{e.hash}</span></td><td>{e.model ?? ''}</td><td>{e.sealed_from ?? ''}</td><td>{e.result ?? ''}</td><td>{e.source_result ?? ''}</td><td style={{ maxWidth: 320, whiteSpace: 'pre-wrap' }}>{e.note ?? ''}</td></tr>)}</tbody></table>}
  </details>
}
