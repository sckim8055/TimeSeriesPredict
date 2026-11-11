import { useEffect, useMemo, useRef, useState } from 'react'
import { createChart, type IChartApi, LineSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts'
import { api, num, pct, type Mode, type ResultRow } from './api'
import { FormulaGuide } from './EasyFlow'
import { SealedInfo, SealedLog, SealedPanel } from './Sealed'

function verdict(r: ResultRow): { cls: 'pass' | 'warn' | 'fail' | 'muted'; text: string } {
  if (r.degenerate) return { cls: 'fail', text: '퇴화: 모델이 입력을 쓰지 않음' }
  if (r.leak_warning) return { cls: 'fail', text: '누출: bfill/interp' }
  if (r.mode === 'forecast') {
    const z = r.dm_vs_zero != null && r.dm_vs_zero < 0 && r.dm_p < 0.05
    const n = r.dm_vs_naive != null && r.dm_vs_naive < 0 && r.dm_naive_p < 0.05
    if (z && n) return { cls: 'pass', text: '기준 통과: 두 기준 모두 유의하게 이김' }
    if (r.dm_vs_naive != null && r.dm_vs_naive > 0 && r.dm_naive_p < 0.05) return { cls: 'fail', text: "'지난 변화 그대로'보다 유의하게 나쁨" }
    if (z) return { cls: 'muted', text: "'변화 없음'은 이기지만 '지난 변화 그대로'와 차이 없음" }
    return { cls: 'muted', text: '유의하지 않음' }
  }
  const ml = r.dm_p != null
  const passDM = ml ? (r.dm_vs_zero < 0 && r.dm_p < 0.05) : true
  const passBoot = r.p_excess != null && r.p_excess < 0.05
  const passDefl = r.deflated_p != null && r.deflated_p < 0.05
  if (passDM && passBoot && passDefl) return { cls: 'pass', text: '기준 통과(개발 구간)' }
  // DM 은 '이 방법을 이 학습 창으로 쓰면 0예측보다 정확한가'를 본다 — 추정 잡음 때문에 정보 없는 변수만 넣어도 이렇게 나온다. 변수에 정보가 있는지는 CW 로 본다
  if (ml && r.dm_vs_zero > 0 && r.dm_p < 0.05) return { cls: 'fail', text: r.cw_p != null && r.cw_p < 0.05 ? '0예측보다 오차 큼(추정 잡음) · 변수 정보는 있음(CW)' : '0예측보다 오차 큼(이 학습 창 기준)' }
  return { cls: 'muted', text: '유의하지 않음' }
}

type SortKey = string
export default function Results({ initial, mode = 'trading' }: { initial?: string | null; mode?: Mode }) {
  const isForecast = mode === 'forecast'
  const [rows, setRows] = useState<ResultRow[]>([])
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'mtime', dir: -1 })
  const [sel, setSel] = useState<string | null>(initial ?? null)
  const [filter, setFilter] = useState('')
  const [picked, setPicked] = useState<string[]>([])
  const [ensMsg, setEnsMsg] = useState('')
  const [hideBad, setHideBad] = useState(false)

  async function refresh() { setRows(await api.results(mode)) }
  useEffect(() => { refresh() }, [])
  useEffect(() => { if (initial) setSel(initial) }, [initial])

  const shown = useMemo(() => {
    let r = rows.filter(x => !x.error)
    if (filter) r = r.filter(x => JSON.stringify(x).toLowerCase().includes(filter.toLowerCase()))
    if (hideBad) r = r.filter(x => !x.degenerate && !x.leak_warning)
    return [...r].sort((a, b) => { const va = a[sort.key], vb = b[sort.key]; if (va == null) return 1; if (vb == null) return -1; return (va > vb ? 1 : va < vb ? -1 : 0) * sort.dir })
  }, [rows, sort, filter, hideBad])
  const th = (key: string, label: string, numeric = true) => <th className={numeric ? 'num' : ''} onClick={() => setSort(s => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : -1 }))}>{label}{sort.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}</th>

  return (
    <div>
      <h2>3. 결과 <span className="hint">— {isForecast ? '예측 연구' : '매매 시뮬레이션'}</span></h2>
      {isForecast ? <div className="readcard">
        <div><b>예측 오차 (모델)</b><span>k기간 뒤 <em>값</em>을 맞혔을 때 평균 몇 % 틀렸나(MAPE). 작을수록 좋음.</span></div>
        <div><b>지난 변화 그대로</b><span>"지난 k기간 변화가 한 번 더 반복된다"고 찍은 오차. 모델이 꼭 이겨야 하는 기준(naive).</span></div>
        <div><b>변화 없음</b><span>"지금 값에서 멈춘다"고 찍은 오차. 가장 쉬운 기준.</span></div>
        <div><b>보정 p (Holm)</b><span>같은 문제(대상·k·평가 구간)를 푼 결과 m개를 <b>함께 비교</b>하므로, 각 p 를 m 에 맞춰 키운 값. "모델 X 는 naive 보다 유의하게 좋다"고 쓰려면 이 값이 0.05 미만이어야 함.</span></div>
        <div><b>naive 대비 오차 배율</b><span>모델 오차 ÷ '지난 변화 그대로' 오차. <b>×1 미만이면 모델 승</b>(×0.8 = 오차 20% 작음, ×1.4 = 40% 큼). <b>판정 p값 &lt; 0.05</b> 여야 우연이 아님.</span></div>
      </div> : <p className="lead">'같은 구간의 Buy&Hold 와 비교합니다. 성과(샤프·CAGR)보다 먼저 볼 것은 판정 열입니다 — 퇴화·누출이면 수치가 좋아도 무효이고, "유의하지 않음"은 우연과 구별이 안 된다는 뜻입니다.'</p>}
      <div className="toolbar">
        <input placeholder="이름·모델·변수로 거르기" value={filter} onChange={e => setFilter(e.target.value)} style={{ width: 260 }} />
        <label><input type="checkbox" checked={hideBad} onChange={e => setHideBad(e.target.checked)} /> 퇴화·누출 숨기기</label>
        <button onClick={refresh}>새로고침</button>
        {isForecast && <>
          <span className="hint" style={{ marginLeft: 12 }}>앙상블: 왼쪽 칸으로 {picked.length}개 선택</span>
          <button disabled={picked.length < 2} onClick={async () => { try { const r = await api.ensemble(picked, 'mean'); setEnsMsg(''); setPicked([]); await refresh(); setSel(r.saved) } catch (e: any) { setEnsMsg(String(e.message ?? e)) } }}>평균으로 결합</button>
          <button disabled={picked.length < 2} onClick={async () => { try { const r = await api.ensemble(picked, 'median'); setEnsMsg(''); setPicked([]); await refresh(); setSel(r.saved) } catch (e: any) { setEnsMsg(String(e.message ?? e)) } }}>중앙값으로 결합</button>
          {ensMsg && <span className="badge fail">{ensMsg}</span>}
        </>}
        <span className="hint">{shown.length}건</span>
      </div>
      <SealedLog mode={mode} />
      <div className="panel" style={{ padding: 0, overflow: 'auto', maxHeight: '45vh' }}>
        <table>
          <thead>{isForecast ? <tr>
            <th title="앙상블에 넣을 결과 선택"></th>{th('name', '실험', false)}{th('model', '모델', false)}{th('target_variable', '대상', false)}{th('k', 'k')}{th('n_features', '변수')}
            <th>판정</th>{th('level_mape', '예측 오차 (모델)')}{th('level_mape_naive', '지난 변화 그대로')}{th('level_mape_zero', '변화 없음')}{th('skill_vs_naive', 'naive 대비 오차 배율')}{th('dir_acc', '방향 맞힘')}{th('dm_naive_p', '판정 p값')}{th('p_holm', '보정 p (Holm)')}{th('n_predictions', '예측수')}{th('mtime', '시각', false)}
          </tr> : <tr>
            {th('name', '실험', false)}{th('model', '모델', false)}{th('frequency', '주기', false)}{th('k', 'k')}{th('n_features', '변수')}
            <th>판정</th>{th('dir_acc', '방향')}{th('dm_p', 'DM p')}{th('sharpe', '샤프')}{th('cagr', 'CAGR')}{th('bh_cagr', 'B&H')}{th('mdd', 'MDD')}{th('p_excess', '초과 p')}{th('deflated_p', 'Defl p')}{th('n_trades', '전환')}{th('trials', '시도#')}{th('mtime', '시각', false)}
          </tr>}</thead>
          <tbody>
            {shown.map(r => { const v = verdict(r); return isForecast ? (
              <tr key={r.file} className={sel === r.file ? 'sel' : ''} onClick={() => setSel(r.file)}>
                <td onClick={e => e.stopPropagation()}><input type="checkbox" checked={picked.includes(r.file)} disabled={r.sealed} onChange={e => setPicked(p => e.target.checked ? [...p, r.file] : p.filter(x => x !== r.file))} /></td>
                <td>{r.name}{r.sealed && <span className="badge fail">봉인</span>}{r.stale && <span className="badge warn" title={r.sealed_inclusive ? '옛 계산 방식 — 평가 구간에 봉인 구간이 섞여 있어 지금 결과와 비교할 수 없습니다. 순위·다중비교 보정에서 뺍니다.' : '옛 계산 방식(엔진 판이 다름) — 지금 결과와 직접 비교하지 마세요. 다중비교 보정에서 뺍니다.'}>옛 방식</span>}{r.lag_unconfirmed?.length > 0 && <span className="badge warn" title={r.lag_unconfirmed.join(', ')}>lag 미확인 {r.lag_unconfirmed.length}</span>}</td>
                <td>{r.model}</td><td>{r.target_variable}</td><td className="num">{r.k}</td><td className="num">{r.n_features}</td>
                <td><span className={'badge ' + v.cls}>{v.text}</span></td>
                <td className="num"><strong>{pct(r.level_mape, 2)}</strong></td><td className="num">{pct(r.level_mape_naive, 2)}</td><td className="num">{pct(r.level_mape_zero, 2)}</td>
                <td className="num"><span className={r.skill_vs_naive == null ? '' : r.skill_vs_naive > 0 ? 'pos' : 'neg'} title="모델 오차 ÷ '지난 변화 그대로' 오차 (MAPE). 1 미만이면 모델이 더 정확">{r.skill_vs_naive == null ? '–' : '×' + (1 - r.skill_vs_naive).toFixed(2)}</span></td>
                <td className="num">{pct(r.dir_acc)}</td><td className="num">{num(r.dm_naive_p)}</td><td className="num" title={r.family_size ? `같은 문제를 푼 결과 ${r.family_size}개로 보정` : ''}>{num(r.p_holm)}{r.family_size ? <span className="hint"> /{r.family_size}</span> : null}</td><td className="num">{r.n_predictions}</td>
                <td>{String(r.mtime).slice(5, 16).replace('T', ' ')}</td>
              </tr>) : (
              <tr key={r.file} className={sel === r.file ? 'sel' : ''} onClick={() => setSel(r.file)}>
                <td>{r.name}{r.sealed && <span className="badge fail">봉인</span>}{r.stale && <span className="badge warn" title={r.sealed_inclusive ? '옛 계산 방식 — 평가 구간에 봉인 구간이 섞여 있어 지금 결과와 비교할 수 없습니다. 순위·다중비교 보정에서 뺍니다.' : '옛 계산 방식(엔진 판이 다름) — 지금 결과와 직접 비교하지 마세요. 다중비교 보정에서 뺍니다.'}>옛 방식</span>}{r.lag_unconfirmed?.length > 0 && <span className="badge warn" title={r.lag_unconfirmed.join(', ')}>lag 미확인 {r.lag_unconfirmed.length}</span>}</td>
                <td>{r.model}</td><td>{r.frequency}</td><td className="num">{r.k}</td><td className="num">{r.n_features}</td>
                <td><span className={'badge ' + v.cls}>{v.text}</span></td>
                <td className="num">{pct(r.dir_acc)}</td><td className="num">{num(r.dm_p)}</td>
                <td className="num"><strong>{num(r.sharpe)}</strong></td><td className="num">{pct(r.cagr)}</td><td className="num">{pct(r.bh_cagr)}</td><td className="num">{pct(r.mdd)}</td>
                <td className="num">{num(r.p_excess)}</td><td className="num">{num(r.deflated_p)}</td><td className="num">{r.n_trades}</td><td className="num">{r.trials}</td>
                <td>{String(r.mtime).slice(5, 16).replace('T', ' ')}</td>
              </tr>) })}
          </tbody>
        </table>
        {shown.length === 0 && <div className="empty">결과가 없습니다.</div>}
      </div>
      {sel && <Detail stem={sel} onDeleted={() => { setSel(null); refresh() }} />}
    </div>
  )
}

function Detail({ stem, onDeleted }: { stem: string; onDeleted: () => void }) {
  const [r, setR] = useState<any>(null)
  const [s, setS] = useState<any>(null)
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)

  useEffect(() => { setR(null); setS(null); api.result(stem).then(setR); api.series(stem).then(setS).catch(() => setS(null)) }, [stem])
  useEffect(() => {
    if (!s || !chartRef.current || s.mode === 'forecast') return
    chart.current?.remove()
    const c = createChart(chartRef.current, { height: 300, layout: { textColor: '#1c2027', background: { color: '#fff' }, fontFamily: 'Pretendard Variable, sans-serif' }, grid: { vertLines: { color: '#eceff3' }, horzLines: { color: '#eceff3' } }, rightPriceScale: { borderColor: '#d9dde4' }, timeScale: { borderColor: '#d9dde4', minBarSpacing: 0.02 } })
    const eq = c.addSeries(LineSeries, { color: '#23466e', lineWidth: 2, title: '전략' })
    const bh = c.addSeries(LineSeries, { color: '#9aa3b2', lineWidth: 1, title: 'B&H' })
    const pos = c.addSeries(HistogramSeries, { color: 'rgba(15,110,99,.25)', priceScaleId: 'pos', title: '포지션' })
    c.priceScale('pos').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } })
    eq.setData(s.date.map((d: string, i: number) => ({ time: d, value: s.equity[i] })))
    bh.setData(s.date.map((d: string, i: number) => ({ time: d, value: s.benchmark[i] })))
    pos.setData(s.date.map((d: string, i: number) => ({ time: d, value: s.position[i] })))
    c.timeScale().fitContent()
    chart.current = c
    const ro = new ResizeObserver(() => { c.applyOptions({ width: chartRef.current!.clientWidth }); c.timeScale().fitContent() })
    ro.observe(chartRef.current)
    return () => { ro.disconnect(); c.remove(); chart.current = null }
  }, [s])

  if (!r) return <div className="panel empty">불러오는 중…</div>
  const pm = r.prediction_metrics ?? {}, pf = r.performance ?? {}, c = r.config
  const isF = c.mode === 'forecast'
  const v0 = verdict({ ...pm, mode: c.mode, dm_p: pm.dm_vs_zero_p, dm_naive_p: pm.dm_vs_naive_p, p_excess: pf.bootstrap_p_excess, deflated_p: pf.deflated_sharpe_p, degenerate: r.degenerate_model, leak_warning: r.leak_warning })
  const v = r.sealed_evaluation ? { ...v0, text: '봉인 구간 결과 — ' + v0.text.replace('(개발 구간)', '') } : v0
  const imp: [string, number][] = r.feature_importance ? r.feature_names.map((n: string, i: number) => [n, r.feature_importance[i]]).sort((a: any, b: any) => b[1] - a[1]).slice(0, 15) : []
  const maxImp = imp.length ? Math.max(...imp.map(x => x[1]), 1e-9) : 1
  return (
    <div className="split">
      <div>
        <div className="panel">
          <div className={'verdict ' + (v.cls === 'muted' ? 'warn' : v.cls)}>{v.text} — 사전 기준: {isF ? "'변화 없음'과 '지난 변화 그대로'(naive) 두 기준보다 오차가 작고, 그 차이가 우연이 아닐 것(Diebold-Mariano 검정 p<0.05, 둘 다)" : <>DM p&lt;0.05(0예측 대비) · 초과수익 부트스트랩 p&lt;0.05 · Deflated Sharpe p&lt;0.05 (시도 {pf.n_trials_so_far}회 반영)</>}</div>
          <h3>{c.name} <span className="hint">{r.model.name} · {isF ? `대상 ${c.target_variable}` : ''} · {r.evaluation_period?.join(' ~ ')} · {r.data_version} · {r.device} · {r.seconds}s</span></h3>
          {!isF && <><div ref={chartRef} />
          <p className="hint">선: 전략 누적수익(진함) vs 같은 구간 Buy&amp;Hold(연함). 아래 막대: 포지션(1=롱). 체결은 다음 주기 시가, 비용 포함.</p></>}
          {isF && <p className="hint">예측 연구 모드: 체결·비용·자산곡선 없음. 아래 "실제 vs 예측"에서 naive(회색)와 비교하세요.</p>}
          <FormulaGuide keys={['metrics', 'dm', 'target', 'decision']} title="성적표의 식 읽는 법 — 총수익·연복리·샤프·최대낙폭·DM 을 작은 예로 직접 풀기" />
        </div>
        {r.sealed_evaluation && <SealedInfo r={r} />}
        {s?.full && <ActualVsPred full={s.full} />}
        {imp.length > 0 && <div className="panel"><h3>변수 중요도 (폴드 평균, 상위 15)</h3>
          {imp.map(([n, x]) => <div key={n} style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 2fr) 3fr 60px', gap: 8, alignItems: 'center', fontSize: 12, marginBottom: 2 }} title={r.feature_descriptions?.[n] ?? ''}><span>{n}{r.feature_descriptions?.[n] ? <span className="hint"> — {r.feature_descriptions[n]}</span> : null}</span><div style={{ background: '#eceff3', height: 10, borderRadius: 2 }}><div style={{ width: (100 * x / maxImp) + '%', background: '#23466e', height: 10, borderRadius: 2 }} /></div><span className="num">{num(x, 4)}</span></div>)}
          {r.degenerate_model && <p className="hint">중요도가 전부 0 — 상수 예측. 파라미터가 학습 표본 수에 비해 큽니다.</p>}
        </div>}
        {!r.sealed_evaluation && !r.ensemble_members && <SealedPanel stem={stem} r={r} />}
      </div>
      <div>
        {isF && pm.level_mape != null && <div className="panel"><h3>예측력 한눈에 — {c.target?.k ?? 1}{({ D: '거래일', W: '주', M: '개월' } as any)[c.frequency] ?? '기간'} 뒤 값</h3>
          {[['모델', pm.level_mape, 'var(--accent)'], ['지난 변화 그대로 (naive)', pm.level_mape_naive, '#8a94a6'], ['변화 없음', pm.level_mape_zero, '#c3c9d4']].filter(x => x[1] != null).map(([lab, v, col]: any) => {
            const mx = Math.max(pm.level_mape ?? 0, pm.level_mape_naive ?? 0, pm.level_mape_zero ?? 0) || 1
            return <div key={lab} style={{ display: 'grid', gridTemplateColumns: '150px 1fr 64px', gap: 8, alignItems: 'center', fontSize: 13, marginBottom: 4 }}><span>{lab}</span><div style={{ background: '#eceff3', height: 12, borderRadius: 2 }}><div style={{ width: `${(v / mx) * 100}%`, height: '100%', background: col, borderRadius: 2 }} /></div><span className="num">{(v * 100).toFixed(2)}%</span></div>
          })}
          {(() => { const sk = pm.skill_mape_vs_naive; if (sk == null) return null; const sig = pm.dm_vs_naive_p < 0.05
            return <p className="hint" style={{ marginTop: 6 }}>오차 배율 ×{(1 - sk).toFixed(2)} — {sk > 0 ? `모델 오차가 '지난 변화 그대로'보다 ${(sk * 100).toFixed(1)}% 작습니다. ${sig && pm.dm_vs_naive < 0 ? '차이는 통계적으로 유의합니다' : '다만 차이가 우연일 수 있습니다'}`
              : `모델 오차가 '지난 변화 그대로'보다 ${(-sk * 100).toFixed(1)}% 큽니다 — 이 모델은 기준을 이기지 못했습니다${sig && pm.dm_vs_naive > 0 ? '(차이 유의)' : ''}`} (판정 p {num(pm.dm_vs_naive_p)} · MAPE 기준 참고 p {num(pm.dm_ape_vs_naive_p)}).</p> })()}
        </div>}
        {r.ensemble_members && <div className="panel"><h3>앙상블 구성 (같은 공통 시점 {r.n_predictions}개에서 다시 채점)</h3>
          <table><thead><tr><th>구성 결과</th><th className="num">예측 오차</th><th className="num">naive 대비 배율</th><th className="num">판정 p</th></tr></thead>
            <tbody>{Object.entries(r.ensemble_members).map(([st, m]: any) => <tr key={st}><td style={{ fontSize: 12 }}>{st}</td><td className="num">{pct(m.level_mape, 2)}</td><td className="num">{m.skill_mape_vs_naive == null ? '' : '×' + (1 - m.skill_mape_vs_naive).toFixed(2)}</td><td className="num">{num(m.dm_vs_naive_p)}</td></tr>)}
              <tr><td><b>앙상블</b></td><td className="num"><b>{pct(pm.level_mape, 2)}</b></td><td className="num"><b>{pm.skill_mape_vs_naive == null ? '' : '×' + (1 - pm.skill_mape_vs_naive).toFixed(2)}</b></td><td className="num">{num(pm.dm_vs_naive_p)}</td></tr></tbody></table>
          <p className="hint">가중치는 데이터로 정하지 않습니다(동일 가중). 성적을 보고 가중치를 고르면 평가 구간을 엿본 것이 됩니다.</p>
        </div>}
        <div className="panel"><h3>{isF ? '세부 지표 (변화율 기준 — 모델이 학습한 눈금)' : '예측 지표'}</h3>
          {r.periods?.first_fold && <dl className="kv periods"><dt>실제 사용 기간</dt><dd>데이터 {r.periods.data?.join(' ~ ')} · 구간 사이 비움 {r.periods.embargo}행 · 검증 {r.periods.validation_rule === 'date' ? '날짜 지정' : r.periods.validation_rule}</dd>
            <dt>{r.n_folds > 1 ? '첫 폴드' : '폴드'}</dt><dd>학습 {r.periods.first_fold.train?.join('~')} ({r.periods.first_fold.n_train}){r.periods.first_fold.val && <> · 검증 {r.periods.first_fold.val.join('~')} ({r.periods.first_fold.n_val})</>} · 테스트 {r.periods.first_fold.test?.join('~')} ({r.periods.first_fold.n_test})</dd>
            {r.n_folds > 1 && <><dt>마지막 폴드</dt><dd>학습 {r.periods.last_fold.train?.join('~')} ({r.periods.last_fold.n_train}){r.periods.last_fold.val && <> · 검증 {r.periods.last_fold.val.join('~')} ({r.periods.last_fold.n_val})</>} · 테스트 {r.periods.last_fold.test?.join('~')}</dd></>}</dl>}
          <dl className="kv"><dt>예측 수</dt><dd>{r.n_predictions} ({r.n_folds} 폴드)</dd><dt>방향정확도</dt><dd>{pct(pm.dir_acc)}{pm.dir_acc_naive != null && <> (naive {pct(pm.dir_acc_naive)})</>}</dd><dt>RMSE</dt><dd>{num(pm.rmse, 5)}{pm.rmse_naive != null && <> · naive {num(pm.rmse_naive, 5)} · 0예측 {num(pm.rmse_zero, 5)}</>}</dd><dt>DM vs 0예측</dt><dd>{num(pm.dm_vs_zero, 2)} (p {num(pm.dm_vs_zero_p)}) {pm.dm_vs_zero != null && (pm.dm_vs_zero < 0 ? '← 0예측보다 오차 작음' : '← 0예측보다 오차 큼')} <span className="hint">이 방법·이 학습 창의 정확도 비교(추정 잡음 포함)</span></dd>
            {pm.cw_vs_zero != null && <><dt>CW vs 0예측</dt><dd>{num(pm.cw_vs_zero, 2)} (한쪽 p {num(pm.cw_vs_zero_p)}) {pm.cw_vs_zero_p < 0.05 ? '← 변수에 예측 정보가 있다는 증거' : '← 예측 정보의 증거 없음'} <span className="hint">Clark–West: 추정 잡음을 뺀 비교</span></dd></>}
            {pm.dm_vs_naive != null && <><dt>DM vs naive</dt><dd>{num(pm.dm_vs_naive, 2)} (p {num(pm.dm_vs_naive_p)}) {pm.dm_vs_naive < 0 ? '← naive 보다 나음' : '← naive 보다 나쁨'}</dd></>}
            {pm.level_rmse != null && <><dt>수준값 오차</dt><dd>RMSE {num(pm.level_rmse, 3)} · MAPE {pct(pm.level_mape, 2)}</dd></>}
            {pm.auc != null && <><dt>AUC</dt><dd>{num(pm.auc)}</dd></>}</dl></div>
        {!isF && <div className="panel"><h3>매매 성과 (평가 구간)</h3>
          <dl className="kv"><dt>샤프</dt><dd>{num(pf.sharpe)}</dd><dt>소르티노</dt><dd>{num(pf.sortino)}</dd><dt>CAGR</dt><dd>{pct(pf.cagr)} (B&amp;H {pct(pf.benchmark_cagr)})</dd><dt>MDD</dt><dd>{pct(pf.mdd)} (B&amp;H {pct(pf.benchmark_mdd)})</dd><dt>승률</dt><dd>{pct(pf.win_rate)}</dd><dt>전환 횟수</dt><dd>{pf.n_trades} · 노출 {pct(pf.exposure, 0)} · 비용 합계 {pct(pf.total_cost, 2)}</dd><dt>초과수익 p (블록 부트스트랩)</dt><dd>{num(pf.bootstrap_p_excess)}</dd><dt>Deflated Sharpe p</dt><dd>{num(pf.deflated_sharpe_p)} (시도 {pf.n_trials_so_far}회)</dd></dl></div>}
        <div className="panel"><h3>설정</h3>
          <dl className="kv"><dt>변수</dt><dd>{(c.variable_ids ?? []).join(', ') || '(없음)'}{c.use_technical ? ' + 기술지표' : ''}</dd>
            {Object.keys(c.variable_transforms ?? {}).length > 0 && <><dt>변환</dt><dd>{Object.entries(c.variable_transforms).map(([k, v]) => `${k}:${v}`).join(', ')}</dd></>}
            {Object.keys(c.lag_override ?? {}).length > 0 && <><dt>lag 재정의</dt><dd>{Object.entries(c.lag_override).map(([k, v]) => `${k}=${v}`).join(', ')}</dd></>}
            {r.lag_unconfirmed?.length > 0 && <><dt>lag 미확인</dt><dd>{r.lag_unconfirmed.join(', ')}</dd></>}
            <dt>모델 파라미터</dt><dd>{JSON.stringify(r.model.params)}</dd>{!isF && <><dt>비용</dt><dd>수수료 {c.trade.fee_rate}, 슬리피지 {c.trade.slippage_bp}bp</dd></>}<dt>검증</dt><dd>학습 {c.validation.train_len} / 재학습 {c.validation.test_len} / 봉인 {c.validation.sealed_from ?? '없음'}</dd><dt>코드</dt><dd>{r.code_version} · 시드 {c.seed} · 해시 {r.config_hash}</dd></dl>
          <details><summary className="hint">설정 JSON</summary><pre className="log">{JSON.stringify(c, null, 1)}</pre></details>
          <div style={{ marginTop: 10 }}><button onClick={async () => { if (confirm('이 결과 파일을 지웁니다. 실험 카운터는 줄지 않습니다.')) { await api.remove(stem); onDeleted() } }}>결과 삭제</button></div>
        </div>
      </div>
    </div>
  )
}


function ActualVsPred({ full }: { full: any }) {
  const [mode, setMode] = useState<'price' | 'return'>('price')
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const c = createChart(ref.current, { height: 280, layout: { textColor: '#1c2027', background: { color: '#fff' }, fontFamily: 'Pretendard Variable, sans-serif' }, grid: { vertLines: { color: '#eceff3' }, horzLines: { color: '#eceff3' } }, rightPriceScale: { borderColor: '#d9dde4' }, timeScale: { borderColor: '#d9dde4', minBarSpacing: 0.02 } })
    const pts = (arr: (number | null)[]) => full.date.map((d: string, i: number) => arr[i] == null ? null : { time: d, value: arr[i] }).filter(Boolean)
    const isF = !!full.target_variable
    if (mode === 'price') {
      const a = c.addSeries(LineSeries, { color: '#1c2027', lineWidth: 1, title: isF ? `실제(${full.target_variable})` : '실제 종가' })
      a.setData(pts(isF ? full.level : full.close))
      if (!full.is_rule) {
        if (isF && full.naive_future_level) { const nv = c.addSeries(LineSeries, { color: '#9aa3b2', lineWidth: 1, title: 'naive(지속)' }); nv.setData(pts(full.naive_future_level)) }
        const fitL = isF ? full.fit_future_level : full.fit_exit_price
        if (fitL) { const ft = c.addSeries(LineSeries, { color: '#0f6e63', lineWidth: 1, title: '학습 적합(in-sample)' }); ft.setData(pts(fitL)) }
        const pr = c.addSeries(LineSeries, { color: '#b3261e', lineWidth: 1, title: isF ? `예측 ${full.k}기간 뒤 값` : `테스트 예측(k=${full.k})` })
        pr.setData(pts(isF ? full.pred_future_level : full.pred_exit_price))
      }
    } else {
      const a = c.addSeries(LineSeries, { color: '#1c2027', lineWidth: 1, title: isF ? '실제 변화율' : '실제 수익률' })
      a.setData(pts(full.actual_return))
      if (!full.is_rule) {
        if (isF && full.naive_return) { const nv = c.addSeries(LineSeries, { color: '#9aa3b2', lineWidth: 1, title: 'naive' }); nv.setData(pts(full.naive_return)) }
        if (full.fit_return) { const ft = c.addSeries(LineSeries, { color: '#0f6e63', lineWidth: 1, title: '학습 적합' }); ft.setData(pts(full.fit_return)) }
        const pr = c.addSeries(LineSeries, { color: '#b3261e', lineWidth: 1, title: isF ? '예측 변화율' : '테스트 예측' })
        pr.setData(pts(full.pred_return))
      }
      const zero = c.addSeries(LineSeries, { color: '#9aa3b2', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false })
      zero.setData(full.date.map((d: string) => ({ time: d, value: 0 })))
    }
    // 구간 표시: 첫 예측(테스트 시작), 봉인 시작
    const base = c.addSeries(LineSeries, { color: 'transparent', priceLineVisible: false, lastValueVisible: false })
    base.setData(pts(mode === 'price' ? (full.target_variable ? full.level : full.close) : full.actual_return))
    const marks: any[] = []
    if (full.first_prediction) marks.push({ time: full.first_prediction, position: 'aboveBar', color: '#23466e', shape: 'arrowDown', text: full.mode === 'split' ? '테스트 시작 (왼쪽 = 학습 구간)' : '테스트(예측) 시작 — 왼쪽은 학습 워밍업' })
    const sealedDay = full.sealed_from ? full.date.find((d: string) => d >= full.sealed_from) : null
    if (sealedDay) marks.push({ time: sealedDay, position: 'aboveBar', color: '#b3261e', shape: 'arrowDown', text: '봉인 구간 시작' })
    if (marks.length) createSeriesMarkers(base, marks)
    c.timeScale().fitContent()
    const ro = new ResizeObserver(() => { c.applyOptions({ width: ref.current!.clientWidth }); c.timeScale().fitContent() })
    ro.observe(ref.current)
    return () => { ro.disconnect(); c.remove() }
  }, [full, mode])
  return (
    <div className="panel">
      <div className="toolbar"><h3 style={{ margin: 0 }}>실제 vs 예측</h3>
        <button className={'chip' + (mode === 'price' ? ' on' : '')} onClick={() => setMode('price')}>{full.target_variable ? '수준값' : '가격'}</button>
        <button className={'chip' + (mode === 'return' ? ' on' : '')} onClick={() => setMode('return')}>{full.target_variable ? '변화율' : '수익률'}</button>
      </div>
      <div ref={ref} />
      <p className="hint">{full.target_variable ? `예측 연구: 검정 = 실제, 빨강 = 모델이 t 에서 내다본 ${full.k}기간 뒤 값(t+${full.k} 위치에 그림), 회색 = naive(직전 ${full.k}기간 변화율 유지)${full.fit_return ? ', 청록 = 학습 적합(in-sample, 성과에 미사용)' : ''}. 빨강이 회색보다 검정에 가까워야 의미가 있습니다.` : full.is_rule ? '규칙 모델은 예측값이 없습니다(포지션만 냄).' : full.mode === 'split' ? `1회 분할: ${full.split_date} 이전은 학습(청록 = 학습 적합값, 모델이 이미 본 데이터라 항상 좋아 보입니다 — 성과·검정에는 쓰지 않음), 이후는 테스트(빨강).` : `예측은 모두 테스트 구간 값입니다 — 워크포워드는 모델이 보지 않은 날만 예측합니다. 가격 모드의 "예측 청산가" = 진입가(t+1 시가) × (1 + 예측수익률). 실제 청산가는 t+1+k 시가.`} 봉인 구간은 봉인 평가(아래 '봉인 평가' 칸 또는 명령줄 --sealed)를 하기 전까지 예측이 없습니다.</p>
    </div>
  )
}
