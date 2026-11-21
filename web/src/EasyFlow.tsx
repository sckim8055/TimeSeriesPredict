/* 쉽게 보기 — 수학을 몰라도 따라갈 수 있게, 모델 계산을 '실제 데이터 하루'로 처음부터 끝까지 보여 준다.
 *  숫자는 모두 서버 /api/explain(engine/explain.py)이 실험과 같은 코드로 폴드 하나를 실제로 돌려 꺼낸 값이다(예시 숫자 없음).
 *  각 단계: 쉬운 설명 → 정확한 식 → 실제 값(표는 생략 없이 전부, CSV 로 내려받기 가능) → 검증(손 계산 = 모델 출력).
 */
import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

type F = any

export function fmt(v: any): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? '예' : '아니오'
  if (typeof v !== 'number') return String(v)
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (Number.isInteger(v) && a < 1e15) return v.toLocaleString()
  if (a >= 1e4) return v.toLocaleString(undefined, { maximumFractionDigits: 2 })
  if (a < 1e-4) return v.toExponential(4)
  return String(Number(v.toPrecision(6)))
}

function toCsv(t: { columns: string[]; rows: any[][] }) {
  const q = (x: any) => { const s = x === null || x === undefined ? '' : String(x); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s }
  return [t.columns.map(q).join(','), ...t.rows.map(r => r.map(q).join(','))].join('\n')
}
function download(name: string, text: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob(['﻿' + text], { type: 'text/csv;charset=utf-8' }))
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

// '수식 읽는 법'(docs/guide/35_수식_읽는_법.md) — 절마다 <!--f:키--> 표시. 한 번만 불러와 단계마다 해당 절을 펼쳐 보여 준다(문서 하나가 원본).
let GUIDE: Promise<Record<string, string>> | null = null
function loadGuide() {
  GUIDE ??= fetch('/api/guide/' + encodeURIComponent('35_수식_읽는_법')).then(r => r.json()).then((d: any) => {
    const out: Record<string, string> = {}
    String(d.markdown ?? '').split(/<!--f:([a-z_]+)-->/).forEach((seg, i, arr) => { if (i % 2 === 1) out[seg] = arr[i + 1] ?? '' })
    return out
  }).catch(() => ({}))
  return GUIDE
}
// 계산 따라가기의 단계 → 해당 '수식 읽는 법' 절
function guideKeys(model: string, step: string, cls: boolean): string[] {
  const series = model === 'arima' || model === 'sarima', rnn = model === 'lstm' || model === 'gru', biva = model === 'biva'
  const m: Record<string, string[]> = {
    fold: ['folds'], target: ['target'], decide: ['decision'], compound: ['compound'],
    scale: ['standardize'], window: biva ? ['minmax'] : rnn ? ['standardize'] : [], scale_params: biva ? ['minmax'] : ['standardize'],
    weighted_sum: cls ? ['ridge_sum', 'logistic'] : ['ridge_sum'],
    fit: model === 'ridge' ? (cls ? ['ridge_fit', 'logistic'] : ['ridge_fit']) : model === 'lightgbm_model' ? ['lgbm'] : model === 'hmm' ? ['hmm_fit'] : series ? (model === 'sarima' ? ['arima', 'sarima'] : ['arima'])
      : model === 'vecm' ? ['vecm'] : (rnn || biva) ? ['nn_train'] : [],
    trees: ['lgbm'], filter: model === 'hmm' ? ['hmm_filter'] : ['kalman'], states: ['hmm_fit'], ahead: ['hmm_filter'], roots: ['roots'], recursion: model === 'vecm' ? ['vecm'] : model === 'sarima' ? ['sarima', 'kalman'] : ['arima', 'kalman'],
    cells: [model === 'gru' ? 'gru_cell' : 'lstm_cell'], head: ['rnn_head'],
    brits: ['brits'], decomp: ['moving_avg'], trend: ['conv1d'], seasonal: ['conv1d'], vae_enc: ['vae', 'lstm_cell'], vae_z: ['vae'], vae_dec: ['vae'], combine: ['biva_combine'], noise: ['noise'],
    rule: ({ ma_cross: ['rule_ma'], momentum: ['rule_momentum'], rsi_rule: ['rule_rsi'], sox_gap_fade: ['rule_quantile'], shannon: ['rule_shannon'] } as Record<string, string[]>)[model] ?? [],
  }
  return m[step] ?? []
}
function md(text: string) { return DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: false }) as string) }
export function FormulaGuide({ keys, title }: { keys: string[]; title?: string }) {
  const [g, setG] = useState<Record<string, string> | null>(null)
  useEffect(() => { loadGuide().then(setG) }, [])
  const parts = keys.filter(k => g?.[k])
  if (!keys.length || (g && !parts.length)) return null
  return <details className="fguide"><summary>{title ?? '이 식 읽는 법 — 기호 · 문장으로 읽기 · 작은 예로 직접 풀기'}</summary>
    {!g ? <p className="muted">불러오는 중…</p> : parts.map(k => <div key={k} className="fguide-body" dangerouslySetInnerHTML={{ __html: md(g[k]) }} />)}</details>
}

function Table({ t, name }: { t: { columns: string[]; rows: any[][] }; name: string }) {
  return <div className="ex-table">
    <div className="ex-table-bar"><span className="muted">{t.rows.length.toLocaleString()}행 × {t.columns.length}열 · 생략 없음 · 숫자에 마우스를 올리면 전체 자릿수</span>
      <button type="button" className="link" onClick={() => download(name + '.csv', toCsv(t))}>CSV 로 내려받기</button></div>
    <div className="ex-scroll"><table><thead><tr>{t.columns.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
      <tbody>{t.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === 'number' ? 'num' : ''} title={typeof v === 'number' ? String(v) : undefined}>{fmt(v)}</td>)}</tr>)}</tbody></table></div>
  </div>
}

export function ExplainView({ config, auto = true, slow, epochKey }: { config: any; auto?: boolean; slow?: string; epochKey?: string }) {
  const [d, setD] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [date, setDate] = useState('')
  const [ep, setEp] = useState<number | ''>(epochKey === 'train_epochs' ? 20 : '')  // BIVA 만 계산 따라가기 기본 20(원본 500 은 오래 걸림). LSTM·GRU 는 비우면 실험 설정 그대로
  const key = JSON.stringify(config)
  const seq = useRef(0)
  const run = (dt?: string) => {
    const my = ++seq.current
    setBusy(true); setErr(null)
    api.explain(config, dt || undefined, epochKey && ep !== '' ? Number(ep) : undefined).then(r => { if (my === seq.current) { setD(r); setDate(r.decision_date) } })
      .catch(e => { if (my === seq.current) setErr(/not found|404|method not allowed|405/i.test(String(e.message)) ? '서버가 예전 코드로 돌고 있어 계산 따라가기를 못 합니다 — run_server.bat 창을 닫고 다시 실행한 뒤 새로고침하세요.' : e.message) }).finally(() => { if (my === seq.current) setBusy(false) })
  }
  useEffect(() => { if (!auto) return; const h = setTimeout(() => run(), 800); return () => clearTimeout(h) }, [key])  // eslint-disable-line react-hooks/exhaustive-deps

  return <div className="ex">
    <div className="ex-head">
      <b>실제 데이터로 계산 따라가기</b>
      <span className="muted">지금 설정으로 실험과 같은 코드를 폴드 하나에 실제로 돌린 값입니다(예시 숫자 아님).{slow ? ` ${slow}` : ''}</span>
      <label>결정일 <input value={date} placeholder="YYYY-MM-DD" onChange={e => setDate(e.target.value)} style={{ width: 120 }} /></label>
      {d?.available_dates && <select value={d.available_dates.includes(date) ? date : ''} onChange={e => { setDate(e.target.value); run(e.target.value) }}>
        <option value="">이 폴드의 다른 날</option>{d.available_dates.map((x: string) => <option key={x} value={x}>{x}</option>)}</select>}
      {epochKey && <label title={`실험 설정의 ${epochKey} 와 별개로, 계산 따라가기에서만 쓰는 학습 반복 수. 실험과 같은 숫자를 보려면 실험 설정값(예: 500)을 넣으세요(오래 걸림).`}>학습 반복(epoch, 비우면 실험 설정) <input type="number" min={1} max={5000} value={ep} onChange={e => setEp(e.target.value === '' ? '' : +e.target.value)} style={{ width: 70 }} /> <span className="muted">실험 설정: {String(config?.model_params?.[epochKey] ?? '기본값')}</span></label>}
      <button type="button" onClick={() => run(date)} disabled={busy}>{busy ? '계산 중…(학습 모델은 수십 초~수 분)' : d ? '이 날짜로 다시 계산' : '계산하기'}</button>
    </div>
    {err && <div className="verdict warn">{err}</div>}
    {d && <>
      <FormulaGuide keys={['symbols']} title="먼저: 수식 기호 읽는 법 (Σ · argmin · 아래 작은 글자 · σ · tanh …)" />
      <div className={'verdict ' + (d.ok ? 'pass' : 'fail')}>
        {d.ok ? '검증 통과' : '검증 불일치'} — {d.checks.map((c: any, i: number) => <span key={i} className="ex-check" title={c.note ?? ''}>{c.info ? 'ℹ' : c.ok ? '✓' : '✗'} {c.what} (최대 차이 {fmt(c.max_abs_diff)}){c.note ? ` — ${c.note}` : ''}</span>)}
      </div>
      {d.notes?.length > 0 && <div className="verdict warn">{d.notes.join(' / ')}</div>}
      <ol className="ex-steps">
        {d.steps.map((s: any) => <li key={s.id} className="ex-step">
          <h5>{s.title}</h5>
          <p>{s.plain}</p>
          <pre className="ex-formula">{s.formula}</pre>
          <FormulaGuide keys={guideKeys(d.model, s.id, ['direction', 'event'].includes(d.target?.kind))} />
          {s.values && <dl className="ex-kv">{Object.entries(s.values).map(([k, v]) => <div key={k}><dt>{k}</dt><dd title={typeof v === 'number' ? String(v) : undefined}>{fmt(v)}</dd></div>)}</dl>}
          {s.table && <Table t={s.table} name={`${d.model}_${d.decision_date}_${s.id}`} />}
          {s.table2 && <><p className="muted">채운 칸만 모은 표</p><Table t={s.table2} name={`${d.model}_${d.decision_date}_${s.id}_2`} /></>}
        </li>)}
        {d.fold_predictions && <li className="ex-step"><h5>이 폴드의 테스트 구간 예측 전부</h5>
          <p>같은 비율로 테스트 구간 {d.fold_predictions.rows.length}개 날짜를 예측한 값과 실제 정답입니다. 다음 폴드에서는 비율을 새로 정합니다.</p>
          <Table t={d.fold_predictions} name={`${d.model}_fold${d.fold?.index}_predictions`} /></li>}
      </ol>
    </>}
  </div>
}

export function RidgeEasy({ f, config, manual }: { f: F; config: any; manual?: boolean }) {
  const k = f.input?.n_features ?? '?'
  const fit = f.input?.fit_plan?.n_fit ?? f.input?.train_len
  const unit = f.freq === 'M' ? '개월' : f.freq === 'W' ? '주' : '일'
  const step = f.input?.test_len
  const thr = f.input?.threshold
  const cls = ['direction', 'event'].includes(f.input?.target?.kind)
  return <div className="easy">
    <p className="easy-lead"><b>Ridge 는 길이 하나뿐인 모델입니다.</b> 재료(입력 열) {k}개에 각각 정해진 <b>비율</b>을 곱해 더한 값이 예측입니다. 갈림길이나 반복은 없고,
      사람이 고르는 값은 비율을 얼마나 누를지(<code>alpha</code>) 하나입니다. 비율 {k}개와 기본값 1개는 데이터가 정합니다. 계산은 아래 두 갈래이고,
      맨 아래 <b>실제 데이터로 계산 따라가기</b>에서 같은 계산을 실제 숫자로 끝까지 보여 줍니다.</p>

    <div className="easy-lane">
      <div className="easy-lane-t">② 맞히기 — 테스트 구간의 결정일마다 장 마감 후 (①에서 정한 비율을 그대로 씀)</div>
      <div className="easy-steps">
        <div className="easy-card data"><div className="easy-n">1</div><h5>그날의 재료 {k}개</h5>
          <p>장이 끝난 뒤 그날까지 알려진 값으로 만든 숫자들입니다. 기술지표는 ETF 의 가격·거래량으로 계산하고, 고른 변수는 발표시차가 지난 값만 씁니다. 내일 일어날 일은 하나도 들어 있지 않습니다.</p>
          <p className="muted">실제 값: 아래 계산 따라가기 4단계</p></div>
        <div className="easy-arrow">→</div>
        <div className="easy-card"><div className="easy-n">2</div><h5>눈금 맞추기 (표준화)</h5>
          <p>재료마다 단위가 다릅니다(수익률은 0.01 = 1%, RSI 는 0~100 점수). 그대로 더하면 큰 숫자가 이겨 버리므로 <b>"평소보다 몇 칸 큰가"</b>로 바꿉니다.</p>
          <p className="rule">칸 수 = (그날 값 − 평소 값) ÷ 평소 흔들림</p>
          <p>평소 값 = 학습 표의 평균, 평소 흔들림 = 학습 표의 표준편차. 테스트 구간·미래 값은 쓰지 않습니다. 예: 평소보다 흔들림만큼 크면 +1칸, 평소와 같으면 0칸.</p>
          <p className="muted">실제 값: 5단계</p></div>
        <div className="easy-arrow">→</div>
        <div className="easy-card"><div className="easy-n">3</div><h5>비율대로 더하기 (가중합)</h5>
          <p>재료마다 학습으로 정해진 <b>비율</b>이 있습니다. 비율 = 그 재료가 평소보다 1칸 클 때 예측이 얼마나 움직이는가. 비율이 +면 그 재료가 클수록 예측이 오르고, −면 내려갑니다.</p>
          <p className="rule">예측 = 기본값 + (비율₁ × 칸 수₁) + … + (비율{k} × 칸 수{k})</p>
          <p>기본값 = 재료가 모두 '평소'(0칸)일 때의 예측. 각 괄호 하나가 그 재료의 <b>기여</b>입니다.</p>
          {cls && <p>오를지/내릴지를 맞히는 설정이면 이 합계를 확률로 바꿉니다: 확률 = 1 ÷ (1 + e^(−합계)). 합계 0 → 50%, 합계가 클수록 100% 에, 작을수록 0% 에 가까워집니다.</p>}
          <p className="muted">실제 값: 6단계</p></div>
        <div className="easy-arrow">→</div>
        <div className="easy-card out"><div className="easy-n">4</div><h5>결정</h5>
          <p>{cls ? <>확률 &gt; 50% + 진입 임계치{thr != null ? `(${thr})` : ''} 이면 산다.</> : <>예상 수익률 &gt; 진입 임계치{thr != null ? `(${thr})` : ''} 이면 산다.</>} 아니면 현금.</p>
          <p>체결은 항상 <b>다음 날 시가</b>(그날 종가에는 이미 장이 끝나 살 수 없음). 이미 들고 있으면 그대로 두고, 사거나 팔 때마다 수수료·슬리피지를 냅니다.</p>
          <p className="muted">실제 값: 8단계</p></div>
      </div>
    </div>

    <div className="easy-lane learn">
      <div className="easy-lane-t">① 배우기 — 폴드마다 한 번{step ? `(${step}${unit}마다)` : ''}, 과거 학습 창으로 비율을 다시 정함</div>
      <div className="easy-steps">
        <div className="easy-card data"><h5>과거 {fit}{unit}의 기록 (학습 표)</h5>
          <p>날마다 "그날의 재료 {k}개"와 "그날 결정했다면 실제로 번 수익률(정답)"을 한 줄로 적은 표. 재료나 정답에 빈 칸이 있는 날은 뺍니다.
            학습 창 끝의 몇 날은 비웁니다 — 그 날들의 정답은 테스트 구간 가격으로 계산되어, 넣으면 미래를 엿보게 됩니다.</p>
          <p className="muted">실제 값: 1·3단계</p></div>
        <div className="easy-arrow">→</div>
        <div className="easy-card filter"><h5>틀린 정도가 가장 작은 비율 찾기</h5>
          <p>비율을 하나 정하면 과거 모든 날의 예측을 계산할 수 있습니다. 그중 <b>(예측 − 정답)을 제곱해 다 더한 값</b>이 가장 작아지는 비율을 찾습니다. 제곱하는 이유: 크게 틀린 날을 더 무겁게 보고, +·− 가 서로 지워지지 않게.
            단, <b>비율이 클수록 벌점</b>을 더합니다.</p>
          <p className="rule">점수 = 틀린 정도의 합 + alpha × (비율을 제곱해 더한 값) → 가장 작게</p>
          <p className="muted">회귀는 이 답을 공식 한 번으로 바로 계산합니다(반복 없음). 오를지/내릴지 설정은 같은 생각을 '정답에 낮은 확률을 준 정도'(로그손실)로 바꿔 반복 계산으로 찾습니다. 기본값에는 벌점이 없습니다. 실제 값: 7단계</p></div>
        <div className="easy-arrow">→</div>
        <div className="easy-card"><h5>alpha = 벌점의 세기 (지금 {String(f.params?.alpha ?? '?')})</h5>
          <ul>
            <li><b>작게</b>(0 에 가깝게): 과거에 딱 맞추려고 비율이 커짐 → 과거를 외운 듯, 새로운 날에 잘 틀림(과적합)</li>
            <li><b>크게</b>: 비율이 모두 0 쪽으로 작아짐 → 예측이 늘 기본값(평균) 근처, 무난하지만 둔함</li>
            <li>비율이 정확히 0 이 되지는 않습니다(재료를 빼 버리지는 않음).</li>
          </ul>
          <p className="muted">alpha 는 데이터가 정하지 않고 사람이 정합니다. 주기별 기본: 일간 1 · 주간 10 · 월간 100.</p></div>
        <div className="easy-arrow">→</div>
        <div className="easy-card out"><h5>결과: 비율 {k}개 + 기본값 1개</h5>
          <p>② 맞히기의 3번 '비율대로 더하기'에 그대로 쓰입니다. 다음 폴드에서는 학습 창을 밀어 처음부터 다시 정합니다.</p>
          <p>결과 화면의 '변수 중요도' = 이 비율의 크기(부호 무시). 눈금을 맞춘 뒤라 재료끼리 비교할 수 있습니다.</p></div>
      </div>
    </div>
    <ExplainView config={config} auto={!manual} />
  </div>
}

/* 모델별 '쉽게 보기' 설명 카드. 숫자는 쓰지 않고(실제 숫자는 아래 계산 따라가기), 코드가 하는 일을 순서대로 말로 적는다.
   각 문장은 해당 모델 코드(models/*.py)와 엔진(engine/experiment.py)에 맞춘 것 — 바꾸면 코드도 확인할 것. */
type Card = { h: string; p: string[]; rule?: string; tone?: 'data' | 'filter' | 'out' }
type Lane = { t: string; learn?: boolean; cards: Card[] }
type Easy = { lead: string; lanes: Lane[]; slow?: string; epochKey?: string }
const TBL_IN: Card = { h: '그날의 재료', tone: 'data', p: ['장 마감 후 그날까지 알려진 값만으로 만든 숫자들(기술지표는 ETF 가격·거래량으로 계산, 고른 변수는 발표시차가 지난 값). 내일 일은 들어 있지 않습니다.'] }
const TBL_SC: Card = { h: '눈금 맞추기 (표준화)', p: ['재료마다 단위가 달라 "평소보다 몇 칸 큰가"로 바꿉니다. 평소 값·흔들림은 학습 표로만 계산합니다.'], rule: '칸 수 = (그날 값 − 평균) ÷ 표준편차' }
const DECIDE: Card = { h: '결정', tone: 'out', p: ['예측이 진입 임계치보다 크면(확률이면 50% + 임계치보다 크면) 다음 날 아침 시가에 사서 들고 가고, 아니면 현금. 사고팔 때마다 수수료·슬리피지.'] }
const EASY: Record<string, Easy> = {
  lightgbm_model: {
    lead: 'LightGBM 은 "질문 나무" 수백 그루의 답을 더해 예측합니다. 나무 하나는 "이 재료가 기준보다 작은가?" 같은 질문을 몇 번 이어 해서 그날을 한 칸(잎)에 넣고, 그 칸의 값을 냅니다. 나무 하나하나는 약하지만, 뒤 나무가 앞 나무들이 틀린 만큼을 고치도록 차례로 만들어 모으면 강해집니다(부스팅).',
    lanes: [
      { t: '① 배우기 — 폴드마다 한 번, 학습 표로 나무를 차례로 만듦', learn: true, cards: [
        { h: '학습 표', tone: 'data', p: ['날짜마다 재료와 정답(그날 결정했다면 번 수익률). 빈 칸 있는 날은 빼고, 학습 창 끝 몇 날은 비웁니다. 검증을 쓸 수 있으면(11개 이상) 뒤쪽 일부를 떼어 "언제 멈출지" 보는 데만 씁니다.'] },
        { h: '시작값', p: ['첫 나무 전에는 모든 날을 정답의 평균으로 예측해 둡니다(오를지/내릴지면 오른 비율의 로그오즈).'] },
        { h: '나무 하나 만들기 (거름)', tone: 'filter', p: ['지금까지의 예측이 틀린 만큼을 맞히는 나무를 만듭니다. 날짜의 일부·재료의 일부만 무작위로 뽑아 쓰고(seed 고정), 틀린 정도를 가장 많이 줄이는 질문(재료·기준값)을 골라 둘씩 나눕니다. 잎 수·깊이·잎의 최소 날짜 수를 넘는 분할은 하지 않습니다.'], rule: '새 예측 = 이전 예측 + 학습률 × 이 나무의 답' },
        { h: '멈추기', p: ['정한 나무 수까지 반복합니다. 검증이 있으면 검증 오차가 50그루 연속 나아지지 않을 때 멈추고, 가장 좋았던 그루 수까지만 씁니다.'] },
      ] },
      { t: '② 맞히기 — 결정일마다', cards: [TBL_IN, TBL_SC,
        { h: '나무마다 내려가기', p: ['나무마다 그날의 값으로 질문에 답하며 내려가 잎에 도착하고, 그 잎의 값을 읽습니다. 모든 나무의 잎 값을 더한 것이 예측(오를지/내릴지면 확률로 바꿈)입니다.'], rule: '예측 = 나무₁ 잎 값 + 나무₂ 잎 값 + …' }, DECIDE] },
    ] },
  arima: {
    lead: 'ARIMA 는 다른 재료를 쓰지 않고 "이 가격의 변화율이 과거의 자기 자신을 얼마나 닮는가"만으로 다음을 맞힙니다. 지난 변화율이 이어지는 정도(AR)와 지난 예상 밖 충격이 남는 정도(MA)를 숫자(계수)로 정해 두고, 그 계수로 한 기간씩 앞으로 이어 계산합니다.',
    lanes: [
      { t: '① 배우기 — 폴드마다 한 번', learn: true, cards: [
        { h: '재료: 변화율 하나', tone: 'data', p: ['학습 창의 "어제 종가 대비 오늘 종가 변화율"만 씁니다. 고른 변수·기술지표는 무시됩니다.'], rule: 'r = 오늘 종가 ÷ 어제 종가 − 1' },
        { h: '계수 정하기', tone: 'filter', p: ['이 변화율들이 가장 그럴듯하게 나오는 계수(상수 c, AR φ, MA θ, 충격의 크기 σ²)를 반복 계산으로 찾습니다(최대우도).'], rule: '변화율 = c + Σ φ × 지난 변화율 + 충격 + Σ θ × 지난 충격' },
        { h: '계수 점검 (제약)', p: ['"정상성 제약"을 켜면 AR 계수를 "한 번의 충격이 시간이 지나면 사라지는" 범위에서만, "가역성 제약"을 켜면 MA 계수를 "먼 과거의 오차일수록 덜 남는" 범위에서만 찾습니다(기본: 둘 다 켬 — statsmodels·원 노트북과 같음). 끄면 그 범위 밖의 계수가 나올 수 있고, 계산 따라가기의 "근 크기" 표에서 확인할 수 있습니다. 두 설정으로 각각 실험해 비교할 수 있습니다.'], rule: '모든 근의 크기 |z| > 1 이면 정상·가역' },
      ] },
      { t: '② 맞히기 — 결정일마다 (계수는 그대로)', cards: [
        { h: '지난 충격 알아내기', tone: 'data', p: ['데이터 처음부터 결정일까지의 변화율을 차례로 넣으며 "그때 예측과 실제의 차이(충격)"를 계산합니다(칼만 필터). 결정일 이후는 쓰지 않습니다.'] },
        { h: '한 기간씩 이어 예측', p: ['다음 기간을 예측하고, 그 예측을 다시 지난 값처럼 넣어 그다음을 예측합니다. 미래의 충격은 모르므로 0.'] },
        { h: 'k 기간 수익률로 묶기', p: ['매매 모드는 다음 날 시가에 사므로 첫 기간 예측은 버리고 그다음 k 개를 복리로 묶습니다. 이 모델은 종가 변화율을 예측해 시가 기준 정답과 하루 안 차이가 남습니다(근사).'], rule: '예측 = (1 + r̂₁) × … × (1 + r̂ₖ) − 1' }, DECIDE] },
    ] },
  vecm: {
    lead: 'VECM 은 "오래 보면 함께 가는 값들(예: 집값과 물가)이 잠깐 벌어지면 다시 붙는다"는 성질을 씁니다. 함께 가는 조합(장기 균형식)을 찾고, 지금 그 균형에서 얼마나 벗어났는지 보고, 벗어난 만큼 되돌아오는 변화와 지난 변화의 영향을 더해 다음 값을 계산합니다.',
    lanes: [
      { t: '① 배우기 — 폴드마다 한 번', learn: true, cards: [
        { h: '재료: 원래 값(수준값)', tone: 'data', p: ['대상과 고른 변수의 변환하지 않은 값(발표시차 적용). 학습 창에서 한 번도 안 바뀐 열은 뺍니다. 기술지표는 쓰지 않습니다.'] },
        { h: '장기 균형식 β', tone: 'filter', p: ['여러 값의 조합 중 오래 보면 일정하게 유지되는 조합을 찾습니다(Johansen). 조합의 개수 r 을 자동으로 고를 수 있고, 0 이면 균형식 없이 변화끼리의 관계(차분 VAR)로 대신합니다.'] },
        { h: '되돌림 α · 단기 Γ', p: ['균형에서 벗어난 만큼 다음 변화를 얼마나 반대로 당기는지(α), 지난 p 기간의 변화가 이번 변화에 주는 영향(Γ)을 추정합니다.'] },
      ] },
      { t: '② 맞히기 — 결정일마다', cards: [
        { h: '균형 이탈', p: ['결정일 값으로 장기 균형식을 계산해 지금 얼마나 벗어났는지 봅니다.'], rule: '이탈 e = β′ × 지금 값' },
        { h: '다음 변화', p: ['되돌림 + 지난 변화의 영향 + 상수·계절을 더해 다음 기간의 변화를 만들고, 지금 값에 더합니다. k 기간 동안 반복합니다.'], rule: '다음 값 = 지금 값 + α·e + Γ·지난 변화' },
        { h: '변화율로 바꾸기', p: ['대상(첫 열)의 k 기간 뒤 예측 값이 기준보다 몇 % 다른지로 바꿉니다.'] }, DECIDE] },
    ] },
  lstm: {
    slow: 'LSTM 은 실제로 학습을 돌리므로 "계산하기"를 누르면 수 초~수십 초 걸립니다.', epochKey: 'epochs',
    lead: 'LSTM 은 최근 L 개 시점의 재료를 순서대로 읽으며 "기억"을 갱신하고, 마지막 시점의 기억으로 다음을 맞힙니다. 시점마다 0~1 사이의 문(게이트) 세 개가 "지난 기억 중 얼마를 남길지·새 정보 중 얼마를 쓸지·기억 중 얼마를 내보낼지"를 정합니다. 문을 여닫는 정도와 마지막 계산의 가중치는 학습이 정합니다.',
    lanes: [
      { t: '① 배우기 — 폴드마다 한 번', learn: true, cards: [
        { h: '학습 창들', tone: 'data', p: ['학습 표에서 연속한 L 개 행을 한 묶음(창)으로 만들고, 창의 마지막 날의 정답을 붙입니다. 빈 칸은 앞 값으로 채우고 그래도 남으면 학습 평균. 표준화는 학습 구간 통계로만.'] },
        { h: '가중치 고치기 반복', tone: 'filter', p: ['창 순서를 섞어 몇 개씩 넣고, 예측과 정답의 차이(제곱)가 줄어드는 쪽으로 가중치를 조금씩 고칩니다(Adam, 기울기 크기 자르기 1.0). 학습 중에는 일부 값을 무작위로 끕니다(Dropout).'] },
        { h: '멈추기', p: ['검증 창이 10개 이상이면 검증 오차가 정한 횟수만큼 연속 나아지지 않을 때 멈추고 가장 좋았던 가중치를 씁니다. 아니면 정한 횟수를 다 돌고 마지막 가중치.'] },
      ] },
      { t: '② 맞히기 — 결정일마다', cards: [
        { h: '입력 창', tone: 'data', p: ['결정일까지 최근 L 개 시점 × 재료(첫 테스트일이면 학습 창 끝의 실제 과거를 붙임).'] },
        { h: '시점마다 셀 계산', p: ['첫 시점부터 차례로: 문 세 개(잊기 f·입력 i·출력 o)와 새 정보 g 를 계산 → 기억 c = f × 지난 기억 + i × g → 내보내는 값 h = o × tanh(c).'] },
        { h: '마지막 시점으로 예측', p: ['마지막 시점의 h 만 남겨 가중치를 곱해 더하고, 학습 때 표준화한 정답 단위를 원래 단위로 되돌립니다.'] }, DECIDE] },
    ] },
}
EASY.biva = {
  slow: 'BIVA 는 실제로 학습을 돌립니다. 기본은 계산 따라가기용 20 epoch(일간 18개 변수 기준 약 3~4분, 월간 34개 변수 약 1분) — 실험과 같은 숫자를 보려면 실험 설정값(원본 500)을 넣으세요.', epochKey: 'train_epochs',
  lead: 'BIVA 는 네 단계로 예측합니다. ① 창 안의 빈 칸을 앞뒤로 읽어 채우고(BRITS) ② 각 변수를 "느린 흐름(추세)"과 "빠른 출렁임(주기)"으로 나눈 뒤 ③ 주기는 여러 변수를 몇 개의 공통요인으로 줄여(LSTM-VAE) 예측하고, 추세는 변수를 섞어 바로 예측해 ④ 두 예측을 학습된 비율로 합칩니다. 아래 계산 따라가기에서는 네 단계를 모두 원본 코드의 가중치로 손 계산해 원본 모듈 출력과 맞춰 봅니다.',
  lanes: [
    { t: '① 배우기 — 폴드마다 한 번 (원본 exp_main.train 순서)', learn: true, cards: [
      { h: '학습 창들', tone: 'data', p: ['학습 표에서 연속한 L 개 행을 한 창으로, 창의 마지막 날 정답을 붙입니다. 눈금은 학습 구간 최소=0·최대=1(MinMax, 원본). 아직 시작 전인 변수의 빈 칸은 NaN 그대로 둬 BRITS 가 채웁니다.'] },
      { h: '한 번에 세 가지 손실', tone: 'filter', p: ['창을 batch_size(원본 3)개씩 섞어 넣고(나머지 버림) 예측 오차(MSE) + 빈 칸 채우기 오차(BRITS) + 요인 압축 오차(VAE)의 합이 줄도록 가중치를 고칩니다(Adam). 세 손실에 가중치를 따로 두지 않습니다(원본).'] },
      { h: '멈추기·되돌리기', p: ['epoch 마다 검증 손실(배치 평균)을 보고 정한 횟수 동안 나아지지 않으면 멈추며, 정한 일정대로 학습률을 줄입니다. 끝나면 검증 손실이 가장 낮았던 가중치로 되돌립니다.'] },
    ] },
    { t: '② 맞히기 — 결정일마다', cards: [
      { h: 'Stage 1 빈 칸 채우기', tone: 'filter', p: ['앞→뒤, 뒤→앞 두 방향으로 읽으며 시점마다 "기억으로 추정"과 "같은 시점 다른 변수로 추정"을 반씩 섞어 빈 칸만 채웁니다. 관측된 칸은 그대로. 두 방향 결과의 평균.'] },
      { h: 'Stage 2 추세·주기 나누기', p: ['변수마다 창 안 이동평균 = 추세, 원래 − 추세 = 주기. 창 양끝은 끝값을 복사해 늘립니다.'] },
      { h: 'Stage 3 주기 → 공통요인', p: ['주기 성분을 LSTM 4층으로 줄여 시점마다 평균 μ·흩어짐 σ 를 만들고 z = μ + 무작위 × σ 로 뽑습니다(예측 때도 뽑음, 원본은 σ ≥ 1). [인코더 출력, z] 를 섞어 주기 예측 하나. 추세는 변수를 섞어(Conv1d 3단) 시간 가중 평균으로 추세 예측 하나.'] },
      { h: 'Stage 4 합치기', tone: 'out', p: ['예측 = α × 주기 예측 + (1 − α) × 추세 예측(α 학습). 학습 구간 평균·표준편차로 원래 단위로 되돌린 뒤, 임계치와 비교해 다음 날 시가에 보유/현금.'] },
    ] },
  ] }
EASY.sarima = { ...EASY.arima, lead: 'SARIMA 는 ARIMA 에 "계절"을 더한 것입니다. 지난 변화율·충격뿐 아니라 s 기간 전(예: 월간이면 작년 같은 달)의 변화율·충격도 봅니다. D=1 이면 먼저 s 기간 전과의 차이를 취해 계절 반복을 지운 뒤 맞힙니다. 계산 방법(계수 정하기 → 칼만 필터 → 한 기간씩 이어 예측 → 복리로 묶기)은 ARIMA 와 같습니다.' }
EASY.gru = { ...EASY.lstm, slow: 'GRU 는 실제로 학습을 돌리므로 "계산하기"를 누르면 수 초~수십 초 걸립니다.', epochKey: 'epochs',
  lead: 'GRU 는 LSTM 의 단순한 형태입니다. 최근 L 개 시점을 순서대로 읽으며 기억 h 를 갱신하는데, 문이 두 개입니다: 재설정 r(새 후보를 만들 때 지난 기억을 얼마나 볼지)과 갱신 z(지난 기억과 새 후보를 어떤 비율로 섞을지). 별도의 기억 줄(c)이 없어 가중치가 약 25% 적습니다.',
  lanes: [EASY.lstm.lanes[0], { t: '② 맞히기 — 결정일마다', cards: [EASY.lstm.lanes[1].cards[0],
    { h: '시점마다 셀 계산', p: ['첫 시점부터 차례로: r·z 문과 새 후보 n = tanh(입력 + r × 지난 기억의 영향) → 새 기억 h = (1 − z) × n + z × 지난 기억.'] },
    EASY.lstm.lanes[1].cards[2], DECIDE] }] }
EASY.hmm = {
  lead: 'HMM 시장 상태는 다른 재료를 쓰지 않고 "이 가격의 변화율 뒤에 눈에 안 보이는 상태(예: 잔잔한 상승기 / 출렁이는 하락기)가 있고, 상태마다 변화율의 평균·흩어짐이 다르다"고 보고 다음을 맞힙니다. 상태는 직접 보이지 않아 확률로만 추정합니다.',
  lanes: [
    { t: '① 배우기 — 폴드마다 한 번', learn: true, cards: [
      { h: '재료: 변화율 하나', tone: 'data', p: ['학습 창의 "어제 종가 대비 오늘 종가 변화율"만 씁니다. 고른 변수·기술지표는 무시됩니다.'], rule: 'r = 오늘 종가 ÷ 어제 종가 − 1' },
      { h: '상태 찾기 (Baum–Welch)', tone: 'filter', p: ['상태별 평균·흩어짐과 "상태가 바뀔 확률"을, 이 변화율들이 가장 그럴듯하게 나오도록 반복 계산으로 정합니다. 난수를 쓰지 않아 다시 해도 같습니다.'], rule: '상태 j 의 변화율 ~ 정규(평균 μ_j, 흩어짐 σ_j)' },
    ] },
    { t: '② 맞히기 — 결정일마다 (모수는 그대로)', cards: [
      { h: '지금 어느 상태일까', tone: 'data', p: ['처음부터 결정일까지 변화율을 하루씩 넣으며 "어제 상태 확률 × 바뀔 확률"로 짐작하고, 오늘 변화율로 다시 맞춥니다. 결정일 이후는 쓰지 않습니다.'] },
      { h: '앞날의 상태 확률 → 예측', p: ['앞날 상태는 모르므로 바뀔 확률만 곱해 나가고, 그 확률로 상태 평균을 가중해 변화율을 예측합니다.'], rule: '예측 r̂ = Σ (상태 확률 × 상태 평균)' },
      { h: 'k 기간 수익률로 묶기', p: ['매매 모드는 다음 날 시가에 사므로 첫 기간 예측은 버리고 그다음 k 개를 복리로 묶습니다(ARIMA 와 같은 근사).'], rule: '예측 = (1 + r̂₁) × … × (1 + r̂ₖ) − 1' }, DECIDE] },
  ] }
const RULE_EASY: Record<string, string> = {
  buy_and_hold: '항상 보유: 평가 구간 첫날 사서 끝까지 들고 있습니다. 학습도 판단도 없습니다. 모든 전략이 이겨야 하는 기준선입니다.',
  ma_cross: '이동평균 교차: 결정일까지 최근 짧은 기간 종가 평균(빠른 흐름)이 긴 기간 평균(느린 흐름)보다 위면 오르는 흐름으로 보고 보유, 아니면 현금. 학습은 없습니다.',
  momentum: '모멘텀: 결정일 종가가 정한 기간 전 종가보다 높으면(올랐으면) 그 흐름이 이어진다고 보고 보유, 아니면 현금. 학습은 없습니다.',
  rsi_rule: 'RSI 규칙: 최근 오른 폭·내린 폭의 평균으로 만든 0~100 점수(RSI)가 아래 문턱보다 낮으면(많이 떨어짐) 사고, 위 문턱보다 높으면(많이 오름) 팔고, 그 사이면 어제 상태를 그대로 둡니다. 학습은 없습니다.',
  shannon: '섀넌 재조정: 가격을 보지 않고 ETF 비율(기본 50%)을 목표로 둡니다. 그 사이에는 비율이 가격을 따라 흘러가고(오르면 늘고 내리면 줄고), 정한 주기(기본 달마다)의 마지막 결정일에 다시 목표 비율로 맞춥니다 — 그때만 비용. 학습은 없습니다.',
  sox_gap_fade: 'SOX 급등 회피: 고른 변수(예: 필라델피아 반도체 지수 변화율)가 그 전 일정 기간 값들 중 상위 몇 % 안에 들 만큼 크면 다음 날은 현금, 아니면 보유. 기준은 결정일 당일을 뺀 과거로만 계산합니다. 학습은 없습니다.',
}

export function ModelEasy({ model, config, manual }: { model: string; config: any; manual?: boolean }) {
  if (RULE_EASY[model]) return <div className="easy"><p className="easy-lead"><b>{RULE_EASY[model]}</b> 진입 임계치는 쓰지 않고 규칙이 바로 포지션({model === 'shannon' ? 'ETF 비율' : '1 = 보유, 0 = 현금'})을 냅니다. 성적은 같은 설정의 학습 모델이 평가받는 날짜와 같은 구간으로 매깁니다.</p><ExplainView config={config} auto={!manual} /></div>
  const e = EASY[model]
  if (!e) return <div className="easy"><p className="easy-lead">이 모델의 계산 따라가기는 준비 중입니다.</p></div>
  return <div className="easy">
    <p className="easy-lead">{e.lead} 맨 아래 <b>실제 데이터로 계산 따라가기</b>에서 같은 계산을 실제 숫자로 끝까지 보여 줍니다.</p>
    {e.lanes.map((l, i) => <div key={i} className={'easy-lane' + (l.learn ? ' learn' : '')}><div className="easy-lane-t">{l.t}</div>
      <div className="easy-steps">{l.cards.map((c, j) => <div key={j} style={{ display: 'contents' }}>{j > 0 && <div className="easy-arrow">→</div>}
        <div className={'easy-card ' + (c.tone ?? '')}><h5>{c.h}</h5>{c.p.map((t, q) => <p key={q}>{t}</p>)}{c.rule && <p className="rule">{c.rule}</p>}</div></div>)}</div></div>)}
    <ExplainView config={config} auto={!e.slow && !manual} slow={e.slow} epochKey={e.epochKey} />
  </div>
}
