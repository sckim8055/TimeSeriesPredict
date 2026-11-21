import TransformSelect from './TransformSelect'
import { useMemo, useState, useEffect } from 'react'
import { api, type Meta, type Mode, type Variable, type VariableSet } from './api'
import { ModelFlow, techCount } from './ModelFlow'
import SplitPlan from './SplitPlan'
import { helpOf } from './paramHelp'
import { ValidationDiagram } from './ValidationDiagram'
import { SourceBadge, SOURCE_INFO, KIND_SYMBOL, KIND_LABEL as SRC_KIND_LABEL } from './source'

type Props = { meta: Meta; mode: Mode; onSubmitted: (jobId: string) => void }

const FREQ_LABEL: Record<string, string> = { D: '일간', W: '주간', M: '월간' }
const MODEL_GROUPS: [string, string][] = [['series', '통계 모델 (대상 자체의 과거만)'], ['levels', '통계 모델 (여러 시계열의 수준값)'], ['table', '머신러닝 (변수 표)'], ['window', '딥러닝 (과거 창)'], ['rule', '규칙 (기준선, 매매 전용)']]
const FC_KIND_LABEL: Record<string, string> = { return: '값 예측 — 변화율로 학습 (권장)', logreturn: '값 예측 — 로그 변화율로 학습', direction: '방향 예측 — 오를까 내릴까' }
const UNIT: Record<string, string> = { D: '거래일', W: '주', M: '개월' }
const lagEvidence = (note: string) => (note || '').split(' | ').find(x => x.startsWith('[lag 검증'))?.replace(/^\[lag 검증 [0-9-]+\] /, '') ?? ''
const KIND_LABEL: Record<string, string> = { return: '수익률(회귀)', logreturn: '로그수익률(회귀)', direction: '방향(분류)', event: '임계치 초과(분류)', volatility: '변동성 — 앞으로 얼마나 출렁일까 (k 는 5 이상 권장)' }
const DEFAULT_VALIDATION: Record<string, { train_len: number; test_len: number }> = { D: { train_len: 756, test_len: 21 }, W: { train_len: 156, test_len: 4 }, M: { train_len: 60, test_len: 1 } }
const DEFAULT_SEQ: Record<string, number> = { D: 20, W: 13, M: 12 }
const PERIOD_DAYS: Record<string, number> = { D: 1, W: 7, M: 31, Q: 92, A: 366 }
const iso = (d: Date) => d.toISOString().slice(0, 10)
const addDays = (s: string, n: number) => { const d = new Date(s + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return iso(d) }
const isDate = (s: string) => /^\d{4}-\d{2}-\d{2}$/.test(s)

/** 기간 막대: 데이터 시작 ~ 끝 위에 학습·검증·테스트·봉인 구간을 비율대로 그린다. 날짜를 모르면 그 칸은 비워 둔다. */
function Timeline({ segs, from, to }: { segs: { label: string; a: string; b: string; cls: string }[]; from: string; to: string }) {
  const t = (s: string) => new Date(s + 'T00:00:00Z').getTime()
  if (!isDate(from) || !isDate(to) || t(to) <= t(from)) return null
  const W = t(to) - t(from)
  return <div className="timeline">
    <div className="tl-bar">{segs.filter(g => isDate(g.a) && isDate(g.b) && t(g.b) > t(g.a)).map(g => {
      const l = Math.max(0, (t(g.a) - t(from)) / W * 100), w = Math.min(100 - l, (t(g.b) - t(g.a)) / W * 100)
      return <div key={g.label} className={'tl-seg ' + g.cls} style={{ left: l + '%', width: w + '%' }} title={`${g.label}: ${g.a} ~ ${g.b}`}><span>{g.label}</span></div>
    })}</div>
    <div className="tl-ends"><span>{from}</span><span>{to}</span></div>
  </div>
}

export default function ExperimentForm({ meta, mode, onSubmitted }: Props) {
  const isForecast = mode === 'forecast'
  const [name, setName] = useState('')
  const [ticker, setTicker] = useState(meta.tickers[0].id)
  const [targetVar, setTargetVar] = useState(meta.forecast_targets.find(t => t.id === 'F11')?.id ?? meta.forecast_targets[0]?.id ?? '')
  const [freq, setFreq] = useState<'D' | 'W' | 'M'>('D')
  const [kind, setKind] = useState('return')
  const [k, setK] = useState(1)
  const [threshold, setThreshold] = useState(0)
  const [modelFile, setModelFile] = useState(meta.models.find(m => m.file === 'ridge')?.file ?? meta.models[0].file)
  const [params, setParams] = useState<Record<string, any>>({})
  const [useTechnical, setUseTechnical] = useState(true)
  const [techSel, setTechSel] = useState<string[] | null>(null)  // 기술지표 개별 선택. null = 전부(기본)
  const [selected, setSelected] = useState<string[]>([])
  const [transforms, setTransforms] = useState<Record<string, string>>({})
  const [lagOverride, setLagOverride] = useState<Record<string, number>>({})
  const [fill, setFill] = useState('ffill')
  const [slippage, setSlippage] = useState(5)
  const [fee, setFee] = useState(0.00015)
  const [tradeThreshold, setTradeThreshold] = useState(0)
  const [sizing, setSizing] = useState<'threshold' | 'kelly'>('threshold')   // 투자 비율 정하기(2026-11-15)
  const [kellyFrac, setKellyFrac] = useState(0.5)
  const [trainLen, setTrainLen] = useState(mode === 'forecast' ? 60 : 756)
  const [testLen, setTestLen] = useState(mode === 'forecast' ? 1 : 21)
  const [sealedFrom, setSealedFrom] = useState(mode === 'forecast' ? '2024-01-01' : '2025-01-01')
  const [valMode, setValMode] = useState<'walkforward' | 'split'>('walkforward')
  const [infoTime, setInfoTime] = useState<'stamp' | 'release'>('stamp')
  const [splitDate, setSplitDate] = useState('2021-01-01')  // 1회 분할의 테스트 시작일
  const [dataStart, setDataStart] = useState('')
  const [dataEnd, setDataEnd] = useState('')
  const [valStart, setValStart] = useState('')
  const [testStart, setTestStart] = useState('')  // 워크포워드: 평가 시작일(비우면 학습 창이 차는 첫날부터)
  const [testEnd, setTestEnd] = useState('')
  const [valFrac, setValFrac] = useState(0.15)
  const [seqLen, setSeqLen] = useState<number | ''>('')
  const [dataVersion, setDataVersion] = useState(meta.data_versions.at(-1)?.id ?? '')
  const [seed, setSeed] = useState<number>(meta.defaults.seed ?? 2025)
  const [poolTickers, setPoolTickers] = useState<string[]>([])   // 종목 묶어 학습: 다른 종목의 행을 학습에만 보탠다
  const [selectTop, setSelectTop] = useState<string>('')   // 입력 선별: 폴드마다 연관성 높은 입력 N개만. 비우면 전부
  const [domainFilter, setDomainFilter] = useState<string>('전체')
  const [kindFilter, setKindFilter] = useState<'all' | 'file' | 'api' | 'derived'>('all')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [sets, setSets] = useState<VariableSet[]>(meta.variable_sets)
  const [setId, setSetId] = useState('')

  const modelList = isForecast ? meta.models.filter(m => m.input_shape !== 'rule') : meta.models
  const model = meta.models.find(m => m.file === modelFile)!
  const isRule = model.input_shape === 'rule'
  const domains = useMemo(() => ['전체', ...Array.from(new Set(meta.variables.map(v => v.domain).filter(Boolean)))], [meta])
  const byGroup = useMemo(() => {
    const m = new Map<string, Variable[]>()
    for (const v of meta.variables) {
      if (domainFilter !== '전체' && v.domain !== domainFilter) continue
      if (kindFilter !== 'all' && (SOURCE_INFO[v.source]?.kind ?? 'api') !== kindFilter) continue
      if (!m.has(v.group)) m.set(v.group, [])
      m.get(v.group)!.push(v)
    }
    return m
  }, [meta, domainFilter, kindFilter])

  function changeFreq(f: 'D' | 'W' | 'M') {
    setFreq(f); setTrainLen(DEFAULT_VALIDATION[f].train_len); setTestLen(DEFAULT_VALIDATION[f].test_len)
  }
  function loadSet(id: string) {
    setSetId(id); const st = sets.find(x => x.id === id); if (!st) return
    const ok = st.variable_ids.filter(v => { const m = meta.variables.find(x => x.id === v); return m && m.usable && m.has_data })
    setSelected(ok); setTransforms({ ...st.transforms }); setLagOverride({ ...st.lag_override })
    if (ok.length < st.variable_ids.length) setMsg({ ok: false, text: `세트의 변수 ${st.variable_ids.length - ok.length}개는 데이터가 없거나 사용 불가라 제외했습니다` })
  }
  async function saveSet() {
    const nm = prompt('세트 이름 (예: 반도체+금리 12개)'); if (!nm) return
    const id = nm.replace(/[^\w가-힣]+/g, '_').toLowerCase()
    const st: VariableSet = { id, name: nm, description: '', variable_ids: selected, transforms: Object.fromEntries(selected.filter(v => transforms[v] != null).map(v => [v, transforms[v]])), lag_override: Object.fromEntries(selected.filter(v => lagOverride[v] != null && lagOverride[v] !== meta.variables.find(x => x.id === v)?.lag_days).map(v => [v, lagOverride[v]])), recommended_for: [modelFile] }
    await api.saveSet(st); setSets(ss => [...ss.filter(x => x.id !== id), st]); setSetId(id)
  }
  function toggle(id: string) { setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]) }
  function toggleGroup(vars: Variable[]) {
    const ids = vars.filter(v => v.usable && v.has_data).map(v => v.id)
    const all = ids.every(i => selected.includes(i))
    setSelected(s => all ? s.filter(x => !ids.includes(x)) : Array.from(new Set([...s, ...ids])))
  }

  const targetFreq = (meta.forecast_targets.find(t => t.id === targetVar)?.freq ?? 'M') as 'D' | 'W' | 'M'
  const effFreq: 'D' | 'W' | 'M' = isForecast ? targetFreq : freq  // 파라미터 기본값을 고르는 주기 (엔진 freq_for_defaults 와 동일 규칙)
  const [hoverParam, setHoverParam] = useState<string | null>(null)
  const [flow, setFlow] = useState<any>(null)
  const [flowBusy, setFlowBusy] = useState(false)
  const [flowErr, setFlowErr] = useState<string | null>(null)
  const [vHover, setVHover] = useState<string | null>(null)
  const seqDefault = model.seq_len_default?.[effFreq] ?? DEFAULT_SEQ[effFreq] ?? 20
  const paramValue = (key: string) => {
    if (key === '@seq_len') return seqLen === '' ? seqDefault : seqLen
    const p: any = model.params[key]; if (!p) return undefined
    return params[key] ?? p.default_by_freq?.[effFreq] ?? p.default
  }
  // 파라미터 묶음: 모델이 group 을 선언하면 그 순서, 아니면 한 묶음. 창 모델은 seq_len 칸을 첫 묶음 앞에 둔다
  const paramGroups: [string, string[]][] = (() => {
    const keys = Object.keys(model.params)
    const order = model.param_groups ?? []
    const by: Record<string, string[]> = {}
    keys.forEach(k => { const g = (model.params[k] as any).group ?? '파라미터'; (by[g] ??= []).push(k) })
    const names = [...order.filter(g => by[g]), ...Object.keys(by).filter(g => !order.includes(g))]
    const out: [string, string[]][] = names.map(g => [g, by[g]])
    if (model.input_shape === 'window') { const g = out.find(([n]) => n === '입력'); if (g) g[1].unshift('@seq_len'); else out.unshift(['입력', ['@seq_len']]) }
    return out
  })()
  const config = {
    name: name || undefined, mode, target_variable: isForecast ? targetVar : null, ...(isForecast ? { forecast_info_time: infoTime } : {}), ticker, frequency: isForecast ? targetFreq : freq, target: { kind, k, threshold }, variable_ids: selected,
    ...((model.input_shape === 'table' || model.input_shape === 'window') && selectTop !== '' && Number(selectTop) >= 1 ? { select_top: Number(selectTop) } : {}),   // 입력으로 학습하는 모델만(검토 12)
    ...(!isForecast && model.input_shape === 'table' && poolTickers.filter(t => t !== ticker).length ? { pool_tickers: poolTickers.filter(t => t !== ticker) } : {}),
    variable_transforms: Object.fromEntries(selected.filter(v => transforms[v] != null && transforms[v] !== (meta.variables.find(x => x.id === v)?.transform ?? 'none')).map(v => [v, transforms[v]])),  // 기본값과 같으면 보내지 않음
    lag_override: Object.fromEntries(selected.filter(v => lagOverride[v] != null && lagOverride[v] !== meta.variables.find(x => x.id === v)?.lag_days).map(v => [v, lagOverride[v]])),
    use_technical: useTechnical, ...(useTechnical && techSel ? { technical_features: techSel } : {}), fill, model: modelFile, model_params: params,
    trade: { threshold: tradeThreshold, fee_rate: fee, slippage_bp: slippage, ...(sizing === 'kelly' ? { sizing, kelly_fraction: kellyFrac } : {}) },
    start: dataStart || null, end: dataEnd || null, ...(model.input_shape === 'window' && seqLen !== '' ? { seq_len: seqLen } : {}),
    validation: { train_len: trainLen, test_len: testLen, sealed_from: sealedFrom || null, mode: valMode, val_frac: valFrac,
      test_start: valMode === 'split' ? splitDate : (testStart || null), test_end: testEnd || null, val_start: valMode === 'split' && valStart ? valStart : null }, data_version: dataVersion || null, seed,
  }
  // 선택한 변수(+예측 대상)가 모두 '실제로 갱신되는' 공통 기간. 시작 = 가장 늦게 시작하는 변수가 쓸 수 있게 되는 날(기준일+발표시차),
  // 끝 = 가장 먼저 끝나는 변수의 마지막 값이 다음 값으로 바뀌었어야 할 날(기준일+1주기+발표시차). 그 뒤는 같은 값이 계속 이어진다(낡은 값).
  // 모델 흐름도: 설정이 바뀌면 0.5초 뒤 서버에서 실제 모양을 다시 잰다
  const flowKey = JSON.stringify(config)
  useEffect(() => {
    setFlowBusy(true)
    const t = setTimeout(() => {
      api.flow(config).then(r => { setFlow(r); setFlowErr(null) }).catch(e => setFlowErr(e.message)).finally(() => setFlowBusy(false))
    }, 500)
    return () => clearTimeout(t)
  }, [flowKey])
  // 검증 계획: 워크포워드·1회 분할 모두 실제로 쓰이는 구간(엔진과 같은 규칙으로 센 값) + 1회 분할의 비율로 날짜 채우기
  const [ratios, setRatios] = useState<[number, number, number]>([60, 10, 30])   // 원본 BIVA data_loader.py 의 학습·검증·시험 비율
  const [plan, setPlan] = useState<any>(null)
  const [planBusy, setPlanBusy] = useState(false)
  const [planErr, setPlanErr] = useState<string | null>(null)
  useEffect(() => {
    if (valMode === 'split' && !splitDate) { setPlan(null); setPlanErr(null); return }
    setPlanBusy(true)
    const t = setTimeout(() => {
      api.splitPlan(config).then(r => { setPlan(r); setPlanErr(null) }).catch(e => { setPlan(null); setPlanErr(e.message) }).finally(() => setPlanBusy(false))
    }, 600)
    return () => clearTimeout(t)
  }, [flowKey, valMode])
  const fillByRatio = () => {
    setPlanBusy(true)
    api.splitPlan(config, ratios).then(r => {
      setValStart(r.dates.val_start ?? ''); setSplitDate(r.dates.test_start ?? ''); setTestEnd(r.dates.test_end ?? '')
      setPlan(r); setPlanErr(null)
    }).catch(e => setPlanErr(e.message)).finally(() => setPlanBusy(false))
  }
  const ratioSum = ratios[0] + ratios[1] + ratios[2]
  const techAll: { name: string; group: string; desc: string }[] = flow?.input?.technical_all ?? []
  useEffect(() => {  // 주기·모드가 바뀌어 고른 이름이 목록에 없으면 전부 선택으로 되돌린다(이름이 주기마다 다름)
    if (techSel && techAll.length && techSel.some(n => !techAll.find(t => t.name === n))) setTechSel(null)
  }, [techAll.map(t => t.name).join(',')])
  function setTech(next: string[]) {
    const all = techAll.map(t => t.name)
    if (next.length === 0) { setUseTechnical(false); setTechSel(null); return }  // 모두 끄면 블록 자체를 끈다
    setTechSel(all.every(n => next.includes(n)) ? null : all.filter(n => next.includes(n)))
  }
  const techOn = (n: string) => techSel === null || techSel.includes(n)
  const common = useMemo(() => {
    const vs = selected.map(id => meta.variables.find(v => v.id === id)).filter(v => v && v.stored) as Variable[]
    const tgt = isForecast ? meta.variables.find(v => v.id === targetVar) : undefined
    if (!vs.length && !tgt?.stored) return null
    const starts = vs.map(v => ({ id: v.id, d: addDays(v.stored!.first.slice(0, 10), v.lag_days || 0) }))
    const ends = vs.map(v => ({ id: v.id, d: addDays(v.stored!.last.slice(0, 10), (PERIOD_DAYS[v.freq] ?? 31) + (v.lag_days || 0)) }))
    if (tgt?.stored) { starts.push({ id: tgt.id, d: tgt.stored.first.slice(0, 10) }); ends.push({ id: tgt.id, d: tgt.stored.last.slice(0, 10) }) }
    const s0 = starts.reduce((a, b) => (b.d > a.d ? b : a)), e0 = ends.reduce((a, b) => (b.d < a.d ? b : a))
    return { start: s0.d, startBy: s0.id, end: e0.d, endBy: e0.id }
  }, [selected, targetVar, isForecast, meta])
  const levelsNoVars = model.input_shape === 'levels' && selected.length === 0
  const tlFrom = dataStart || common?.start || ''
  const tlTo = dataEnd || common?.end || ''
  const testA = valMode === 'split' ? splitDate : testStart
  const testB = testEnd || (sealedFrom ? addDays(sealedFrom, -1) : tlTo)
  const segs = valMode === 'split'
    ? [{ label: '학습', a: tlFrom, b: valStart || testA, cls: 'tl-train' }, ...(valStart ? [{ label: '검증', a: valStart, b: testA, cls: 'tl-val' }] : []),
       { label: '테스트', a: testA, b: testB, cls: 'tl-test' }, ...(sealedFrom ? [{ label: '봉인', a: sealedFrom, b: tlTo, cls: 'tl-seal' }] : [])]
    : [{ label: '첫 학습 창', a: tlFrom, b: testA || tlFrom, cls: 'tl-train' }, { label: '평가(창을 밀며)', a: testA || tlFrom, b: testB, cls: 'tl-test' }, ...(sealedFrom ? [{ label: '봉인', a: sealedFrom, b: tlTo, cls: 'tl-seal' }] : [])]
  const unconfirmed = selected.filter(id => !meta.variables.find(v => v.id === id)?.lag_confirmed)
  const leak = fill !== 'ffill'

  async function submit() {
    setBusy(true); setMsg(null)
    try {
      const r = await api.submit(config)
      onSubmitted(r.job.id)
    } catch (e: any) { setMsg({ ok: false, text: e.message }) } finally { setBusy(false) }
  }

  return (
    <div>
      <h2>1. 실험 설정 <span className="hint">— {isForecast ? '예측 연구' : '매매 시뮬레이션'}</span></h2>
      {isForecast ? <p className="lead">한 번 돌릴 때마다 예측 실험 카운터가 올라갑니다(지금 {(meta as any).experiment_counters?.forecast ?? 0}회). 설정을 여러 번 바꿔 가장 좋은 하나만 보고하면 우연을 실력으로 착각하기 쉽습니다 — 논문에는 돌린 실험 전부와 그 횟수를 함께 보고하세요.</p>
        : <p className="lead">한 번 돌릴 때마다 실험 카운터가 올라가고, 결과의 신뢰도(Deflated Sharpe)는 지금까지 돌린 횟수({meta.experiment_counter}회)를 반영해 깎입니다. 변수를 바꿔 가며 여러 번 돌리는 것 자체가 다중검정입니다.</p>}

      <div className="panel">
        <h3>대상과 타겟</h3>
        <div className="row">
          <div className="field"><label>실험 이름</label><input value={name} onChange={e => setName(e.target.value)} placeholder={isForecast ? "예: f11_ridge_macro_k1" : "예: h6_weekly_sox"} /></div>
          {isForecast ? <div className="field"><label>예측 대상(시계열)</label><select value={targetVar} onChange={e => { setTargetVar(e.target.value); const f = meta.forecast_targets.find(t => t.id === e.target.value)?.freq as any; if (f && DEFAULT_VALIDATION[f]) { setTrainLen(DEFAULT_VALIDATION[f].train_len); setTestLen(DEFAULT_VALIDATION[f].test_len) } }}>
            {meta.forecast_targets.map(t => <option key={t.id} value={t.id}>{t.name} ({t.id}, {FREQ_LABEL[t.freq] ?? t.freq})</option>)}</select></div>
          : <>
          <div className="field"><label>종목</label><select value={ticker} onChange={e => setTicker(e.target.value)}>{meta.tickers.map(t => <option key={t.id} value={t.id}>{t.name} ({t.id})</option>)}</select></div>
          <div className="field"><label>매매 주기</label><select value={freq} onChange={e => changeFreq(e.target.value as any)}>{Object.entries(FREQ_LABEL).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></div>
          </>}
          <div className="field"><label>{isForecast ? '맞히는 방식' : '타겟 종류'}</label><select value={kind} onChange={e => setKind(e.target.value)}>{Object.entries(isForecast ? FC_KIND_LABEL : KIND_LABEL).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></div>
          <div className="field"><label>{isForecast ? `몇 ${UNIT[targetFreq] ?? '기간'} 뒤를 맞히나 (k)` : '예측 지평 k (주기 수)'}</label><input type="number" min={1} value={k} onChange={e => setK(+e.target.value)} /></div>
          {kind === 'event' && <div className="field"><label>이벤트 임계치</label><input type="number" step="0.001" value={threshold} onChange={e => setThreshold(+e.target.value)} /></div>}
          <div className="field"><label>데이터 버전</label><select value={dataVersion} onChange={e => setDataVersion(e.target.value)}>{meta.data_versions.map(v => <option key={v.id} value={v.id}>{v.id}{v.incomplete ? ' (일부 받지 못함)' : ''}</option>)}</select></div>
        </div>

        {isForecast ? <div className="problem">
          <div className="pq"><span className="step">문제</span><span><b>{(meta.forecast_targets.find(t => t.id === targetVar)?.name ?? targetVar)}</b>의 지금(t)까지 값{selected.length > 0 ? ` + 선택한 변수 ${selected.length}개` : ''}{useTechnical ? ' + 자체 지표' : ''}로 → <b>{k}{UNIT[targetFreq] ?? '기간'} 뒤의 값</b>을 맞힌다.</span></div>
          <div className="pq"><span className="step">학습</span><span>{kind === 'direction' ? '오르면 1, 내리면 0 을 맞히도록 학습합니다(값은 나오지 않음).' : <>모델은 값 자체가 아니라 <b>{kind === 'logreturn' ? '로그 변화율 ln(v[t+k]/v[t])' : '변화율 v[t+k]/v[t] − 1'}</b> 을 배웁니다 — 값에는 추세가 있어 그대로 배우면 추세만 따라가기 때문. 예측한 변화율은 다시 <b>값</b>으로 바꿉니다: v̂[t+k] = v[t] × {kind === 'logreturn' ? 'e^(예측)' : '(1 + 예측)'}.</>}</span></div>
          <div className="pq"><span className="step">채점</span><span>{kind === 'direction' ? '방향 맞힘(%)과 Brier 점수를 "지난 방향 그대로"와 비교합니다.' : <>k기간 뒤 <b>값의 평균 오차율(MAPE)</b> 을 두 기준과 비교: <b>지난 변화 그대로</b>(naive, 꼭 이겨야 함) · <b>변화 없음</b>. 체결·비용은 없습니다.</>}</span></div>
        </div> : <p className="hint">'타겟 = t 마감에 결정 → t+1 시가 진입 → t+1+k 시가 청산 수익률. 체결은 항상 다음 주기 시가입니다.'</p>}
      </div>

      <div className="panel">
        <h3>모델</h3>
        <div className="row">
          <div className="field"><label>모델</label><select value={modelFile} onChange={e => { setModelFile(e.target.value); setParams({}); setSeqLen('') }}>{MODEL_GROUPS.map(([shape, label]) => { const ms = modelList.filter(m => m.input_shape === shape); return ms.length ? <optgroup key={shape} label={label}>{ms.map(m => <option key={m.file} value={m.file}>{m.name}</option>)}</optgroup> : null })}</select></div>
          {model.doc && <p className="hint model-doc">{model.doc}</p>}
        </div>
        <ModelFlow model={modelFile} flow={flow} loading={flowBusy} error={flowErr} value={paramValue} hoverParam={hoverParam} onHoverParam={setHoverParam} config={config} />
        <div className="pgrid">{paramGroups.flatMap(([g, keys]) => keys.map(key => {
            const gtag = paramGroups.length > 1 ? <span className="pc-group">{g}</span> : null
            const hot = hoverParam === key
            if (key === '@seq_len') return <div key={key} id={'pc-' + key} className={'pcard' + (hot ? ' hot' : '') + (seqLen !== '' ? ' changed' : '')} onMouseEnter={() => setHoverParam(key)} onMouseLeave={() => setHoverParam(null)}>
              {gtag}<PcHead h={helpOf(modelFile, key)} code="seq_len" def={`기본 ${seqDefault}${seqLen !== '' ? ' · 변경됨' : ''}`} />
              <input type="number" min={2} placeholder={String(seqDefault)} value={seqLen} onChange={e => setSeqLen(e.target.value === '' ? '' : +e.target.value)} />
              <PcHelp h={helpOf(modelFile, key)} detail={`단위: ${UNIT[effFreq] ?? '기간'}. 실험 설정 값이라 모든 딥러닝 모델에 공통.${model.seq_len_default ? ' 원본 기본을 따름.' : ''}`} />
            </div>
            const p: any = model.params[key]
            const def = p.default_by_freq?.[effFreq] ?? p.default
            const cur = params[key] ?? def
            const changed = params[key] !== undefined && params[key] !== def
            const num = (v: string) => (p.type === 'int' ? parseInt(v, 10) : parseFloat(v))
            return <div key={key} id={'pc-' + key} className={'pcard' + (hot ? ' hot' : '') + (changed ? ' changed' : '')} onMouseEnter={() => setHoverParam(key)} onMouseLeave={() => setHoverParam(null)}>
              {gtag}<PcHead h={helpOf(modelFile, key)} code={key} def={`${p.default_by_freq ? `기본(${FREQ_LABEL[effFreq] ?? effFreq})` : '기본'} ${def === '' ? '없음' : String(def)}${changed ? ' · 변경됨' : ''}`} />
              {p.choices ? <select value={String(cur)} onChange={e => setParams({ ...params, [key]: p.type === 'str' ? e.target.value : num(e.target.value) })}>{p.choices.map((c: any) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}</select>
                : p.type === 'bool' ? <select value={String(cur)} onChange={e => setParams({ ...params, [key]: e.target.value === 'true' })}><option value="true">예</option><option value="false">아니오</option></select>
                : <input type={p.type === 'str' ? 'text' : 'number'} step={p.type === 'float' ? 'any' : 1} min={p.min} max={p.max} value={cur} onChange={e => setParams({ ...params, [key]: p.type === 'str' ? e.target.value : num(e.target.value) })} />}
              <PcHelp h={helpOf(modelFile, key)} detail={p.desc + (p.min != null && p.type !== 'bool' && !p.choices ? ` (범위 ${p.min}~${p.max})` : '')} />
            </div>
          }))}</div>
        <p className="hint">비워 둔 칸은 기본값(주기별 기본이 있으면 그 값)이 쓰이고, 결과 json 의 model.params 에 실제 사용값이 기록됩니다. 바꾼 칸은 테두리가 진해집니다. <button type="button" className="link" onClick={() => { setParams({}); setSeqLen('') }} disabled={Object.keys(params).length === 0 && seqLen === ''}>기본값으로 되돌리기</button></p>
        {model.input_shape === 'window' && <p className="hint">창(seq_len)은 <b>실제 있는 행으로만</b> 만듭니다 — 학습 구간 맨 앞 seq_len−1 행은 과거가 모자라 창을 만들지 않고(빈 칸을 0이나 가짜 값으로 채우지 않음), 테스트 첫 창은 바로 앞의 실제 과거 행을 붙여 완성합니다. 창 안에 <b>변수가 아직 없던 칸</b>이 있으면 기본은 학습 구간 평균으로 채우고, BIVA 에서 use_brits=예(원본 기본) 이면 그 칸을 <b>BRITS 가 그 창 안의 값으로 보간</b>합니다(창 밖·미래 값은 쓰지 않음).</p>}
        {levelsNoVars && <div className="verdict fail">VECM 은 아래에서 <b>입력 변수를 1개 이상</b> 골라야 실행됩니다 — 대상과 변수들의 수준값이 장기적으로 함께 움직이는 관계(공적분)를 쓰는 모델이라 대상 하나만으로는 정의되지 않습니다. 기술지표(자체 지표)는 쓰이지 않습니다. 예: F11 이면 F7(소비자물가)·B1(금리) 등 2~5개.</div>}
        {model.input_shape === 'levels' && <p className="hint">VECM 은 대상과 <b>아래에서 고른 변수들의 수준값</b>(변환하지 않은 원래 값, 발표시차는 적용)이 장기적으로 함께 움직이는 관계(공적분)를 씁니다. 변수의 변환 칸과 기술지표 블록은 무시됩니다. 변수는 2~5개 정도가 적당합니다 — 많으면 월간 60개 학습 창에서 추정이 불안정해집니다.</p>}
        {model.input_shape === 'series' && <p className="hint">통계 모델(ARIMA·SARIMA)은 <b>대상 시계열 자체의 과거 변화율만</b> 씁니다 — 아래 입력 변수·기술지표 선택은 무시됩니다. 결정일까지의 변화율로 k기간 앞을 예측해 복리로 묶습니다.</p>}
        {isRule && <p className="hint">규칙 모델은 학습이 없습니다. 외생변수를 쓰는 규칙(예: SOX Gap Fade)은 아래에서 해당 변수를 골라야 합니다.</p>}
      </div>

      <div className="panel">
        <h3>입력 변수 <span className="hint">선택 {selected.length}개</span></h3>
        <div className="toolbar">
          <span className="hint">변수 세트:</span>
          <select value={setId} onChange={e => loadSet(e.target.value)}>
            <option value="">(직접 선택)</option>
            {sets.map(st => <option key={st.id} value={st.id}>{st.name} · {st.variable_ids.length}개{st.recommended_for.includes(modelFile) ? ' ★' : ''}</option>)}
          </select>
          <button onClick={saveSet} disabled={selected.length === 0}>현재 선택을 세트로 저장</button>
          {setId && <span className="hint">{sets.find(x => x.id === setId)?.description}</span>}
        </div>
        <p className="hint">★ = 이 모델에 권장된 세트. 세트는 모델과 독립이라 어떤 모델에나 붙일 수 있고, 같은 세트로 모델을 바꿔 가며 비교하는 것이 이 도구의 기본 사용법입니다.</p>
        <div className="toolbar">
          <label><input type="checkbox" checked={useTechnical} onChange={e => setUseTechnical(e.target.checked)} /> {isForecast ? '대상 시계열 자체의 지표 블록(A: 변화율·이동평균 비율·변동성·RSI·MACD — 거래량 없음)' : '기술지표 블록(A: 수익률·이동평균·변동성·RSI·MACD·거래량)'}</label>
          <span className="hint">도메인:</span>
          {domains.map(d => <button key={d} className={'chip' + (domainFilter === d ? ' on' : '')} onClick={() => setDomainFilter(d)}>{d}</button>)}
          <span className="hint" style={{ marginLeft: 8 }}>출처:</span>
          {(['all', 'file', 'api', 'derived'] as const).map(k => <button key={k} className={'chip' + (kindFilter === k ? ' on' : '')} onClick={() => setKindFilter(k)}>{k === 'all' ? '전체' : `${KIND_SYMBOL[k]} ${SRC_KIND_LABEL[k]}`}</button>)}
        </div>
        {useTechnical && flow?.input && !flow.input.technical_all && <p className="hint">기술지표 고르기 목록이 안 보이면 서버(run_server.bat)를 다시 시작하세요 — 화면만 새것이고 서버는 예전 코드입니다.</p>}
        {useTechnical && techAll.length > 0 && <details className="techlist">
          <summary>모델에 들어가는 열 = <b>기술지표 {techSel ? techSel.length : techAll.length}/{techAll.length}개</b>{flow?.input ? ` + 아래에서 고른 변수 ${flow.input.n_features - techCount(flow)}개 = ${flow.input.n_features}개` : ''} · 기술지표 고르기</summary>
          <p className="hint">기술지표는 아래 변수 목록에 없습니다 — 변수 사전에서 가져오는 것이 아니라 {isForecast ? '예측 대상' : `ETF(${ticker})`} 의 가격{isForecast ? '' : '·거래량'}으로 그때그때 계산하기 때문입니다. 처음에는 전부 켜져 있고, 하나씩 또는 묶음으로 뺄 수 있습니다. 모두 빼면 블록이 꺼집니다. 고른 목록은 결과 json 의 config.technical_features(전부면 비어 있음)와 feature_names 에 남습니다. 이름(창 길이)은 주기(일·주·월)마다 달라 주기를 바꾸면 전부 선택으로 돌아갑니다.</p>
          <div className="toolbar">
            <button type="button" className="chip" onClick={() => setTechSel(null)} disabled={techSel === null}>전부 선택</button>
            <button type="button" className="chip" onClick={() => setTech([])} title="기술지표를 모두 빼고 블록을 끕니다(다시 넣으려면 위 '기술지표 블록' 체크)">전부 빼기</button>
            {Array.from(new Set(techAll.map(t => t.group))).map(g => {
              const ns = techAll.filter(t => t.group === g).map(t => t.name)
              const allOn = ns.every(techOn)
              const cur = techAll.map(t => t.name).filter(techOn)
              return <button key={g} type="button" className={'chip' + (allOn ? ' on' : '')} title={allOn ? `${g} 묶음 빼기` : `${g} 묶음 넣기`}
                onClick={() => setTech(allOn ? cur.filter(n => !ns.includes(n)) : Array.from(new Set([...cur, ...ns])))}>{g} {ns.filter(techOn).length}/{ns.length}</button>
            })}
          </div>
          <ul style={{ listStyle: 'none', paddingLeft: 4 }}>{techAll.map(t => <li key={t.name}><label><input type="checkbox" checked={techOn(t.name)} onChange={() => { const cur = techAll.map(x => x.name).filter(techOn); setTech(techOn(t.name) ? cur.filter(n => n !== t.name) : [...cur, t.name]) }} /> <code>{t.name}</code> {t.desc}</label></li>)}</ul>
        </details>}
        <p className="hint legend">출처 배지: <span className="badge srcb src-xlsx">■ 엑셀</span> 업로드한 엑셀 파일 · <span className="badge srcb src-ecos">● ECOS</span><span className="badge srcb src-fred">● FRED</span><span className="badge srcb src-yf">● Yahoo</span> API 자동 수집 · <span className="badge srcb src-derived">◆ 파생</span> 다른 변수로 계산. 마우스를 올리면 파일·시트·열 또는 API 코드와 저장 기간이 보입니다.</p>
        <p className="hint legend"><span className="badge lagb pass">✓ 37일</span> 발표시차 = 기준일(예: 1월 CPI → 1-01) 뒤 며칠부터 그 값을 쓰는가. <b>✓</b> 공표일정으로 확인 · <span className="badge lagb warn">? 82일</span> 근거 부족. 마우스를 올리면 근거가 보입니다. 변수를 체크하면 변환·발표시차 칸이 열리고 <b>기본값이 미리 들어가 있습니다</b> — 바꾼 칸만 주황색으로 표시되고 <b>이 실험에만</b> 적용됩니다 — 엑셀 변수 사전과 받아 둔 데이터는 그대로이고, 바꾼 내용은 이 실험의 결과에 기록됩니다. 기본값 자체를 바꾸려면 변수 사전 화면의 [값 보기] → '고치기'(도움말 20장 4-5).</p>
        {Array.from(byGroup.entries()).map(([g, vars]) => (
          <div key={g} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 2 }}>
              <strong style={{ fontSize: 13 }}>{g || '(묶음 없음)'}</strong>
              <button className="chip" onClick={() => toggleGroup(vars)}>그룹 전체</button>
            </div>
            <div className="varlist">
              {vars.map(v => {
                const disabled = !v.usable || !v.has_data
                const on = selected.includes(v.id)
                return (
                  <label key={v.id} title={disabled ? (v.duplicate_of ? `${v.duplicate_of} 와 중복 → 자동 치환` : !v.active ? '마스터에서 active=N' : '데이터 없음') : ''} className={on ? 'on' : undefined} style={{ opacity: disabled ? .45 : 1 }}>
                    <input type="checkbox" disabled={disabled} checked={on} onChange={() => toggle(v.id)} />
                    <span className="id" title={v.id}>{v.id}</span><span className="nm" title={v.description ? `${v.name} — ${v.description}` : v.name}>{v.name}</span>
                    <span className="badge muted">{v.freq}</span>
                    <SourceBadge v={v} />
                    <span className={'badge lagb ' + (v.lag_confirmed ? 'pass' : 'warn')} title={`발표시차: 기준일 뒤 ${v.lag_days}일부터 사용 (${v.lag_confirmed ? '공표일정으로 확인' : '미확인'})\n${lagEvidence(v.note)}`}>{v.lag_confirmed ? '✓' : '?'} {v.lag_days}일</span>
                    {on && (() => {
                      const lagNow = lagOverride[v.id] ?? v.lag_days
                      const lagChanged = lagOverride[v.id] != null && lagOverride[v.id] !== v.lag_days
                      const trNow = transforms[v.id] ?? v.transform ?? 'none'
                      const trChanged = transforms[v.id] != null && transforms[v.id] !== (v.transform ?? 'none')
                      return <span className="ctl" onClick={e => e.preventDefault()}>
                        <span className={'ctlbox' + (trChanged ? ' changed' : '')} title={`변환 — 원래 주기에서 먼저 적용. 기본값(변수 사전) ${v.transform ?? 'none'}`}>
                          <span className="cl">변환</span>
                          <TransformSelect compact value={trNow} onChange={t => setTransforms({ ...transforms, [v.id]: t })} options={meta.transform_options} />
                        </span>
                        <span className={'ctlbox' + (lagChanged ? ' changed' : '')} title={`발표시차(달력일). 기본값 ${v.lag_days}일 — 바꾸면 이 실험에서만 적용되고 결과에 기록됩니다.\n${lagEvidence(v.note)}`}>
                          <span className="cl">발표시차</span>
                          <input type="number" min={0} value={lagNow} onChange={e => setLagOverride({ ...lagOverride, [v.id]: e.target.value === '' ? (undefined as any) : +e.target.value })} />
                          <span className="cl">일</span>
                        </span>
                        {(lagChanged || trChanged) && <button type="button" className="link" onClick={() => { const lo = { ...lagOverride }; delete lo[v.id]; setLagOverride(lo); const tr = { ...transforms }; delete tr[v.id]; setTransforms(tr) }}>기본값으로</button>}
                      </span>
                    })()}
                  </label>
                )
              })}
            </div>
          </div>
        ))}
        {unconfirmed.length > 0 && <p className="hint">lag 미확인 변수 {unconfirmed.length}개 — 결과에 "lag 미확인" 표시가 붙습니다. 변수 사전에서 lag_days 를 확인하면 사라집니다.</p>}
      </div>

      <div className="panel">
        <h3>{isForecast ? '검증' : '매매 규칙과 검증'}</h3>
        <div className="row">
          <div className="field"><label>저주기 채움</label><select value={fill} onChange={e => setFill(e.target.value)}><option value="ffill">ffill (미래 정보 없음)</option><option value="bfill">bfill (미래 정보 누출!)</option><option value="interp">interp (미래 정보 누출!)</option></select></div>
          {!isForecast && <>
          <div className="field"><label title="문턱: 예측이 문턱을 넘으면 전부 보유, 아니면 현금(기본). 켈리: 과거에 같은 자신감으로 예측했을 때의 실제 수익으로 비율(0~100%)을 정함 — 결정일에 결과가 나온 과거만 씀">투자 비율 정하기</label>
            <select value={sizing} onChange={e => setSizing(e.target.value as any)}><option value="threshold">문턱 (넘으면 전부, 아니면 현금)</option><option value="kelly">켈리 (과거 결과로 비율 정함)</option></select></div>
          {sizing === 'kelly' && <div className="field"><label title="0.5 = 절반 켈리(기본, 추정이 틀려도 덜 잃음). 1 = 전부 켈리">켈리 분수</label><input type="number" step="0.05" min={0.05} max={1} value={kellyFrac} onChange={e => setKellyFrac(+e.target.value)} /></div>}
          {sizing === 'threshold' && <div className="field"><label>진입 임계치</label><input type="number" step="any" value={tradeThreshold} onChange={e => setTradeThreshold(+e.target.value)} /></div>}
          <div className="field"><label>수수료율</label><input type="number" step="any" value={fee} onChange={e => setFee(+e.target.value)} /></div>
          <div className="field"><label>슬리피지(bp, 편도)</label><input type="number" step="any" value={slippage} onChange={e => setSlippage(+e.target.value)} /></div>
          </>}
          {isForecast && <div className="field"><label>외생변수 정보 시점</label><select value={infoTime} onChange={e => setInfoTime(e.target.value as any)}><option value="stamp">대상 기준일 t (보수적)</option><option value="release">대상 값 공개 시점 t+lag (현실적)</option></select></div>}
          <div className="field"><label>검증 방식</label><select value={valMode} onChange={e => setValMode(e.target.value as any)}><option value="walkforward">워크포워드(창을 밀며 반복 재학습)</option><option value="split">1회 분할(학습·검증·테스트 날짜 지정)</option></select></div>
          <div className="field"><label>시드</label><input type="number" value={seed} onChange={e => setSeed(+e.target.value)} /></div>
          {(model.input_shape === 'table' || model.input_shape === 'window') && <div className="field"><label>연관성 높은 입력만 쓰기 <span className="hint">개수 · 비우면 전부</span></label><input value={selectTop} onChange={e => setSelectTop(e.target.value.replace(/[^0-9]/g, ''))} placeholder="비움 = 전부 사용" />
            <span className="hint">폴드마다 학습 구간 안에서만 연관성을 재서 고릅니다(미래를 보고 고르지 않음). 비워도 연관성 순위는 결과에 나옵니다.</span></div>}
          {!isForecast && model.input_shape === 'table' && meta.tickers.length > 1 && <div className="field"><label>종목 묶어 학습 <span className="hint">다른 종목의 행을 학습에만 보탬</span></label>
            <div>{meta.tickers.filter(t => t.id !== ticker).map(t => { const have = meta.data_versions.find(v => v.id === dataVersion)?.tickers?.includes(t.id) ?? true
              return <label key={t.id} style={{ display: 'block', fontWeight: 'normal' }}><input type="checkbox" disabled={!have} checked={poolTickers.includes(t.id)} onChange={e => setPoolTickers(e.target.checked ? [...poolTickers, t.id] : poolTickers.filter(x => x !== t.id))} /> {t.name} ({t.id}){have ? '' : ' — 이 데이터 버전에 가격이 없습니다(데이터 가져오기 필요)'}</label> })}</div>
            <span className="hint">예측·평가·검정은 고른 종목({ticker})의 날짜에서만 합니다. 같은 지수를 따르는 ETF(예: KODEX·TIGER 반도체)는 거의 같은 값이라 보태도 새 정보가 적습니다.</span></div>}
        </div>
        <details className="techlist">
          <summary>이 칸들은 모든 모델에 똑같이 적용됩니다 — 용어 설명 보기</summary>
          <ul style={{ columns: 1 }}>
            {!isForecast && <li><b>매매 규칙</b>(진입 임계치·수수료·슬리피지): 모델이 무엇이든 같은 규칙으로 사고팝니다. 수익률을 예측하는 모델은 예측값 &gt; 임계치이면 다음날 시가에 매수(보유), 아니면 현금. 확률을 내는 모델은 확률 &gt; 0.5+임계치. 규칙 모델(이동평균 교차 등)은 스스로 0/1 신호를 내므로 임계치를 쓰지 않습니다. 비교가 공정하려면 모든 실험의 매매 규칙이 같아야 합니다.</li>}
            <li><b>워크포워드</b>: 실제로 투자하는 사람처럼 시간 순서대로 진행합니다. 과거 {trainLen}{UNIT[effFreq] ?? '주기'}(학습 창)로 학습 → 그다음 {testLen}{UNIT[effFreq] ?? '주기'}(재학습 간격) 동안 예측·평가 → 창을 {testLen}칸 앞으로 밀어 다시 학습 … 을 끝까지 반복합니다. 미래 데이터로 학습하는 일이 없습니다.</li>
            <li><b>폴드</b>: 위 반복의 한 번(학습 창 하나 + 그 뒤 평가 구간 하나)을 말합니다. 예) 일간·학습 창 756·간격 21 로 10년이면 폴드가 약 90개. 결과의 '폴드 수'가 이 숫자입니다. 각 폴드의 평가 구간을 이어 붙인 것이 최종 성적입니다.</li>
            <li><b>검증 비율</b>: 값은 하나지만 쓰는 모델만 씁니다 — LSTM·GRU·BIVA·LightGBM 은 학습 창 뒤쪽 {Math.round(valFrac * 100)}% 를 떼어 '언제 학습을 멈출지'(조기종료)를 정하고, Ridge·ARIMA·SARIMA·VECM·규칙 모델은 쓰지 않습니다(칸이 잠김). 검증 구간은 성적에 들어가지 않습니다.</li>
            <li><b>규칙 모델</b>(이동평균 교차·RSI 등): 학습이 없지만, 같은 설정의 학습 모델이 평가받는 날짜와 <b>같은 구간만</b> 성적에 넣습니다 — 그래야 같은 시장 상황에서 비교됩니다.</li>
            <li><b>1회 분할</b>: 학습·검증·테스트를 날짜로 한 번만 나눕니다. 논문 재현용. 테스트 구간이 하나라 운이 섞이기 쉽습니다.</li>
          </ul>
        </details>
        <h4 className="sub">기간 <span className="hint">— 날짜는 YYYY-MM-DD. 비우면 자동</span></h4>
        <div className="row">
          <div className={"field" + (vHover === 'start' ? " hot" : "")} onMouseEnter={() => setVHover('start')} onMouseLeave={() => setVHover(null)}><label>데이터 사용 시작</label><input value={dataStart} placeholder={common?.start ?? '처음부터'} onChange={e => setDataStart(e.target.value)} /></div>
          <div className="field"><label>데이터 사용 종료</label><input value={dataEnd} placeholder={common?.end ?? '끝까지'} onChange={e => setDataEnd(e.target.value)} /></div>
          {common && <div className="field"><label>&nbsp;</label><button type="button" onClick={() => { setDataStart(common.start); setDataEnd(common.end) }}>선택 변수 공통 기간으로</button>
            <span className="hint" style={{ fontSize: 11 }}>{common.start}({common.startBy}부터 쓸 수 있음) ~ {common.end}({common.endBy} 갱신 끝)</span></div>}
        </div>
        <div className="row">
          {valMode === 'walkforward' ? <>
            <div className={"field" + (vHover === 'train_len' ? " hot" : "")} onMouseEnter={() => setVHover('train_len')} onMouseLeave={() => setVHover(null)}><label>학습 창(주기 수)</label><input type="number" value={trainLen} onChange={e => setTrainLen(+e.target.value)} /></div>
            <div className={"field" + (vHover === 'test_len' ? " hot" : "")} onMouseEnter={() => setVHover('test_len')} onMouseLeave={() => setVHover(null)}><label>재학습 간격</label><input type="number" value={testLen} onChange={e => setTestLen(+e.target.value)} /></div>
            <div className={"field" + (vHover === 'val_frac' ? " hot" : "")} onMouseEnter={() => setVHover('val_frac')} onMouseLeave={() => setVHover(null)}><label>검증 비율 <span className="hint">— {model.uses_validation ? `${model.name ?? modelFile} 은 학습 창 뒤쪽 이 비율로 학습 멈출 때를 정함` : `${model.name ?? modelFile} 은 쓰지 않음(학습 창 전체로 학습)`}</span></label><input type="number" step="0.01" min={0} max={0.5} value={valFrac} disabled={!model.uses_validation} onChange={e => setValFrac(+e.target.value)} /></div>
            <div className={"field" + (vHover === 'test_start' ? " hot" : "")} onMouseEnter={() => setVHover('test_start')} onMouseLeave={() => setVHover(null)}><label>평가 시작일</label><input value={testStart} placeholder="학습 창이 차는 첫날" onChange={e => setTestStart(e.target.value)} /></div>
          </> : <>
            <div className="field"><label>비율로 나누기 <span className="hint">— 학습 : 검증 : 테스트 (%)</span></label>
              <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                {[0, 1, 2].map(i => <input key={i} type="number" min={0} max={100} step={1} style={{ width: 56 }} value={ratios[i]}
                  onChange={e => { const r = [...ratios] as [number, number, number]; r[i] = +e.target.value; setRatios(r) }} />)}
                <button type="button" disabled={ratioSum !== 100 || planBusy} onClick={fillByRatio}>비율로 날짜 채우기</button>
              </div>
              <span className="hint">{ratioSum !== 100 ? `합이 ${ratioSum} — 100 이어야 합니다` : '정답이 있는 결정일 수 N 으로 학습 = int(N×학습%), 테스트 = int(N×테스트%), 나머지 = 검증(원본 BIVA 규칙). 채운 날짜는 아래에서 고칠 수 있습니다.'}</span></div>
            <div className={"field" + (vHover === 'val_start' ? " hot" : "")} onMouseEnter={() => setVHover('val_start')} onMouseLeave={() => setVHover(null)}><label>검증 시작일 <span className="hint">— 딥러닝·LightGBM 만</span></label><input value={valStart} placeholder={`비우면 학습 뒤쪽 ${Math.round(valFrac * 100)}%`} onChange={e => setValStart(e.target.value)} /></div>
            <div className={"field" + (vHover === 'test_start' ? " hot" : "")} onMouseEnter={() => setVHover('test_start')} onMouseLeave={() => setVHover(null)}><label>테스트 시작일</label><input value={splitDate} onChange={e => setSplitDate(e.target.value)} /></div>
          </>}
          <div className={"field" + (vHover === 'test_end' ? " hot" : "")} onMouseEnter={() => setVHover('test_end')} onMouseLeave={() => setVHover(null)}><label>{valMode === 'split' ? '테스트 종료일' : '평가 종료일'}</label><input value={testEnd} placeholder={sealedFrom ? '봉인 전날' : '데이터 끝'} onChange={e => setTestEnd(e.target.value)} /></div>
          <div className={"field" + (vHover === 'sealed_from' ? " hot" : "")} onMouseEnter={() => setVHover('sealed_from')} onMouseLeave={() => setVHover(null)}><label>봉인 시작일 <span className="hint">— 최종 확인용, 개발 중 안 봄</span></label><input value={sealedFrom} onChange={e => setSealedFrom(e.target.value)} /></div>
        </div>
        <Timeline segs={segs} from={tlFrom} to={tlTo} />
        <details className="techlist">
          <summary><b>봉인이란?</b> — 마지막 기간을 '시험지 봉투'에 넣어 두는 것 · 날짜는 위 '봉인 시작일' 칸에서 정합니다{sealedFrom ? ` (지금 ${sealedFrom})` : ' (지금 비어 있음 = 봉인 없음)'}</summary>
          <ul style={{ columns: 1 }}>
            <li><b>왜 필요한가</b>: 변수·모델·파라미터를 바꿔 가며 여러 번 돌리고 결과를 보며 고치다 보면, 그 기간에만 우연히 잘 맞는 설정을 고르게 됩니다(데이터 엿보기·과적합). 그래서 마지막 기간을 봉투에 넣어 <b>개발하는 동안에는 아예 계산하지 않고</b>, 최종 후보가 정해지면 <b>단 한 번</b> 열어 채점합니다. 논문의 '표본 밖(out-of-sample) 최종 검증'에 해당합니다.</li>
            <li><b>어디서 정하나</b>: 이 화면 '매매 규칙과 검증 → 기간 → <b>봉인 시작일</b>'. 기본은 매매 시뮬레이션 2025-01-01, 예측 연구 2024-01-01. 비우면 봉인 없이 전 기간을 개발 평가에 씁니다(논문 최종 결과로는 권장하지 않음).</li>
            <li><b>개발 중에는</b>: 봉인 시작일 이후는 폴드를 만들지 않고, 규칙 모델 신호도 지우고, 성적에서도 뺍니다. 개발 평가는 <b>봉인 전날까지 빠짐없이</b>(마지막 폴드는 남은 날만큼) 계산합니다. 계산 따라가기의 결정일이 봉인 전으로만 고를 수 있는 이유입니다.</li>
            <li><b>날짜 고르는 법(권장)</b>: ① 실험을 시작하기 전에 정하고 결과를 본 뒤 옮기지 않기 — 옮기면 봉인의 의미가 없어집니다. ② 모든 실험이 같은 날짜를 쓰기 — 그래야 비교됩니다. ③ 봉인 구간이 너무 짧지 않게 — 일간은 1년(약 250거래일) 이상, 월간은 24개월 이상을 권합니다(짧으면 운의 영향이 큼).</li>
            <li><b>여는 법</b>: 개발 실험을 돌린 뒤 <b>결과 화면 → 결과 하나 선택 → 상세 아래 '봉인 평가'</b>에서 엽니다(미리보기 → 사전 기록 → 확인 2개 → 실행). 명령줄 <code>python cli.py run 설정.json --sealed</code> 도 같은 경로입니다. 워크포워드는 봉인 첫날부터 창을 밀며(실전과 같은 절차), 1회 분할은 봉인 전 전부로 한 번 학습해 봉인 구간만 채점합니다. 결과는 '…_sealed' 로 따로 저장되고 기록이 <code>results/{'{trading|forecast}'}/sealed_log.json</code> 에 남으며, 같은 설정을 다시 열려면 '재개봉' 확인을 따로 받습니다. <b>봉인을 연 가설은 그 결과로 끝</b>(통과 또는 폐기) — 결과를 보고 다시 고쳐 또 열면 봉인이 아닙니다.</li>
          </ul>
        </details>
        <ValidationDiagram mode={valMode} trainLen={trainLen} testLen={testLen} valFrac={valFrac} emb={isForecast ? k : 1 + k}
          seqLen={model.input_shape === 'window' ? (seqLen === '' ? seqDefault : seqLen) : null} usesVal={!!model.uses_validation} unit={UNIT[effFreq] ?? ''}
          dates={{ start: dataStart || common?.start, valStart, testStart: valMode === 'split' ? splitDate : testStart, testEnd, sealed: sealedFrom, end: dataEnd || common?.end }}
          hoverKey={vHover} onHover={setVHover} />
        <SplitPlan plan={plan} busy={planBusy} err={planErr} />
        <p className="hint">구간 사이에는 <b>{isForecast ? 'k' : '1+k'}개 행을 비웁니다</b>(학습 정답이 다음 구간의 값을 쓰지 않게). 검증 구간은 조기종료(언제 학습을 멈출지)에만 씁니다 — 검증을 쓰지 않는 모델(Ridge·ARIMA·SARIMA·VECM)은 그 구간도 학습에 써서, <b>모든 모델의 정보 마감일이 같습니다</b>. 실제로 쓴 기간은 결과의 periods 에 기록됩니다.
          {valMode === 'walkforward' && ' 모델마다 필요한 과거 길이가 달라 첫 평가일이 달라질 수 있으니, 비교할 실험들은 평가 시작일을 같게 정하세요.'}</p>
        {common && tlTo && common.end < tlTo && <div className="verdict warn">{common.endBy} 의 값이 {common.end} 이후로는 갱신되지 않습니다. 그 뒤 기간은 마지막 값이 그대로 이어진 낡은 입력으로 학습·평가됩니다 — 데이터 사용 종료를 {common.end} 로 두는 것을 권합니다.</div>}
        {isForecast && <p className="hint">예측 대상의 데이터가 봉인 시작일 이후에 충분히 있는지 확인하세요(예: F11은 2025-03까지 → 봉인 2025-01-01이면 3개월뿐. 2024-01-01 권장).</p>}
        {isForecast && <p className="hint">외생변수 정보 시점 — 예: 10월 주택가격지수(기준일 10-01)는 10월 하순에야 공개됩니다. <b>기준일 t</b>는 10-01 까지 공개된 거시지표만 쓰고(8월 CPI까지), <b>공개 시점</b>은 지수가 실제 나오는 날(10-01+lag)까지 공개된 지표를 씁니다(9월 CPI까지). 둘 다 누출은 없고, 후자가 실제로 예측을 하는 날의 정보와 같습니다. 결과에 어떤 쪽인지 기록됩니다.</p>}
        {valMode === 'split' && <p className="hint">1회 분할은 테스트 구간이 하나라 운이 섞이기 쉽습니다. 같은 설정을 워크포워드로도 돌려 비교하세요.</p>}
        {leak && <div className="verdict warn">bfill/interp 는 아직 발표되지 않은 값을 과거로 끌어옵니다. 결과에 누출 경고가 붙고, 비교 대상에서 제외하는 것이 원칙입니다.</div>}
      </div>

      <div className="row" style={{ alignItems: 'center' }}>
        <button className="primary" disabled={busy || levelsNoVars || (!useTechnical && selected.length === 0 && !isRule)} title={levelsNoVars ? 'VECM 은 입력 변수를 1개 이상 골라야 합니다' : undefined} onClick={submit}>{busy ? '제출 중…' : '실험 실행'}</button>
        <span className="hint">실행 장치: {meta.device}. LSTM·BIVA 는 GPU 가 없으면 매우 느립니다.</span>
      </div>
      {msg && <div className={'verdict ' + (msg.ok ? 'pass' : 'fail')} style={{ marginTop: 12 }}>{msg.text}</div>}
      <details style={{ marginTop: 12 }}><summary className="hint">설정 JSON 보기</summary><pre className="log">{JSON.stringify(config, null, 1)}</pre></details>
    </div>
  )
}


/** 파라미터 칸 머리: 쉬운 이름(크게) + 코드 이름(작게) + 기본값 */
function PcHead({ h, code, def }: { h: { name: string } | null; code: string; def: string }) {
  return <div className="pc-head"><span className="pc-name">{h?.name ?? code}{h && <code>{code}</code>}</span><span className="pc-def">{def}</span></div>
}
/** 무엇인지 + 키우면/줄이면 + 자세히(원래 설명) */
function PcHelp({ h, detail }: { h: { what: string; up?: string; down?: string; choice?: boolean; how?: string } | null; detail: string }) {
  if (!h) return <div className="pc-desc">{detail}</div>
  return <div className="pc-desc">
    {h.what && <div className="pc-what">{h.what}</div>}
    {h.up && <div className="pc-up"><b>{h.choice || !h.down ? '•' : '▲ 키우면'}</b> {h.up}</div>}
    {h.down && <div className="pc-down"><b>{h.choice ? '•' : '▼ 줄이면'}</b> {h.down}</div>}
    {h.how && <div className="pc-how"><b>고르는 법</b> {h.how}</div>}
    <details><summary>자세히</summary>{detail}</details>
  </div>
}
