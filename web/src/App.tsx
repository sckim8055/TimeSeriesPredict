import { useEffect, useState } from 'react'
import { api, type Meta, type Mode } from './api'
import ExperimentForm from './ExperimentForm'
import Jobs from './Jobs'
import Results from './Results'
import Variables from './Variables'
import Help from './Help'
import { Logs } from './Logs'

type Page = 'new' | 'jobs' | 'results' | 'vars' | 'help' | 'logs'

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [page, setPage] = useState<Page>('new')
  const [mode, setMode] = useState<Mode>('trading')
  const [openJob, setOpenJob] = useState<string | null>(null)
  const [openResult, setOpenResult] = useState<string | null>(null)

  const load = () => api.meta().then(setMeta).catch(e => setErr(e.message))
  useEffect(() => { load() }, [])

  if (err) return <div className="main"><h2>서버에 연결할 수 없습니다</h2><p className="lead">프로젝트 루트에서 <code>python -m uvicorn api.main:app --port 8000</code> 을 먼저 실행하세요. ({err})</p></div>
  if (!meta) return <div className="main empty">불러오는 중…</div>

  return (
    <div className="layout">
      <nav className="nav">
        <h1>TimeSeriesPredict<small>시계열 예측 · 매매 시뮬레이션</small></h1>
        <div className="mode-switch">
          <button className={mode === 'trading' ? 'on' : ''} onClick={() => { setMode('trading'); setOpenResult(null); setPage('new') }}>매매 시뮬레이션</button>
          <button className={mode === 'forecast' ? 'on' : ''} onClick={() => { setMode('forecast'); setOpenResult(null); setPage('new') }}>예측 연구</button>
        </div>
        <p className="mode-desc">{mode === 'trading' ? 'ETF를 사고팔아 돈이 벌리는지 — 체결·비용·B&H 대비' : '시계열이 얼마가 될지 — 오차·naive 대비, 매매 없음'}</p>
        {([['new', '1. 실험 설정'], ['jobs', '2. 실행'], ['results', '3. 결과']] as [Page, string][]).map(([p, l]) =>
          <button key={p} className={page === p ? 'active' : ''} onClick={() => setPage(p)}>{l}</button>)}
        <div className="nav-sep" />
        {([['vars', '변수 사전'], ['help', '도움말'], ['logs', '개발 로그']] as [Page, string][]).map(([p, l]) =>
          <button key={p} className={page === p ? 'active' : ''} onClick={() => setPage(p)}>{l}</button>)}
        <div className="foot">
          {mode === 'trading' ? '매매' : '예측'} 실험 {meta.experiment_counters?.[mode] ?? 0}회 · 봉인 평가 {meta.sealed_evaluations_by_mode?.[mode]?.length ?? 0}회<br />
          데이터 {meta.data_versions.at(-1)?.id ?? '없음'}<br />
          장치 {meta.device}
        </div>
      </nav>
      <main className="main">
        {page === 'new' && <ExperimentForm key={mode} meta={meta} mode={mode} onSubmitted={jobId => { load(); setOpenJob(jobId); setPage('jobs') }} />}
        {page === 'jobs' && <Jobs mode={mode} initialJobId={openJob} onOpenResult={s => { setOpenResult(s); setPage('results') }} onOpenLogs={id => { setOpenJob(id); setPage('logs') }} />}
        {page === 'results' && <Results key={mode} mode={mode} initial={openResult} />}
        {page === 'vars' && <Variables meta={meta} />}
        {page === 'help' && <Help />}
        {page === 'logs' && <Logs initialRun={openJob} />}
      </main>
    </div>
  )
}
