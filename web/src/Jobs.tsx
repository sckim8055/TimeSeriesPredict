import { useEffect, useState } from 'react'
import { api, type Job } from './api'

const STATUS: Record<string, string> = { queued: '대기', running: '실행 중', done: '완료', failed: '실패', cancelled: '취소됨', interrupted: '중단됨(서버 재시작)' }

export default function Jobs({ onOpenResult, onOpenLogs, initialJobId, mode }: { onOpenResult: (stem: string) => void; onOpenLogs?: (jobId: string) => void; initialJobId?: string | null; mode?: string }) {
  void mode
  const [jobs, setJobs] = useState<Job[]>([])
  const [sel, setSel] = useState<Job | null>(null)

  async function refresh() {
    const js = await api.jobs(); setJobs(js)
    if (sel) setSel(await api.job(sel.id))
  }
  useEffect(() => { refresh(); const t = setInterval(refresh, 3000); return () => clearInterval(t) }, [sel?.id])
  useEffect(() => { if (initialJobId) api.job(initialJobId).then(setSel).catch(() => {}) }, [initialJobId])

  const stem = (j: Job) => j.result_path ? j.result_path.replace(/\\/g, '/').split('/').pop()!.replace(/\.json$/, '') : null

  return (
    <div>
      <h2>2. 실행</h2>
      <p className="lead">제출한 실험이 순서대로 실행됩니다(한 번에 하나, GPU·CPU 경합 방지). 3초마다 갱신되고, 끝나면 "결과" 버튼이 생깁니다.</p>
      <div className="split">
        <div className="panel" style={{ padding: 0, overflow: 'auto', maxHeight: '75vh' }}>
          {jobs.length === 0 ? <div className="empty">아직 작업이 없습니다. "새 실험"에서 실행하세요.</div> :
            <table>
              <thead><tr><th>제출</th><th>구분</th><th>이름</th><th>모델</th><th>상태</th><th>소요</th><th></th></tr></thead>
              <tbody>
                {jobs.map(j => (
                  <tr key={j.id} className={sel?.id === j.id ? 'sel' : ''} onClick={async () => setSel(await api.job(j.id))}>
                    <td>{j.submitted_at.slice(5, 16).replace('T', ' ')}</td>
                    <td><span className="badge muted">{(j as any).mode === 'forecast' ? '예측' : '매매'}</span></td>
                    <td>{j.name}{j.sealed && <span className="badge fail">봉인</span>}</td>
                    <td>{j.model}</td>
                    <td className={'status-' + j.status}>{STATUS[j.status] ?? j.status}</td>
                    <td className="num">{j.started_at && j.finished_at ? Math.round((Date.parse(j.finished_at) - Date.parse(j.started_at)) / 1000) + 's' : ''}</td>
                    <td>{j.status === 'done' && stem(j) && <button onClick={e => { e.stopPropagation(); onOpenResult(stem(j)!) }}>결과</button>}{onOpenLogs && <button style={{ marginLeft: 4 }} title="이 작업의 단계별 개발 로그" onClick={e => { e.stopPropagation(); onOpenLogs(j.id) }}>로그</button>}
                      {(j.status === 'queued' || j.status === 'running') && <button onClick={e => { e.stopPropagation(); api.cancel(j.id).then(refresh) }}>취소</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>}
        </div>
        <div className="panel">
          {sel ? <>
            <h3>{sel.name} <span className={'status-' + sel.status}>{STATUS[sel.status] ?? sel.status}</span></h3>
            <pre className="log">{sel.log_tail || '(로그 없음)'}</pre>
          </> : <div className="empty">작업을 고르면 로그가 보입니다.</div>}
        </div>
      </div>
    </div>
  )
}
