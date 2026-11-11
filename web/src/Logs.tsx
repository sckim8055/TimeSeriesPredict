/** 개발 로그 보기 — logs/tsp_YYYYMMDD.log 를 걸러서 본다. 실험 하나를 따라가려면 작업 ID 를 '작업' 칸에 넣는다. */
import { useEffect, useRef, useState } from 'react'
import { api } from './api'

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR']

export function Logs({ initialRun }: { initialRun?: string | null }) {
  const [date, setDate] = useState('')
  const [run, setRun] = useState(initialRun ?? '')
  const [level, setLevel] = useState('DEBUG')
  const [q, setQ] = useState('')
  const [tail, setTail] = useState(400)
  const [auto, setAuto] = useState(true)
  const [data, setData] = useState<{ files: string[]; file: string | null; lines: string[]; total: number } | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const box = useRef<HTMLPreElement>(null)
  useEffect(() => { if (auto && box.current) box.current.scrollTop = box.current.scrollHeight }, [data, auto])
  const load = () => api.logs({ date: date || undefined, run, level, q, tail }).then(d => { setData(d); setErr(null) }).catch(e => setErr(e.message))
  useEffect(() => { load() }, [date, run, level, q, tail])
  useEffect(() => { if (!auto) return; const t = setInterval(load, 3000); return () => clearInterval(t) }, [auto, date, run, level, q, tail])
  const cls = (l: string) => (/ ERROR /.test(l) ? 'lg-err' : / WARNING /.test(l) ? 'lg-warn' : / INFO /.test(l) ? 'lg-info' : 'lg-dbg')
  return <div>
    <h2>개발 로그</h2>
    <p className="lead">값이 어디서 어디로 넘어갔는지 단계별 기록입니다. ▶ = 단계 시작(넘긴 값 요약), ◀ = 끝(걸린 시간), ✖ = 그 단계에서 오류. 한 줄의 [run=…] 는 작업 ID — 실험 하나만 보려면 아래 '작업' 칸에 넣으세요. 파일 위치: <code>logs/tsp_날짜.log</code></p>
    <div className="panel">
      <div className="row">
        <div className="field"><label>날짜 파일</label><select value={date} onChange={e => setDate(e.target.value)}><option value="">최신</option>{data?.files.slice().reverse().map(f => <option key={f} value={f.slice(4, 12)}>{f}</option>)}</select></div>
        <div className="field"><label>작업 ID(일부)</label><input value={run} onChange={e => setRun(e.target.value)} placeholder="예: 20261106_101230" /></div>
        <div className="field"><label>수준 이상</label><select value={level} onChange={e => setLevel(e.target.value)}>{LEVELS.map(l => <option key={l}>{l}</option>)}</select></div>
        <div className="field"><label>포함 문자열</label><input value={q} onChange={e => setQ(e.target.value)} placeholder="예: 변수 F7, 폴드, fit" /></div>
        <div className="field"><label>마지막 줄 수</label><input type="number" value={tail} min={50} max={5000} step={50} onChange={e => setTail(+e.target.value)} /></div>
        <div className="field"><label>&nbsp;</label><label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 13, color: 'var(--ink)' }}><input type="checkbox" checked={auto} onChange={e => setAuto(e.target.checked)} />3초마다 새로고침</label></div>
      </div>
      {err && <div className="verdict fail">{err}</div>}
      <p className="hint">{data?.file ?? '로그 파일 없음'} · 조건에 맞는 줄 {data?.total ?? 0}개 중 마지막 {data?.lines.length ?? 0}개</p>
      <pre className="logview" ref={box}>{data?.lines.map((l, i) => <div key={i} className={cls(l)}>{l}</div>)}</pre>
    </div>
  </div>
}
