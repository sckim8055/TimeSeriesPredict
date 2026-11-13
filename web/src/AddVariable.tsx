import { useEffect, useState } from 'react'
import { api } from './api'

/** 화면에서 변수 추가 + 데이터 가져오기 (2026-11-12).
 *  순서를 하나로: ① 출처를 고르고 코드를 넣는다 → ② [받아 보기]로 실제 값이 오는지 본다 → ③ 이름을 넣고 [변수 추가] → ④ [데이터 가져오기].
 *  접속 키는 여기서 받지 않는다(.env 파일에만). 지원하는 출처만 고를 수 있다. */
export default function AddVariable({ onChanged }: { onChanged: () => void }) {
  const [sources, setSources] = useState<any[]>([])
  const [source, setSource] = useState('YF')
  const [code, setCode] = useState('')
  const [chk, setChk] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [id, setId] = useState('')
  const [name, setName] = useState('')
  const [freq, setFreq] = useState('D')
  const [group, setGroup] = useState('')
  const [transform, setTransform] = useState('pct')
  const [lag, setLag] = useState('')
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [col, setCol] = useState<any>({ state: 'none' })
  const [tid, setTid] = useState('')
  const [tname, setTname] = useState('')

  useEffect(() => { api.varSources().then(setSources).catch(() => {}); api.collectStatus().then(setCol).catch(() => {}) }, [])
  useEffect(() => {
    if (col.state !== 'running') return
    const t = setInterval(() => api.collectStatus().then(c => { setCol(c); if (c.state === 'done') onChanged() }).catch(() => {}), 2000)
    return () => clearInterval(t)
  }, [col.state])

  const src = sources.find(s => s.source === source)
  const check = async () => { setBusy(true); setMsg(null); try { setChk(await api.varCheck(source, code)) } catch (e: any) { setChk({ ok: false, error: e.message }) } setBusy(false) }
  const add = async () => {
    setBusy(true); setMsg(null)
    try {
      const r = await api.varAdd({ id, name, source, source_code: code, freq, group, transform, lag_days: lag === '' ? null : Number(lag) })
      setMsg({ ok: true, text: r.note }); setId(''); setName(''); setCode(''); setChk(null); onChanged()
    } catch (e: any) { setMsg({ ok: false, text: e.message }) }
    setBusy(false)
  }
  const tadd = async () => {
    setBusy(true); setMsg(null)
    try {
      const r = await api.tickerAdd({ id: tid.trim(), name: tname })
      const c = r.check
      setMsg({ ok: true, text: `${r.added} 추가(${r.ticker.kind === 'etf' ? 'ETF' : '주식'} — ${r.kind_basis}) — 야후에서 ${c.rows}일(${c.first} ~ ${c.last}) 받아 확인했습니다. 최근 종가 ${c.sample.map((x: any) => `${x.date}: ${x.close}`).join(' · ')}. ${r.note}` })
      setTid(''); setTname(''); onChanged()
    } catch (e: any) { setMsg({ ok: false, text: e.message }) }
    setBusy(false)
  }
  const collect = async () => { setMsg(null); try { setCol(await api.collectStart()) } catch (e: any) { setMsg({ ok: false, text: e.message }) } }

  return <details className="panel" style={{ marginBottom: 16 }}>
    <summary><b>변수 · 종목 추가 · 데이터 가져오기</b> <span className="hint">— 인터넷 출처(야후·FRED·한국은행)의 자료를 코드만 넣어 추가합니다</span></summary>
    <div className="grid" style={{ marginTop: 12 }}>
      <div className="field"><label>① 출처</label>
        <select value={source} onChange={e => { setSource(e.target.value); setChk(null); setFreq(e.target.value === 'YF' ? 'D' : freq) }}>
          {sources.map(s => <option key={s.source} value={s.source}>{s.label}</option>)}
        </select>
        {src && !src.key_present && <span className="hint" style={{ color: 'var(--warn)' }}>이 출처는 접속 키가 필요합니다 — 프로젝트 폴더의 .env 파일에 {src.key_env} 를 넣고 서버를 다시 시작하세요</span>}
      </div>
      <div className="field"><label>② 코드 <span className="hint">{src?.code_help}</span></label>
        <input value={code} onChange={e => { setCode(e.target.value); setChk(null) }} placeholder={src?.example} />
        <span className="hint">예: {src?.example}</span>
      </div>
      <div className="field"><label>&nbsp;</label><button onClick={check} disabled={busy || !code.trim()}>받아 보기</button></div>
    </div>
    {chk && !chk.ok && <p className="hint" style={{ color: 'var(--bad)' }}>{chk.error}</p>}
    {chk?.ok && <>
      <p className="hint">받았습니다: {chk.rows}줄 ({chk.first} ~ {chk.last}, 최근 3년). 최근 값 — {chk.sample.map((x: any) => `${x.date}: ${x.value}`).join(' · ')}</p>
      <div className="grid">
        <div className="field"><label>③ 변수 이름(id) <span className="hint">영문·숫자·밑줄</span></label><input value={id} onChange={e => setId(e.target.value)} placeholder="예: SOX" /></div>
        <div className="field"><label>설명</label><input value={name} onChange={e => setName(e.target.value)} placeholder="예: 필라델피아 반도체 지수" /></div>
        <div className="field"><label>주기</label><select value={freq} onChange={e => setFreq(e.target.value)}>{(src?.freqs ?? ['D']).map((f: string) => <option key={f} value={f}>{({ D: '일', W: '주', M: '월', Q: '분기' } as any)[f]}</option>)}</select></div>
        <div className="field"><label>모델에 넣는 꼴</label><select value={transform} onChange={e => setTransform(e.target.value)}>
          <option value="pct">변화율 (가격·지수·환율 — 보통 이것)</option><option value="diff">차이 (금리·비율처럼 % 단위인 값)</option><option value="none">값 그대로</option></select></div>
        <div className="field"><label>묶음</label><input value={group} onChange={e => setGroup(e.target.value)} placeholder="예: C" /></div>
        <div className="field"><label>발표시차(일) <span className="hint">모르면 비워 두세요</span></label><input value={lag} onChange={e => setLag(e.target.value.replace(/[^0-9]/g, ''))} placeholder="비우면 기본값(미확인 표시)" /></div>
      </div>
      <button className="primary" onClick={add} disabled={busy || !id.trim()}>변수 추가</button>
    </>}
    <hr />
    <p><b>종목 추가</b> <span className="hint">— 실험의 '종목 묶어 학습'에 쓸 다른 종목(한국거래소 6자리 코드, 코스피). 가격은 야후에서 받습니다.</span></p>
    <div className="grid">
      <div className="field"><label>종목 코드</label><input value={tid} onChange={e => setTid(e.target.value.replace(/[^0-9]/g, '').slice(0, 6))} placeholder="예: 005930" /></div>
      <div className="field"><label>이름</label><input value={tname} onChange={e => setTname(e.target.value)} placeholder="예: 삼성전자" /></div>
      <div className="field"><label>&nbsp;</label><button className="primary" onClick={tadd} disabled={busy || tid.length !== 6}>{busy ? '확인하는 중…' : '종목 추가'}</button></div>
    </div>
    <p className="hint">[종목 추가]를 누르면 야후에서 그 종목의 가격이 실제로 받아지는지 확인한 뒤에만 목록에 넣습니다(받아지지 않으면 이유를 보여 주고 넣지 않음). ETF 인지 주식인지는 공식 ETF 시세에 그 코드가 있는지로 자동으로 정합니다.</p>
    {msg && <p className="hint" style={{ color: msg.ok ? 'var(--good)' : 'var(--bad)' }}>{msg.text}</p>}
    <hr />
    <div className="toolbar">
      <button onClick={collect} disabled={col.state === 'running'}>④ 데이터 가져오기</button>
      <span className="hint">{col.state === 'running' ? '가져오는 중입니다… (몇 분 걸립니다. 이 화면을 떠나도 계속됩니다)' : col.state === 'done' ? col.message : '변수 사전의 모든 출처에서 자료를 받아 새 데이터 버전을 만듭니다. 실험이 도는 중에는 할 수 없습니다.'}</span>
    </div>
    {col.log_tail?.length > 0 && <pre className="log" style={{ maxHeight: 220, overflow: 'auto' }}>{col.log_tail.join('\n')}</pre>}
  </details>
}
