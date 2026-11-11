import { useEffect, useMemo, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { api } from './api'
import { ModelFlow } from './ModelFlow'

type Doc = { id: string; title: string }

// 도움말의 모델 흐름도: 문서의 <!--flow:모델--> 자리에, 실험 화면과 같은 ModelFlow 그림을 그린다.
// 숫자(모양·가중치 수 등)는 아래 기본 설정으로 서버(/api/flow)가 실제 모델을 만들어 잰 값 — 글과 그림과 코드가 같은 출처.
const BASE_CFG = { mode: 'trading', ticker: '091160', frequency: 'D', use_technical: true, variable_ids: [] as string[], target: { kind: 'return', k: 1 } }
const HELP_FLOW_CFG: Record<string, any> = {
  vecm: { ...BASE_CFG, use_technical: false, variable_ids: ['SOX', 'USDKRW'] },  // VECM 은 변수가 1개 이상 있어야 정의됨
  sox_gap_fade: { ...BASE_CFG, variable_ids: ['SOX'] },
  biva: { preset: 'repro_biva_orig' },  // 원본 노트북 설정(F11 월간, 34채널, 창 6)
}
const CFG_TEXT: Record<string, string> = {
  vecm: '반도체 ETF(091160) 일간 + 변수 SOX·USDKRW',
  sox_gap_fade: '반도체 ETF(091160) 일간 + 변수 SOX',
  biva: '원본 노트북 재현 설정(configs/repro_biva_orig.json: F11 월간, 34채널, 창 6)',
}

function HelpFlow({ model }: { model: string }) {
  const [f, setF] = useState<any>(null)
  const [err, setErr] = useState<string | null>(null)
  const [hp, setHp] = useState<string | null>(null)
  useEffect(() => { api.flow({ ...(HELP_FLOW_CFG[model] ?? BASE_CFG), model }).then(setF).catch(e => setErr(e.message)) }, [model])
  return <div className="help-flow">
    <p className="hint">그림 — 기본값과 {CFG_TEXT[model] ?? '반도체 ETF(091160) 일간 + 기술지표 블록'} 으로 실제 모델을 만들어 잰 값입니다. 실험 화면에서는 고른 변수·파라미터로 다시 그려집니다.</p>
    <ModelFlow model={model} flow={f} loading={!f && !err} error={err} value={k => (k === '@seq_len' ? f?.seq_len : f?.params?.[k])} hoverParam={hp} onHoverParam={setHp} inHelp config={{ ...(HELP_FLOW_CFG[model] ?? BASE_CFG), model }} />
  </div>
}

export default function Help({ initialDoc }: { initialDoc?: string | null }) {
  const [docs, setDocs] = useState<Doc[]>([])
  const [cur, setCur] = useState<string | null>(initialDoc ?? null)
  const [md, setMd] = useState('')
  const [q, setQ] = useState('')

  useEffect(() => { fetch('/api/guide').then(r => r.json()).then((d: Doc[]) => { setDocs(d); if (!cur && d.length) setCur(d[0].id) }) }, [])
  useEffect(() => { if (cur) fetch(`/api/guide/${encodeURIComponent(cur)}`).then(r => r.json()).then(d => { setMd(d.markdown); window.scrollTo(0, 0) }) }, [cur])

  // 문서를 <!--flow:모델--> 표시로 잘라 [글, 그림, 글, 그림 …] 순서로 그린다(정적 문서용 안내 문구는 화면에서 뺌)
  const parts = useMemo(() => {
    const src = md.replace(/<!--flow:([a-z_]+)-->\n\*\(그림:[^\n]*\)\*/g, '<!--flow:$1-->')
    const re = q.trim() ? new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi') : null
    return src.split(/<!--flow:([a-z_]+)-->/).map((seg, i) => {
      if (i % 2 === 1) return { flow: seg }
      const text = re ? seg.replace(re, m => `==${m}==`) : seg
      const raw = marked.parse(text, { gfm: true, breaks: false }) as string
      return { html: DOMPurify.sanitize(raw.replace(/==([^=]+)==/g, '<mark>$1</mark>')) }
    })
  }, [md, q])

  const headings = useMemo(() => Array.from(md.matchAll(/^## (.+)$/gm)).map(m => m[1]), [md])

  return (
    <div>
      <h2>도움말</h2>
      <p className="lead">사용법·모델·변수·알고리즘·용어를 한 곳에. 🟢 쉽게 말하면 / 🔵 정확히 말하면 두 층으로 적었습니다. 파라미터 표와 변수 표는 코드와 마스터 엑셀에서 자동 생성됩니다(<code>docs/GUIDE.md</code>에 합본).</p>
      <div className="help-layout">
        <aside className="help-nav">
          <input placeholder="이 문서에서 찾기" value={q} onChange={e => setQ(e.target.value)} style={{ width: '100%', marginBottom: 8 }} />
          {docs.map(d => <button key={d.id} className={'help-link' + (cur === d.id ? ' active' : '')} onClick={() => setCur(d.id)}>{d.title}</button>)}
          {headings.length > 0 && <div className="help-toc">{headings.map(h => <a key={h} href={'#' + h.replace(/\s+/g, '-')} onClick={e => { e.preventDefault(); const el = Array.from(document.querySelectorAll('.help-body h2')).find(x => x.textContent === h); el?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}>{h}</a>)}</div>}
        </aside>
        <article className="help-body panel">{parts.map((p, i) => p.flow ? <HelpFlow key={i} model={p.flow} /> : <div key={i} dangerouslySetInnerHTML={{ __html: p.html! }} />)}</article>
      </div>
    </div>
  )
}
