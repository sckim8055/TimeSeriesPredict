/** 결과의 '설정 전체와 뜻'(2026-11-20) — 설정 JSON 의 모든 칸을 이 결과의 값·기본값·뜻과 함께 표로.
 *  뜻은 설정 스키마(engine/experiment.py ExperimentConfig 의 Field description, /api/config-guide)와 모델 코드의 PARAMS 설명에서 그대로 가져온다. */
import { useEffect, useState } from 'react'
import { api } from './api'
import { helpOf } from './paramHelp'

type F = { path: string; desc: string; default: any; choices: any[] | null; group: boolean }
let GUIDE: Promise<{ fields: F[] }> | null = null

const show = (v: any) => v === null || v === undefined ? '(비움)' : typeof v === 'object' ? JSON.stringify(v) : String(v)
const get = (o: any, path: string) => path.split('.').reduce((a, k) => (a == null ? undefined : a[k]), o)

export function ConfigTable({ config }: { config: any }) {
  const [g, setG] = useState<F[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    GUIDE ??= fetch('/api/config-guide').then(r => { if (!r.ok) throw new Error(r.statusText); return r.json() })
    GUIDE.then(x => setG(x.fields)).catch(e => { GUIDE = null; setErr(String(e.message ?? e)) })
  }, [])
  if (err) return <p className="hint">설정 설명을 불러오지 못했습니다: {err}</p>
  if (!g) return <p className="hint">설정 설명 불러오는 중…</p>
  return <table className="tbl cfg-table">
    <thead><tr><th>칸</th><th>이 결과의 값</th><th>기본값</th><th>뜻</th></tr></thead>
    <tbody>{g.map(f => {
      if (f.group) return <tr key={f.path} className="cfg-group"><td colSpan={4}><code>{f.path}</code> — {f.desc}</td></tr>
      const v = get(config, f.path)
      const changed = v !== undefined && JSON.stringify(v) !== JSON.stringify(f.default ?? null) && !(v == null && f.default == null)
      return <tr key={f.path}>
        <td><code>{f.path}</code></td>
        <td style={changed ? { fontWeight: 600 } : undefined}>{v === undefined ? <span className="hint">(기록 없음 = 기본값)</span> : show(v)}{changed && <span className="hint"> · 바꿈</span>}</td>
        <td className="hint">{show(f.default)}</td>
        <td>{f.desc}{f.choices && <span className="hint"> [선택: {f.choices.join(' · ')}]</span>}</td>
      </tr>
    })}</tbody>
  </table>
}

export function ParamTable({ modelFile, params }: { modelFile: string; params: Record<string, any> }) {
  const [meta, setMeta] = useState<Record<string, any> | null>(null)
  useEffect(() => { api.meta().then(m => setMeta(m.models.find(x => x.file === modelFile)?.params ?? {})).catch(() => setMeta({})) }, [modelFile])
  const keys = Object.keys(params ?? {})
  if (!keys.length) return <p className="hint">파라미터 없음(규칙 모델 등)</p>
  return <table className="tbl cfg-table">
    <thead><tr><th>파라미터</th><th>실제로 쓰인 값</th><th>기본값</th><th>뜻</th></tr></thead>
    <tbody>{keys.map(k => {
      const p = meta?.[k]; const h = helpOf(modelFile, k)
      const dflt = p ? (p.default_by_freq ? JSON.stringify(p.default_by_freq) : show(p.default)) : ''
      return <tr key={k}><td><code>{k}</code>{h?.name && h.name !== k && <span className="hint"> {h.name}</span>}</td><td>{show(params[k])}</td><td className="hint">{dflt}</td>
        <td>{h?.what || p?.desc || ''}{h?.how && <div className="hint">고르는 법: {h.how}</div>}</td></tr>
    })}</tbody>
  </table>
}

export function FoldTable({ periods }: { periods: any }) {
  const folds: any[] | undefined = periods?.folds
  if (!folds) return <p className="hint">이 결과는 모든 폴드 기록이 생기기 전(2026-11-20 이전)에 만들어져 첫·마지막 폴드만 있습니다. 같은 설정으로 다시 돌리면 모든 폴드가 기록됩니다.</p>
  const d = (x: any) => x ? `${x[0]} ~ ${x[1]}` : '—'
  return <table className="tbl">
    <thead><tr><th>폴드</th><th>학습 (개수)</th><th>검증 (개수)</th><th>테스트 (개수)</th></tr></thead>
    <tbody>{folds.map((f, i) => <tr key={i}><td>{i + 1}</td><td>{d(f.train)} ({f.n_train})</td><td>{f.val ? `${d(f.val)} (${f.n_val})` : '없음'}</td><td>{d(f.test)} ({f.n_test})</td></tr>)}</tbody>
  </table>
}
