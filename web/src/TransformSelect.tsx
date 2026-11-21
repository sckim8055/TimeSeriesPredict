/** 변환 고르기 — 실험 설정·[값 보기] 고치기·변수 추가가 함께 쓴다(2026-11-14).
 *  목록과 설명은 서버 meta.transform_options = engine.experiment.TRANSFORM_OPTIONS 한 곳에서 온다(도움말 20장 4절 표와 같은 것).
 *  지금 값이 목록에 없으면(예: 변수 사전의 pct3) 목록 맨 위에 그 값을 그대로 보여 준다 — 엔진은 pct{n}·diff{n} 을 모두 계산한다. */
export type TransformOption = { value: string; name: string; formula: string; meaning: string; use: string; caution?: string }

export default function TransformSelect({ value, onChange, options, compact = false, className }:
  { value: string; onChange: (v: string) => void; options: TransformOption[]; compact?: boolean; className?: string }) {
  const known = options.find(o => o.value === value)
  const list = known ? options : [{ value, name: '변수 사전의 값', formula: '', meaning: '목록 밖의 값(엔진은 계산합니다)', use: '' } as TransformOption, ...options]
  const title = known ? `${known.name}: ${known.formula} — ${known.meaning}. 맞는 자료: ${known.use}${known.caution ? '. 주의: ' + known.caution : ''}` : '목록 밖의 값'
  return <>
    <select className={className} value={value} title={title} onChange={e => onChange(e.target.value)}>
      {list.map(o => <option key={o.value} value={o.value}>{compact ? o.value : `${o.value} — ${o.name}`}</option>)}
    </select>
    {!compact && known && <span className="hint" style={{ display: 'block' }}>{known.formula} · {known.meaning}. 맞는 자료: {known.use}.{known.caution ? ` 주의: ${known.caution}.` : ''} <span>(도움말 20장 4절)</span></span>}
  </>
}
