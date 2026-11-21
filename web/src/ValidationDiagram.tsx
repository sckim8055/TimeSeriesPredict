/** 검증 방식 그림. 숫자는 지금 설정 값. 길이는 비율이 아니라 모양을 보여주는 도식(학습 756 : 테스트 21 을 그대로 그리면 테스트가 안 보임).
 *  hoverKey: 설정 칸(train_len, test_len, val_frac, embargo, seq_len, val_start, test_start, test_end, sealed_from)과 서로 강조. */
type Props = {
  mode: 'walkforward' | 'split'; trainLen: number; testLen: number; valFrac: number; emb: number; seqLen: number | null
  usesVal: boolean; unit: string; dates: { start?: string; valStart?: string; testStart?: string; testEnd?: string; sealed?: string; end?: string }
  hoverKey: string | null; onHover: (k: string | null) => void
}

export function ValidationDiagram(p: Props) {
  const hot = (k: string) => (p.hoverKey === k ? ' hot' : '')
  const seg = (k: string, cls: string, x: number, w: number, y: number, h: number, label: string, title: string) =>
    <g key={k + x + y} className={'vd-seg ' + cls + hot(k)} onMouseEnter={() => p.onHover(k)} onMouseLeave={() => p.onHover(null)}>
      <rect x={x} y={y} width={Math.max(w, 2)} height={h} rx={3} /><title>{title}</title>
      {w > 34 && <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle">{label}</text>}
    </g>
  const W = 1000
  if (p.mode === 'walkforward') {
    const rows = 4, y0 = 30, rh = 26, gap = 8, L = 70
    const trW = 460, gapW = 22, teW = 90, step = 118
    // 한 폴드의 창을 자르는 순서(엔진과 같음, 빈 칸·정답 없는 행이 없을 때): 창 → 끝 비움 → (창 모델) 앞 seq_len−1 → 검증 int(n×비율) → 사이 비움 → 학습
    const warm = p.seqLen != null ? p.seqLen - 1 : 0
    const base = p.trainLen - p.emb - warm
    const nVal = p.usesVal ? Math.floor(base * p.valFrac) : 0
    const nFit = base - (p.usesVal ? nVal + p.emb : 0)
    const valW = p.usesVal ? Math.max(40, trW * p.valFrac) : 0
    const out = []
    for (let i = 0; i < rows; i++) {
      const x = L + i * step, y = y0 + i * (rh + gap)
      const fit = trW - valW - (p.usesVal ? gapW : 0)
      out.push(seg('train_len', 'vd-train', x, fit, y, rh, `학습 ${nFit}`, `학습 창 ${p.trainLen}${p.unit} 중 가중치를 맞추는 데 쓰는 ${nFit}개`))
      if (p.usesVal) {
        out.push(seg('embargo', 'vd-gap', x + fit, gapW, y, rh, '', `학습과 검증 사이 비움 ${p.emb}행 — 학습 정답이 검증 입력 기간과 겹치지 않게`))
        out.push(seg('val_frac', 'vd-val', x + fit + gapW, valW, y, rh, `검증 ${nVal}`, `검증(조기종료용) = 학습 창 뒤쪽 ${Math.round(p.valFrac * 100)}%`))
      }
      out.push(seg('embargo', 'vd-gap', x + trW, gapW, y, rh, '', `학습 창과 테스트 사이 비움 ${p.emb}행`))
      out.push(seg('test_len', 'vd-test', x + trW + gapW, teW, y, rh, `테스트 ${p.testLen}`, `이 폴드에서 예측·채점하는 ${p.testLen}${p.unit}. 다음 폴드는 ${p.testLen}${p.unit} 밀어서 다시 학습`))
      out.push(<text key={'f' + i} x={x - 4} y={y + rh / 2 + 4} textAnchor="end" className="vd-lab">{i < rows - 1 ? `폴드 ${i + 1}` : '…'}</text>)
    }
    const sx = L + trW + gapW
    return <svg className="diagram vdiag" viewBox={`0 0 ${W} ${y0 + rows * (rh + gap) + 66}`} role="img" aria-label="워크포워드 검증 도식">
      <text x={20} y={16} className="vd-head">워크포워드: 학습 창 {p.trainLen}{p.unit}를 {p.testLen}{p.unit}씩 밀며 매번 다시 학습 → 테스트 부분만 이어 붙여 채점{p.dates.testStart ? ` (평가 ${p.dates.testStart}~${p.dates.testEnd || ''})` : ''}</text>
      {out}
      {p.seqLen != null && <g className={'vd-win' + hot('seq_len')} onMouseEnter={() => p.onHover('seq_len')} onMouseLeave={() => p.onHover(null)}>
        <path d={`M${sx - 70},${y0 - 4} L${sx - 70},${y0 - 9} L${sx + 12},${y0 - 9} L${sx + 12},${y0 - 4}`} />
        <text x={sx - 29} y={y0 - 12} textAnchor="middle">창 {p.seqLen}</text>
        <title>딥러닝 입력 창: 결정일 포함 과거 {p.seqLen}행. 테스트 첫 창은 바로 앞 실제 행으로 채운다(빈 칸을 만들지 않음)</title>
      </g>}
      <text x={20} y={y0 + rows * (rh + gap) + 18} className="vd-note">
        {p.usesVal ? '검증을 쓰는 모델(딥러닝·LightGBM): 학습 창 뒤쪽을 조기종료용으로 떼고, 그 앞을 비웁니다.' : '이 모델은 검증을 쓰지 않아 학습 창 전체로 학습합니다.'} 회색 = 비움({p.emb}행). 막대 길이는 비율이 아닌 모양입니다.
      </text>
      <text x={20} y={y0 + rows * (rh + gap) + 38} className="vd-note">
        창 {p.trainLen} = {warm ? `앞 ${warm}(창을 채울 과거·입력 전용) + ` : ''}학습 {nFit}{p.usesVal ? ` + 비움 ${p.emb} + 검증 ${nVal}` : ''} + 비움 {p.emb} → 다음 테스트 {p.testLen}. 정답 없는 행·빈 칸이 있으면 더 줄어듭니다(아래 표가 실제 개수).
      </text>
    </svg>
  }
  const parts: { k: string; cls: string; w: number; label: string; t: string; d?: string }[] = [
    { k: 'start', cls: 'vd-train', w: 380, label: '학습', t: '학습 구간', d: p.dates.start || '처음' },
    ...(p.usesVal ? [{ k: 'embargo', cls: 'vd-gap', w: 24, label: '', t: `비움 ${p.emb}행` }, { k: 'val_start', cls: 'vd-val', w: 140, label: '검증', t: '조기종료용 검증 구간', d: p.dates.valStart || `학습 뒤 ${Math.round(p.valFrac * 100)}%` }] : []),
    { k: 'embargo', cls: 'vd-gap', w: 24, label: '', t: `비움 ${p.emb}행` },
    { k: 'test_start', cls: 'vd-test', w: 250, label: '테스트', t: '한 번만 채점하는 구간', d: p.dates.testStart || '' },
    { k: 'sealed_from', cls: 'vd-seal', w: 130, label: '봉인', t: '최종 확인용 — 개발 중 보지 않음', d: p.dates.sealed || '없음' },
  ]
  let x = 20
  const y = 34, h = 34
  return <svg className="diagram vdiag" viewBox={`0 0 ${W} 120`} role="img" aria-label="1회 분할 검증 도식">
    <text x={20} y={16} className="vd-head">1회 분할: 학습 → (검증) → 테스트를 한 번 · {p.usesVal ? '검증은 조기종료에만 쓰고' : '이 모델은 검증을 쓰지 않아 검증 구간까지 학습에 씀'}</text>
    {parts.map((q, i) => { const g = seg(q.k, q.cls, x, q.w, y, h, q.label, q.t); const lx = x; x += q.w + 2
      return <g key={i}>{g}{q.d && <text x={lx + 2} y={y + h + 16} className="vd-lab">{q.d}</text>}</g> })}
    <text x={20} y={y + h + 40} className="vd-note">구간 사이 회색 = 비움 {p.emb}행. 테스트 종료 {p.dates.testEnd || (p.dates.sealed ? '봉인 전날' : '데이터 끝')}{p.seqLen != null ? ` · 창 ${p.seqLen}행: 각 구간 첫 창은 앞의 실제 행으로 채움` : ''}</text>
  </svg>
}
