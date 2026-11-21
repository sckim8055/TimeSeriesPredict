/** 모델별 알고리즘·데이터 흐름 그림. 모델마다 그 모델에 맞는 모양으로 그린다(공통 양식 없음).
 *  숫자(모양 [B×창×변수], 가중치 수, 나무·시차·행렬 크기)는 서버 /api/flow 가 지금 설정으로 실제 모델을 만들어 잰 값.
 *  색: 파랑 = 데이터, 회색 = 계산, 노랑 = 거르는 단계(무엇을 남기고 무엇을 버리나), 초록 = 출력, 점선 = 선택/학습 때만.
 *  파라미터 글자(밑줄 없는 파란 글씨)에 마우스를 올리면 아래 파라미터 칸이 강조된다. */
import { FormulaGuide, ModelEasy, RidgeEasy } from './EasyFlow'
import { helpOf } from './paramHelp'
import { Children, createContext, isValidElement, useContext, type ReactNode } from 'react'

const HP = createContext<string | null>(null)
/** 아래 파라미터 칸으로 이동해 강조(칸 id = pc-<키>) */
export function gotoParam(k: string) {
  const el = document.getElementById('pc-' + k); if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 1200)
  ;(el.querySelector('input, select') as HTMLElement | null)?.focus({ preventScroll: true })
}

type F = any
type Ctx = { f: F; v: (k: string) => any; hp: string | null; onHp: (k: string | null) => void }
type Tone = 'data' | 'core' | 'filter' | 'out' | 'loss' | 'opt'

/** 모양 표기: 첫 숫자(배치)를 B 로. 중첩(출력 튜플)이면 첫 원소. */
export function fs(a: any): string {
  if (!a) return '?'
  if (Array.isArray(a) && Array.isArray(a[0])) return fs(a[0])
  if (!Array.isArray(a)) return '?'
  return '[' + a.map((n: number, i: number) => (i === 0 ? 'B' : n)).join('×') + ']'
}
const num = (n: number) => (n == null ? '?' : n.toLocaleString())
const short = (v: any) => (typeof v === 'boolean' ? (v ? '예' : '아니오') : v === '' || v == null ? '—' : String(v))

function Box({ x, y, w, h, t, s = [], tone = 'core', children }: { x: number; y: number; w: number; h: number; t: string; s?: (string | null | false)[]; tone?: Tone; children?: ReactNode }) {
  const hp = useContext(HP)  // 상자 안의 파라미터 칸에 마우스가 있으면 상자 테두리 강조
  const hot = hp != null && Children.toArray(children).some(ch => isValidElement(ch) && (ch.props as any).k === hp)
  return <g className={'mf-box mf-' + tone + (hot ? ' hot' : '')}>
    <rect x={x} y={y} width={w} height={h} rx={5} />
    <text x={x + 8} y={y + 16} className="mf-t">{t}</text>
    {s.filter(Boolean).map((l, i) => <text key={i} x={x + 8} y={y + 31 + i * 13.5} className="mf-s">{l}</text>)}
    {children}
  </g>
}
function Ar({ x1, y1, x2, y2, label, dash, lx, ly }: { x1: number; y1: number; x2: number; y2: number; label?: string; dash?: boolean; lx?: number; ly?: number }) {
  const mx = (x1 + x2) / 2
  const d = Math.abs(y1 - y2) < 2 ? `M${x1},${y1} L${x2},${y2}` : Math.abs(x1 - x2) < 2 ? `M${x1},${y1} L${x2},${y2}` : `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`
  return <g><path d={d} className={'mf-ar' + (dash ? ' dash' : '')} markerEnd="url(#mfa)" />
    {label && <text x={lx ?? mx} y={ly ?? (Math.min(y1, y2) - 5)} textAnchor="middle" className="mf-sh">{label}</text>}</g>
}
function Pc({ c, x, y, k, label }: { c: Ctx; x: number; y: number; k: string; label?: string }) {
  return <text x={x} y={y} className={'mf-p' + (c.hp === k ? ' hot' : '')} onMouseEnter={() => c.onHp(k)} onMouseLeave={() => c.onHp(null)} onClick={() => gotoParam(k)}>
    ✎ {(label ?? k).replace('@', '')} = {short(c.v(k))}<title>{(() => { const h = helpOf(c.f.model, k); return h ? `${h.name}: ${h.what}${h.up && h.down ? (h.choice ? `\n• ${h.up}\n• ${h.down}` : `\n▲ 키우면 ${h.up}\n▼ 줄이면 ${h.down}`) : ''}\n(누르면 아래 칸으로 이동)` : `누르면 아래 '${k.replace('@', '')}' 칸으로 이동` })()}</title></text>
}
function Sh({ x, y, a, anchor = 'start' }: { x: number; y: number; a: any; anchor?: 'start' | 'middle' | 'end' }) {
  return <text x={x} y={y} textAnchor={anchor} className="mf-sh">{typeof a === 'string' ? a : fs(a)}</text>
}
function Svg({ w, h, label, children }: { w: number; h: number; label: string; children: ReactNode }) {
  return <svg className="diagram mflow" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
    <defs><marker id="mfa" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" className="mf-head" /></marker></defs>
    {children}
  </svg>
}
function names(f: F, n = 6) {
  const fs_: string[] = f.input?.features ?? []
  return fs_.slice(0, n).join(', ') + (fs_.length > n ? ` 외 ${fs_.length - n}개` : '')
}
const unitOf = (f: F) => (f.freq === 'M' ? '개월' : f.freq === 'W' ? '주' : f.freq === 'Q' ? '분기' : '거래일')
// 입력 열의 출처: 'A_' = 기술지표 블록(가격으로 계산, 아래 변수 목록에는 없음) / 나머지 = 아래 목록에서 고른 변수
// 한 폴드에서 실제로 학습에 쓰는 개수(서버 fit_plan: 학습 창 − 비움 − 검증, 결측 행 제외 전)
const fitText = (f: F) => { const p = f.input?.fit_plan; if (!p) return `학습 창 ${f.input?.train_len}행`; return `한 폴드 학습 ${p.n_fit}행 = 창 ${f.input?.train_len} − 비움 ${p.embargo}${p.has_val ? ` − 검증 ${p.n_val} − 비움 ${p.embargo}` : ''}${p.uses_val && !p.has_val ? ' (검증 10개 미만 → 검증 없음)' : ''}` }
// 'A_' = 기술지표 블록. 서버가 개수를 안 주면(구버전 서버) 열 이름으로 센다
export const techCount = (f: F) => f.input?.n_technical ?? (f.input?.features ?? []).filter((n: string) => String(n).startsWith('A_')).length
const srcText = (f: F) => { const a = techCount(f), n = f.input?.n_features ?? 0; return `기술지표 ${a} + 고른 변수 ${f.input?.n_selected ?? n - a}` }
const tgtText = (f: F) => `${f.input?.target?.variable ?? '대상'} 의 ${f.input?.target?.k ?? 'k'}기간 뒤 ${({ return: '변화율', logreturn: '로그 변화율', direction: '방향(오를 확률)', event: '사건 확률' } as any)[f.input?.target?.kind] ?? '값'}`

/* ------------------------------------------------------------------ LSTM / GRU */
function RnnFlow({ c, gru }: { c: Ctx; gru?: boolean }) {
  const f = c.f, L = f.seq_len ?? 0, k = f.input?.n_features ?? 0, H = c.v('hidden'), layers = c.v('layers') ?? 1
  const tr = f.trace ?? {}, rnnT = tr[gru ? 'gru' : 'lstm']
  const slots = L <= 5 ? Array.from({ length: L }, (_, i) => i) : [0, 1, -1, L - 2, L - 1]
  const lab = (i: number) => (i < 0 ? '…' : L - 1 - i === 0 ? 't' : `t−${L - 1 - i}`)
  const rows = Math.min(layers, 2), x0 = 240, sw = 108, cw = 88, ch = 46
  const rowY = (r: number) => 158 - r * 70
  const G = gru ? 3 : 4, W = 1280
  return <Svg w={W} h={480} label={`${gru ? 'GRU' : 'LSTM'} 데이터 흐름`}>
    <Box x={16} y={40} w={196} h={168} t="입력 창 X" tone="data" s={[`${L}${unitOf(f)} × 변수 ${k}개`, srcText(f), '결정일 t 까지의 과거만', names(f, 3)]}>
      {Array.from({ length: Math.min(L, 10) }).map((_, i) => Array.from({ length: Math.min(k, 6) }).map((__, j) =>
        <rect key={i + '-' + j} x={28 + i * 17} y={104 + j * 12} width={14} height={9} className="mf-cell" />))}
      <Pc c={c} x={24} y={198} k="@seq_len" label="seq_len" />
    </Box>
    <Sh x={114} y={226} a={rnnT?.in} anchor="middle" />
    <line x1={212} y1={244} x2={x0 + 4 * sw + cw} y2={244} className="mf-bus" />
    <Ar x1={114} y1={208} x2={114} y2={244} />
    {slots.map((i, si) => {
      const x = x0 + si * sw
      if (i < 0) return <text key="dots" x={x + cw / 2} y={rowY(0) + 28} textAnchor="middle" className="mf-t">…</text>
      return <g key={si}>
        <Ar x1={x + cw / 2} y1={244} x2={x + cw / 2} y2={rowY(0) + ch} />
        <text x={x + cw / 2 + 4} y={238} className="mf-sh">x_{lab(i)}</text>
        {Array.from({ length: rows }).map((_, r) => <g key={r}>
          <Box x={x} y={rowY(r)} w={cw} h={ch} t={`${gru ? 'GRU' : 'LSTM'} 셀`} s={[r === 0 ? `시점 ${lab(i)}` : `층 ${r === rows - 1 && layers > 2 ? layers : r + 1}`]} />
          {r > 0 && <Ar x1={x + cw / 2} y1={rowY(r - 1)} x2={x + cw / 2} y2={rowY(r) + ch} />}
          {si < slots.length - 1 && <Ar x1={x + cw} y1={rowY(r) + ch / 2} x2={x + sw} y2={rowY(r) + ch / 2} />}
        </g>)}
      </g>
    })}
    <text x={x0 + sw * 0.9} y={rowY(rows - 1) - 8} className="mf-sh">{gru ? 'h' : 'h, c'} [B×{H}] 를 다음 시점으로 넘김{layers > 2 ? ` · 층 ${layers}개(가운데 생략)` : ''}</text>
    <Pc c={c} x={x0} y={22} k="hidden" label="은닉 크기 hidden" /><Pc c={c} x={x0 + 200} y={22} k="layers" label="층 수 layers" />
    {(() => { const xr = x0 + 4 * sw + cw; const yy = rowY(rows - 1) + ch / 2; return <g>
      <Ar x1={xr} y1={yy} x2={xr + 40} y2={yy} />
      <Box x={xr + 40} y={yy - 40} w={150} h={80} t="마지막 시점만" tone="filter" s={['t 의 h 만 남기고', `앞 ${L - 1}개 시점 출력은 버림`, fs(rnnT?.out) + ' → ' + fs(tr.drop?.in)]} />
      <Ar x1={xr + 190} y1={yy} x2={xr + 215} y2={yy} />
      <Box x={xr + 215} y={yy - 40} w={124} h={80} t="Dropout" tone="filter" s={['학습 때만 은닉 값', '일부를 0 으로 끔', '(예측 때는 통과)']}><Pc c={c} x={xr + 223} y={yy + 34} k="dropout" /></Box>
      <Ar x1={xr + 339} y1={yy} x2={xr + 362} y2={yy} />
      <Box x={xr + 362} y={yy - 40} w={118} h={80} t="Linear" s={[`${H} → 1`, `가중치 ${num(tr.fc?.params)}`, fs(tr.fc?.out)]} />
      <Ar x1={xr + 421} y1={yy + 40} x2={xr + 421} y2={yy + 70} />
      <Box x={xr + 280} y={yy + 70} w={200} h={52} t="예측 ŷ" tone="out" s={[tgtText(f), '회귀: 학습 구간 평균·표준편차로 되돌림 · 분류: sigmoid 확률']} />
    </g> })()}
    {/* 셀 하나 확대 */}
    <Box x={16} y={268} w={W - 32} h={196} t={`셀 하나 안에서 (시점 τ) — 게이트 = 0~1 사이 값으로 '얼마나 통과시킬지' 정하는 거름 장치. 가중치 W_ih ${fs([0, ...(f.weights?.weight_ih_l0 ?? [])]).replace('[B×', '[')} = ${G}개 게이트 × ${H}, W_hh ${fs([0, ...(f.weights?.weight_hh_l0 ?? [])]).replace('[B×', '[')}`} tone="opt" />
    {(gru ? [
      ['재설정 게이트 r', 'r = σ(W_r·[h_{τ−1}, x_τ])', '새 후보를 만들 때', '지난 기억 h_{τ−1} 중 얼마를 볼지', '0 이면 과거 무시'],
      ['후보 n', 'n = tanh(W_n·x_τ + r⊙(U_n·h_{τ−1}))', '이번 입력 x_τ 와 (걸러진)', '과거로 만든 새 기억 후보', ''],
      ['갱신 게이트 z', 'z = σ(W_z·[h_{τ−1}, x_τ])', '지난 기억과 새 후보를', '섞는 비율', '1 이면 과거 그대로 유지'],
      ['새 은닉 h_τ', 'h_τ = (1−z)⊙n + z⊙h_{τ−1}', `→ 다음 시점·윗층으로 [B×${H}]`, '', ''],
    ] : [
      ['잊기 게이트 f', 'f = σ(W_f·[h_{τ−1}, x_τ] + b_f)', '지난 기억 c_{τ−1} 중', '얼마를 남길지', '0 = 버림, 1 = 유지'],
      ['입력 게이트 i · 후보 g', 'i = σ(…), g = tanh(…)', '이번 입력으로 만든 새 정보 g 중', '얼마를 기억에 쓸지(i)', ''],
      ['기억 갱신 c_τ', 'c_τ = f⊙c_{τ−1} + i⊙g', '걸러 남긴 과거 + 걸러 받은 새 정보', '(장기 기억 줄)', ''],
      ['출력 게이트 o', 'o = σ(…),  h_τ = o⊙tanh(c_τ)', '기억 중 얼마를', `내보낼지 → h_τ [B×${H}]`, ''],
    ]).map((g, i) => <Box key={i} x={30 + i * 308} y={296} w={292} h={110} t={g[0]} tone={i === (gru ? 1 : 2) || (gru && i === 3) ? 'core' : 'filter'} s={[g[1], g[2], g[3], g[4]]} />)}
    <text x={30} y={428} className="mf-s">σ = 시그모이드(0~1), ⊙ = 같은 자리끼리 곱. 게이트는 변수가 아니라 은닉 단위 {H}개 각각에 걸린다 — 어느 단위가 무엇을 뜻하는지는 학습이 정하므로 "금리를 거른다"처럼 단정할 수 없다.</text>
    <text x={30} y={446} className="mf-s">전체 가중치 {num(tr._total?.params)}개. 학습: Adam, 기울기 크기 1.0 으로 자름, 검증 손실이 가장 낮은 epoch 의 가중치 사용 — 검증이 없으면(검증 10개 미만) 마지막 epoch 가중치 (</text>
    <Pc c={c} x={470} y={446} k="epochs" /><Pc c={c} x={580} y={446} k="lr" /><Pc c={c} x={690} y={446} k="batch_size" /><Pc c={c} x={830} y={446} k="patience" />
  </Svg>
}

/* ------------------------------------------------------------------ BIVA */
function BivaFlow({ c }: { c: Ctx }) {
  const f = c.f, tr = f.trace ?? {}, L = f.seq_len, k = f.input?.n_features, lat = f.latent
  const o = (n: string) => fs(tr[n]?.out), pn = (n: string) => num(tr[n]?.params)
  const brits = c.v('use_brits'), enc = [1, 2, 3, 4].map(i => tr[`LSTM_VAE.encoder_lstm_${i}`]?.out?.[0]?.[2])
  const ch = (n: string) => tr[n]?.out?.[1]
  return <Svg w={1280} h={650} label="BIVA 데이터 흐름">
    <Box x={16} y={20} w={170} h={136} t="입력 창 X" tone="data" s={[`${L}${unitOf(f)} × 변수 ${k}개`, srcText(f), `빈 칸 ${f.nan_cells ?? '?'}개(전체 표)`, '늦게 시작한 변수 ' + Object.keys(f.late_start ?? {}).length + '개', fs(tr.BRITS?.in)]}>
      <Pc c={c} x={24} y={128} k="@seq_len" label="창 길이" /><Pc c={c} x={24} y={144} k="input_scaler" label="스케일" /></Box>
    <Ar x1={186} y1={86} x2={206} y2={86} />
    <Box x={206} y={46} w={92} h={80} t="RIN" tone="opt" s={['평균 빼고', '표준편차로', '나눔(+배율)']}><Pc c={c} x={214} y={116} k="RIN" /></Box>
    <Ar x1={298} y1={86} x2={318} y2={86} />
    <Box x={318} y={8} w={660} h={158} t={`Stage 1 결측값 삽입 — BRITS ${brits ? '' : '(꺼짐 — 입력을 그대로 넘기고 빈 칸은 학습 평균)'} · 가중치 ${pn('BRITS')}`} tone={brits ? 'filter' : 'opt'}
      s={['앞→뒤 RITS 와 뒤→앞 RITS(각 LSTMCell)가 창 안의 값만 읽고, 두 결과를 평균. 시점 τ 마다:',
        `① 과거 기억으로 추정 x̂ = sigmoid(hist_reg: 은닉 ${c.v('rnn_hid_size')} → 변수 ${k})`,
        '② 같은 시점의 다른 변수로 추정 ẑ = sigmoid(feat_reg, 자기 자신은 0 으로 막음)',
        '③ 둘을 섞음 ĉ = |0.5·ẑ + 0.5·x̂| — 고정 비율(원본은 학습 비율 β 를 계산만 하고 쓰지 않음)',
        '④ 거름: 관측된 칸은 원래 값, 빈 칸만 ĉ  →  m·x + (1−m)·ĉ',
        '⑤ 오래 빈 변수일수록 기억을 줄임 γ = exp(−max(0, W·δ + b)) (원본 δ 는 이웃 변수 열을 참조)',
        `출력: 빈 칸 없는 창 ${fs(tr.BRITS?.out?.imputations)} + 보간 손실 = 관측 칸 L1 오차 × imputed_weight + 0.1·|앞−뒤|`]}>
      <Pc c={c} x={330} y={156} k="use_brits" /><Pc c={c} x={470} y={156} k="rnn_hid_size" /><Pc c={c} x={640} y={156} k="imputed_weight" /></Box>
    <Ar x1={978} y1={90} x2={1010} y2={90} />
    <Box x={1010} y={14} w={254} h={152} t="Stage 2 추세·주기 분해" tone="filter" s={['이동평균(창 양끝은 끝값으로 늘림)', '추세 = 이동평균  → 느린 흐름만 통과', '주기 = 원래 − 추세 → 빠른 출렁임', `둘 다 ${o('decomposition')}`, '변수마다 따로 계산']}>
      <Pc c={c} x={1018} y={156} k="moving_avg" /></Box>
    {/* 추세 가지 */}
    <path d="M1080,166 L1080,198 L110,198 L110,224" className="mf-ar" markerEnd="url(#mfa)" /><text x={600} y={193} className="mf-sh">추세 [B×L×k] → 변수·시간 축을 바꿔 [B×k×L]</text>
    <text x={16} y={214} className="mf-t">추세 → 선형 예측 (차원감소 없이 바로 Stage 4 로)</text>
    {[['Conv1d_Trend_1', '변수 축 섞기 ①'], ['Conv1d_Trend_2', '변수 축 섞기 ②'], ['Conv1d_Trend_3', '변수 축 섞기 ③']].map(([n, t], i) =>
      <g key={n}><Box x={16 + i * 186} y={226} w={168} h={78} t={t} s={[`채널 ${tr[n]?.in?.[1]} → ${ch(n)}  ${['(변수 수÷4)', '(변수 수÷16)', '(→1)'][i]}`, `${fs(tr[n]?.in)} → ${o(n)}`, `가중치 ${pn(n)}`]}>
        {i === 0 && <Pc c={c} x={24 + i * 186} y={296} k="conv1d" />}{i === 1 && <Pc c={c} x={24 + i * 186} y={296} k="conv_kernal" label="커널" />}</Box>
        <Ar x1={184 + i * 186} y1={265} x2={202 + i * 186} y2={265} /></g>)}
    <Box x={574} y={226} w={168} h={78} t="Linear (시간 축)" s={[`${L}${unitOf(f)} → 1 (창 길이)`, `${fs(tr.Linear_Trend?.in)} → ${o('Linear_Trend')}`, `가중치 ${pn('Linear_Trend')}`]} />
    <text x={16} y={318} className="mf-s">변수 {k}개를 Conv1d 3단으로 1개 흐름으로 합친 뒤(같은 시점끼리만 섞음), 시간 {L}칸을 가중 평균해 다음 값 하나로.</text>
    {/* 계절 가지 */}
    <path d="M1073,166 L1073,330 L290,330 L290,340 L140,340 L140,348" className="mf-ar" markerEnd="url(#mfa)" /><text x={900} y={325} className="mf-sh">계절 [B×L×k]</text>
    <text x={16} y={344} className="mf-t" style={{ fontSize: 12 }}>Stage 3 주기정보 차원감소 (LSTM-VAE)</text>
    <Box x={16} y={350} w={250} h={92} t="인코더 LSTM 4단" s={[`채널 ${k} → ${enc.join(' → ')}`, '(변수 수 → ÷2 → ÷4 → 잠재 크기, 원본 규칙)', `가중치 ${num([1, 2, 3, 4].reduce((a, i) => a + (tr[`LSTM_VAE.encoder_lstm_${i}`]?.params ?? 0), 0))}`]}>
      <Pc c={c} x={24} y={432} k="vae_latent_size" label="잠재 크기" /></Box>
    <Ar x1={266} y1={396} x2={290} y2={396} />
    <Box x={290} y={350} w={230} h={92} t="잠재요인 z (거름)" tone="filter" s={['μ, log σ² = ReLU(Linear(·)) ≥ 0', 'z = μ + ε·σ, σ ≥ 1 (ε 무작위)', `${k}개 변수 주기 성분 → ${lat}개 요인`, '(논문: DFM 의 요인 추출)']} />
    <Ar x1={520} y1={396} x2={544} y2={396} />
    <Box x={544} y={350} w={250} h={92} t="Conv1d 3단 + Linear" s={[`[인코더 출력 ${lat}, z ${lat}] → 채널 ${2 * lat} (잠재×2)`, `채널 ${2 * lat} → ${ch('Conv1d_Seasonal_1')} → ${ch('Conv1d_Seasonal_2')} → ${ch('Conv1d_Seasonal_3')}`, `시간 ${L}${unitOf(f)} → 1:  ${o('Linear_Seasonal')}`, `가중치 ${num(['Conv1d_Seasonal_1', 'Conv1d_Seasonal_2', 'Conv1d_Seasonal_3', 'Linear_Seasonal'].reduce((a, n) => a + (tr[n]?.params ?? 0), 0))}`]} />
    <Ar x1={405} y1={442} x2={405} y2={470} dash />
    <Box x={16} y={470} w={504} h={62} t="디코더 LSTM 4단 → 재구성 (예측에는 쓰지 않음)" tone="opt" s={[`[인코더 출력, z] → 계절 성분 ${fs(tr['LSTM_VAE.output_fc']?.out)} 복원 · VAE 손실 = 0.5·${lat}·MSE + 0.5·ΣKLD_j`, '(원본이 0.5·MSE + 0.5·KLD 를 잠재 차원마다 더함) · z 만으로 복원하는 구조는 아님']} />
    {/* 합치기 */}
    <Ar x1={742} y1={265} x2={880} y2={330} /><Ar x1={794} y1={396} x2={880} y2={350} />
    <Box x={880} y={290} w={180} h={96} t="Stage 4 합치기" s={['ŷ = α·계절 + (1−α)·추세', 'α 학습(시작 1, 제한 없음)', '아니오면 단순 합']}><Pc c={c} x={888} y={376} k="combination" /></Box>
    <Ar x1={1060} y1={338} x2={1086} y2={338} />
    <Box x={1086} y={290} w={178} h={96} t="예측 ŷ" tone="out" s={[tgtText(f), '대상 스케일 되돌림']}><Pc c={c} x={1094} y={376} k="target_scale" /></Box>
    <Box x={16} y={552} w={1248} h={84} t={`학습 (원본 exp_main.train) · 전체 가중치 ${num(tr._total?.params)}개`} tone="loss"
      s={['손실 = VAE 손실 + BRITS 보간 손실 + 예측 MSE  (가중치 없는 합).  매 epoch: 학습(섞기·나머지 버림) → 검증 손실 → 조기종료 판정 → 학습률 조정',
        '끝나면 검증 손실이 가장 낮았던 가중치로 되돌림.  예측할 때도 z 는 무작위로 뽑힘(원본 그대로) — 같은 시드여야 같은 값']}>
      <Pc c={c} x={24} y={624} k="learning_rate" /><Pc c={c} x={170} y={624} k="lradj" /><Pc c={c} x={260} y={624} k="batch_size" /><Pc c={c} x={380} y={624} k="train_epochs" /><Pc c={c} x={530} y={624} k="patience" /><Pc c={c} x={650} y={624} k="init_checkpoint" />
    </Box>
  </Svg>
}

/* ------------------------------------------------------------------ Ridge */
function RidgeFlow({ c }: { c: Ctx }) {
  const f = c.f, feats: string[] = f.input?.features ?? [], k = feats.length, show = feats.slice(0, 9)
  const yy = (i: number) => 46 + i * 24
  return <Svg w={1280} h={330} label="Ridge 데이터 흐름">
    <text x={16} y={24} className="mf-t">입력 표: 결정일 t 하루 = 한 행, 입력 열 {k}개 = {srcText(f)} ({fitText(f)})</text>
    {show.map((n, i) => <g key={n}><Box x={16} y={yy(i) - 14} w={200} h={20} t={n} tone="data" /><Ar x1={216} y1={yy(i) - 4} x2={262} y2={yy(i) - 4} /></g>)}
    {k > show.length && <text x={24} y={yy(show.length) + 2} className="mf-s">… 외 {k - show.length}개</text>}
    <Box x={262} y={32} w={190} h={210} t="표준화" s={['z_j = (x_j − 평균_j) / 표준편차_j', '평균·표준편차는 학습 창으로만', '→ 단위가 다른 변수(%, 원, 지수)를', '같은 눈금으로', '', `[N×${k}] → [N×${k}]`]} />
    {show.map((_, i) => <Ar key={i} x1={452} y1={yy(i) - 4} x2={560} y2={140} />)}
    <Box x={560} y={92} w={260} h={96} t="가중합" s={['ŷ = b + Σ_j w_j · z_j', `계수 w ${k}개 + 절편 1개`, '변수 하나가 1 표준편차 오르면', 'ŷ 가 w_j 만큼 움직인다']} />
    <Box x={560} y={210} w={260} h={80} t="L2 벌점 (거름) — alpha 는 설정값" tone="filter" s={['회귀: 오차² 합 + alpha·Σ w_j² 최소화(닫힌 해)', '분류: 로그손실 + (alpha/2)·Σ w_j² (C = 1/alpha, 반복 최적화)', '→ 큰 계수를 0 쪽으로 줄임(0 이 되지는 않음)']}><Pc c={c} x={568} y={282} k="alpha" /></Box>
    <Ar x1={690} y1={210} x2={690} y2={188} />
    <Ar x1={820} y1={140} x2={866} y2={140} />
    <Box x={866} y={100} w={220} h={80} t="예측 ŷ" tone="out" s={[tgtText(f), '방향·사건이면 로지스틱:', '확률 = 1 / (1 + e^(−가중합))']} />
    <text x={866} y={210} className="mf-s">학습 후 결과의 '변수 중요도' = |w_j|</text>
    <text x={866} y={232} className="mf-s">w_j = 데이터로 학습되는 값(변수마다 1개)</text>
    <text x={866} y={252} className="mf-s">alpha = 사람이 정하는 설정값(변수 아님)</text>
  </Svg>
}

/* ------------------------------------------------------------------ LightGBM */
function Tree({ x, y, depth, leaves }: { x: number; y: number; depth: number; leaves: number }) {
  const d = Number.isFinite(depth) ? Math.max(1, Math.min(depth < 0 ? 3 : depth, 3)) : 2, nodes: ReactNode[] = []
  leaves = Number.isFinite(leaves) ? leaves : 4
  let leafCount = 0
  const rec = (cx: number, cy: number, lvl: number, span: number) => {
    if (lvl === d || leafCount + (1 << (d - lvl)) > Math.max(leaves, 2) && lvl > 0) { leafCount++; nodes.push(<rect key={cx + '-' + cy} x={cx - 9} y={cy - 6} width={18} height={12} rx={2} className="mf-leaf" />); return }
    nodes.push(<circle key={cx + '-' + cy} cx={cx} cy={cy} r={6} className="mf-node" />)
    for (const s of [-1, 1]) { const nx = cx + s * span / 2, ny = cy + 30; nodes.push(<line key={'l' + cx + cy + s} x1={cx} y1={cy + 6} x2={nx} y2={ny - 6} className="mf-ar" />); rec(nx, ny, lvl + 1, span / 2) }
  }
  rec(x, y, 0, 90)
  return <g>{nodes}</g>
}
function LgbmFlow({ c }: { c: Ctx }) {
  const f = c.f, t = f.trees ?? {}, k = f.input?.n_features
  const tx = [530, 750, 970]
  return <Svg w={1280} h={340} label="LightGBM 데이터 흐름">
    <Box x={16} y={60} w={180} h={110} t="입력 표" tone="data" s={[`학습 ${t.n_train}행 × 변수 ${k}개`, srcText(f), '결정일 t 하루 = 한 행', names(f, 2)]} />
    <Ar x1={196} y1={115} x2={220} y2={115} />
    <Box x={220} y={40} w={210} h={150} t="무작위 추출 (거름)" tone="filter" s={['나무를 만들 때마다', `행 ${t.rows_per_tree}개(${Math.round(c.v('subsample') * 100)}%),`, `변수 ${t.cols_per_tree}개(${Math.round(c.v('colsample_bytree') * 100)}%)만 씀`, '→ 나무끼리 덜 닮게(과적합 줄임)']}>
      <Pc c={c} x={228} y={164} k="subsample" /><Pc c={c} x={228} y={180} k="colsample_bytree" /></Box>
    <Ar x1={430} y1={115} x2={tx[0] - 96} y2={115} />
    {tx.map((x, i) => <g key={i}>
      <text x={x} y={34} textAnchor="middle" className="mf-t">{i === 2 ? `나무 ${t.n}` : `나무 ${i + 1}`}</text>
      <Tree x={x} y={56} depth={t.depth} leaves={t.leaves} />
      {i < 2 && <Ar x1={x + 96} y1={146} x2={tx[i + 1] - 96} y2={146} label={i === 1 ? '… 계속 …' : '남은 오차 →'} ly={138} />}
    </g>)}
    <text x={tx[0] - 90} y={196} className="mf-s">● = 질문 "변수 j ≤ 기준값?" (학습이 고름) · ■ = 잎(그 칸 표본들의 평균 보정값)</text>
    <Box x={16} y={210} w={560} h={110} t="나무 하나의 모양 (거름)" tone="filter" s={['각 마디에서 오차를 가장 많이 줄이는 변수·기준값 하나로 둘로 나눔', `잎 수 ≤ ${t.leaves}, 깊이 ≤ ${t.depth < 0 ? '제한 없음' : t.depth}, 잎마다 표본 ≥ ${t.min_leaf} — 못 지키는 분할은 하지 않음`, '잎 값에 L2 벌점(reg_lambda)으로 극단값을 줄임']}>
      <Pc c={c} x={24} y={308} k="num_leaves" /><Pc c={c} x={160} y={308} k="max_depth" /><Pc c={c} x={290} y={308} k="min_child_samples" /><Pc c={c} x={460} y={308} k="reg_lambda" /></Box>
    <Box x={600} y={210} w={330} h={110} t="부스팅 (차례로 더하기)" s={['ŷ = Σ (학습률 × 나무_m 의 잎 값)', '나무 m 은 앞 나무들이 못 맞힌 오차를 맞히도록 학습', f.input?.fit_plan?.has_val ? `검증 ${f.input.fit_plan.n_val}개로 50그루 연속 개선 없으면 멈춤 → 가장 좋았던 개수` : '검증 11개 미만 → 조기종료 없이 n_estimators 그루 모두 사용']}>
      <Pc c={c} x={608} y={308} k="n_estimators" /><Pc c={c} x={760} y={308} k="learning_rate" /></Box>
    <Ar x1={1062} y1={110} x2={1080} y2={110} />
    <Box x={1080} y={70} w={184} h={84} t="예측 ŷ" tone="out" s={[tgtText(f), '결과의 중요도 = 분할 이득 합']} />
  </Svg>
}

/* ------------------------------------------------------------------ ARIMA / SARIMA */
function SeriesFlow({ c, seasonal }: { c: Ctx; seasonal?: boolean }) {
  const f = c.f, ar: number[] = f.lags?.ar ?? [], ma: number[] = f.lags?.ma ?? [], kk = f.input?.target?.k ?? 1
  const maxLag = Math.max(1, ...ar, ...ma), x1 = 980, unit = Math.min(70, 860 / maxLag), X = (lag: number) => x1 - lag * unit, ax = 196
  const fx = (j: number) => x1 + j * 54
  return <Svg w={1280} h={330} label={`${seasonal ? 'SARIMA' : 'ARIMA'} 데이터 흐름`}>
    <Box x={16} y={14} w={420} h={64} t="입력: 대상 하나의 1기간 변화율 r_τ" tone="data" s={[`선택한 변수·기술지표는 쓰지 않음(단변량) · ${fitText(f)}`, `${seasonal ? '계절 차분 D=1 이면 r_τ − r_{τ−s} 로 바꾼 뒤 맞힘' : '차분 d=0: 변화율이 이미 정상(평균 회귀) 가정'}`]} />
    <line x1={60} y1={ax} x2={fx(kk) + 20} y2={ax} className="mf-bus" />
    {Array.from(new Set([0, ...ar.map(a => a - 1), ...ma.map(a => a - 1)])).map(l => <g key={l}><line x1={X(l)} y1={ax - 5} x2={X(l)} y2={ax + 5} className="mf-ar" /><text x={X(l)} y={ax + 20} textAnchor="middle" className="mf-sh">{l === 0 ? 't' : `t−${l}`}</text></g>)}
    {Array.from({ length: Math.min(kk, 4) }).map((_, j) => <g key={j}><circle cx={fx(j + 1)} cy={ax} r={7} className="mf-outdot" /><text x={fx(j + 1)} y={ax + 20} textAnchor="middle" className="mf-sh">t+{j + 1}</text></g>)}
    {ar.map(l => <path key={'a' + l} d={`M${X(l - 1)},${ax - 6} Q${(X(l - 1) + fx(1)) / 2},${ax - 40 - l * (70 / maxLag)} ${fx(1) - 4},${ax - 8}`} className="mf-arcA" markerEnd="url(#mfa)" />)}
    {ma.map(l => <path key={'m' + l} d={`M${X(l - 1)},${ax + 26} Q${(X(l - 1) + fx(1)) / 2},${ax + 60 + l * (40 / maxLag)} ${fx(1) - 4},${ax + 8}`} className="mf-arcM" markerEnd="url(#mfa)" />)}
    <text x={60} y={108} className="mf-legA">위 화살표 = AR: 지난 변화율 값 r_{'{t+1−ℓ}'} 에 계수 φ 를 곱해 더함 (ℓ = {ar.join(', ') || '없음'})</text>
    <text x={60} y={300} className="mf-legM">아래 화살표 = MA: 지난 예측 오차 ε 에 계수 θ 를 곱해 더함 (ℓ = {ma.join(', ') || '없음'}) — 갑작스러운 충격이 몇 기간 남는지</text>
    <Pc c={c} x={500} y={30} k="p" /><Pc c={c} x={560} y={30} k="q" /><Pc c={c} x={620} y={30} k="d" />
    {seasonal ? <><Pc c={c} x={680} y={30} k="P" /><Pc c={c} x={740} y={30} k="Q" /><Pc c={c} x={800} y={30} k="D" /><Pc c={c} x={860} y={30} k="s" />
      <text x={500} y={52} className="mf-s">계절 항 = s 기간 전(작년 같은 달) 값·오차. 곱 항(s+1 등)은 단기·계절 효과가 겹친 것</text></>
      : <><Pc c={c} x={680} y={30} k="trend" label="상수항" /><text x={500} y={52} className="mf-s">c: 상수항(절편) 포함 — 평균 변화율 = c ÷ (1 − Σφ)</text></>}
    <Pc c={c} x={500} y={74} k="enforce_stationarity" label="정상성 제약" /><Pc c={c} x={650} y={74} k="enforce_invertibility" label="가역성 제약" />
    <text x={800} y={74} className="mf-s">예: AR·MA 다항식의 근이 모두 |z| &gt; 1 인 범위에서만 계수를 찾음(9-2절)</text>
    <Box x={1040} y={240} w={224} h={76} t="예측 ŷ" tone="out" s={[`1기간씩 ${kk}번 이어 예측(재귀)`, `→ 복리로 묶어 ${kk}기간 변화율`, `추정 계수 ${f.lags?.n_coef}개 + 오차 분산${seasonal ? ' (곱 항은 φ·Φ 의 곱)' : ''}`]} />
  </Svg>
}

/* ------------------------------------------------------------------ HMM (2026-11-15) */
function HmmFlow({ c }: { c: Ctx }) {
  const f = c.f, kk = f.input?.target?.k ?? 1
  return <Svg w={1280} h={190} label="HMM 데이터 흐름">
    <Box x={16} y={20} w={300} h={150} t="입력: 대상 하나의 1기간 변화율 r_τ" tone="data" s={['선택한 변수·기술지표는 쓰지 않음(단변량)', fitText(f), '결정일까지의 값만']} />
    <Ar x1={316} y1={95} x2={346} y2={95} />
    <Box x={346} y={20} w={300} h={150} t="숨은 상태 (학습 창, Baum–Welch)" tone="filter" s={['상태마다 평균 μ·흩어짐 σ', '상태가 바뀔 확률 A', '출발점 2개 중 로그우도 최대(난수 없음)']}>
      <Pc c={c} x={356} y={140} k="n_states" label="상태 수" /></Box>
    <Ar x1={646} y1={95} x2={676} y2={95} />
    <Box x={676} y={20} w={300} h={150} t="지금 상태 확률 (앞방향만)" tone="core" s={['하루씩: 어제 확률 × A → 오늘 변화율로 다시 맞춤', '결정일 뒤 자료는 쓰지 않음', 'p_t(j) = P(상태 j | 결정일까지)']} />
    <Ar x1={976} y1={95} x2={1006} y2={95} />
    <Box x={1006} y={20} w={258} h={150} t="예측 ŷ" tone="out" s={['앞날 상태 확률 = p_t · A^h', '예측 r̂ = 상태 확률로 가중한 평균', `→ 복리로 묶어 ${kk}기간 변화율`]} />
  </Svg>
}

/* ------------------------------------------------------------------ VECM */
function VecmFlow({ c }: { c: Ctx }) {
  const f = c.f, v = f.vecm ?? {}, K = v.K, r = v.r, p = v.p
  return <Svg w={1280} h={380} label="VECM 데이터 흐름">
    <Box x={16} y={20} w={190} h={150} t={`수준값 y(t−1) [${K}]`} tone="data" s={[...(v.vars ?? []).slice(0, 6).map((n: string, i: number) => `${i + 1}. ${n}${i === 0 ? ' (예측 대상)' : ''}`), '변환 없음 · 발표시차 적용']}>
      <Pc c={c} x={24} y={162} k="log_levels" label="로그" /></Box>
    <Ar x1={206} y1={95} x2={236} y2={95} />
    <Box x={236} y={20} w={230} h={150} t={`β′ 장기 균형식 [${r}×${K}] (거름)`} tone="filter" s={[`${K}개 수준값의 조합 중`, '오래 보면 일정하게 유지되는', `조합 ${r}개만 남김 → 균형 이탈 e = β′y`, 'Johansen 고유값 분해로 추정']}>
      <Pc c={c} x={244} y={146} k="coint_rank" label="r" /><Pc c={c} x={320} y={146} k="rank_test" label="검정" /><Pc c={c} x={244} y={162} k="deterministic" label="상수" /></Box>
    <Ar x1={466} y1={95} x2={496} y2={95} label={`e [${r}]`} />
    <Box x={496} y={20} w={230} h={150} t={`α 되돌아오는 속도 [${K}×${r}]`} s={['균형에서 벗어난 만큼', '다음 변화를 반대쪽으로 당김', `변수마다 따로: α·e [${K}]`, 'α 가 0 이면 그 변수는', '균형을 맞추지 않음(약외생)']} />
    <Box x={16} y={200} w={450} h={110} t={`지난 변화 Δy(t−1) … Δy(t−${p}) [${K}×${p}]  →  Γ 단기 동학 [${v.gamma?.join('×')}]`} s={['지난 p 기간의 모든 변수의 변화가 이번 변화에 주는 영향', '(VAR 과 같은 부분)']}>
      <Pc c={c} x={24} y={260} k="k_ar_diff" label="p" /><Pc c={c} x={100} y={260} k="lag_select" label="p 고르기" /></Box>
    <Box x={496} y={200} w={230} h={110} t="상수·계절" tone="opt" s={['co: 균형식 밖 상수(꾸준한 추세)', 'ci: 균형식 안 상수', '계절 더미: 달마다 다른 평균 변화']}>
      <Pc c={c} x={504} y={290} k="seasons" label="계절" /></Box>
    <Ar x1={726} y1={95} x2={780} y2={160} /><Ar x1={466} y1={255} x2={496} y2={255} /><Ar x1={726} y1={255} x2={780} y2={190} />
    <Box x={780} y={120} w={200} h={110} t={`Δŷ(t) [${K}]`} s={['= α·e + Γ·Δy + 상수·계절', '모든 변수의 이번 변화']} />
    <Ar x1={980} y1={175} x2={1010} y2={175} />
    <Box x={1010} y={120} w={254} h={110} t="예측 ŷ" tone="out" s={['수준 = 앞 수준 + Δŷ', `k(${f.input?.target?.k})번 이어 계산(재귀)`, '대상(1번)만 결과로 → 변화율로', `폴드마다 학습 창 ${v.n_train}개로 적합`]} />
    {K < 2 && <text x={500} y={190} className="mf-legM">입력 변수를 1개 이상 고르세요 — 대상 하나로는 장기 균형식이 없습니다</text>}
    <text x={16} y={340} className="mf-s">계수 개수: α {K}×{r} + β {K}×{r} + Γ {v.gamma?.join('×')} = {typeof r === 'number' ? (K * r * 2 + K * K * p).toLocaleString() : '?'}개 (+상수·계절). 학습 행이 이보다 충분히 많아야 함 — 부족하면 실행 전에 멈춤.</text>
    <text x={16} y={358} className="mf-s">r=0(자동): 학습 창마다 Johansen 검정 · 0 이 나오면 장기 관계 없음 → α·β 없이 차분 VAR 로 대체.</text>
  </Svg>
}

/* ------------------------------------------------------------------ 규칙 모델 */
function RuleFlow({ c }: { c: Ctx }) {
  const m = c.f.model as string, tk = c.f.input?.target?.variable ?? '종목'
  const R: Record<string, [string, string[], string[]][]> = {
    buy_and_hold: [['종가', [`${tk} 매일 종가`], []], ['평가 첫날에 산다', ['이후 아무것도 하지 않음'], []], ['끝까지 보유', ['모든 전략의 기준선'], []]],
    ma_cross: [['종가', [`${tk} 종가`], []], ['이동평균 두 개 (거름)', ['짧은 창 = 빠른 흐름, 긴 창 = 느린 흐름', '창이 길수록 잔물결을 더 지움'], ['fast', 'slow']], ['비교', ['짧은 평균 > 긴 평균 → 보유', '아니면 현금'], []]],
    momentum: [['종가', [`${tk} 종가`], []], ['과거 수익률', ['lookback 일 전 대비 변화'], ['lookback']], ['부호로 거름', ['> 0 → 보유, ≤ 0 → 현금'], []]],
    rsi_rule: [['종가', [`${tk} 종가`], []], ['RSI', ['오른 날 평균 ÷ (오른 날 + 내린 날 평균)', '0~100'], ['window']], ['문턱으로 거름', ['< lower → 매수', '> upper → 매도'], ['lower', 'upper']]],
    shannon: [['종가', [`${tk} 종가 — 비율이 흘러가는 계산에만`], []], ['목표 비율', ['ETF weight · 현금 1 − weight', '가격을 보지 않음'], ['weight']], ['주기마다 다시 맞춤', ['그 주기 마지막 결정일 → 다음 시가 체결', '그 사이 비율은 가격 따라 흘러감, 비용은 맞출 때만'], ['every']]],
    sox_gap_fade: [['변수 값(t)', ['SOX 등 일간 변화율'], ['column']], ['과거 분위수', ['t 이전 window 일(당일 제외)', '그 중 q 분위 값'], ['window', 'q']], ['규칙으로 거름', ['avoid_top: 급등한 다음날 현금'], ['mode']]],
  }
  const steps = R[m] ?? []
  return <Svg w={1280} h={170} label="규칙 모델 흐름">
    {steps.map(([t, s, ps], i) => <g key={i}>
      <Box x={16 + i * 420} y={20} w={380} h={130} t={t} tone={i === 0 ? 'data' : t.includes('거름') ? 'filter' : i === steps.length - 1 ? 'out' : 'core'} s={s}>
        {ps.map((k, j) => <Pc key={k} c={c} x={24 + j * 120} y={138} k={k} />)}</Box>
      {i < steps.length - 1 && <Ar x1={396 + i * 420} y1={85} x2={436 + i * 420} y2={85} />}
    </g>)}
  </Svg>
}

const FLOW_KEYS: Record<string, string[]> = {
  ridge: ['standardize', 'ridge_sum', 'ridge_fit', 'logistic'], lightgbm_model: ['standardize', 'lgbm'], arima: ['arima', 'roots', 'kalman', 'compound'], sarima: ['arima', 'roots', 'sarima', 'kalman', 'compound'],
  vecm: ['vecm'], lstm: ['standardize', 'lstm_cell', 'rnn_head', 'nn_train'], gru: ['standardize', 'gru_cell', 'rnn_head', 'nn_train'],
  biva: ['minmax', 'brits', 'moving_avg', 'conv1d', 'vae', 'biva_combine', 'nn_train', 'noise'],
  ma_cross: ['rule_ma'], momentum: ['rule_momentum'], rsi_rule: ['rule_rsi'], sox_gap_fade: ['rule_quantile'],
}
// 코드 따라가기 화면이 있는 모델 → 화면 이름(tools/codewalk/*_script.py 가 docs/guide_build/walk/<이름>.html·.mp4 를 만든다). 규칙 모델은 하나(백테스트 포함)
const WALKS: Record<string, string> = { biva: 'biva', lstm: 'lstm', gru: 'gru', arima: 'arima', sarima: 'sarima', ridge: 'ridge', lightgbm_model: 'lgbm', vecm: 'vecm',
  buy_and_hold: 'rule', ma_cross: 'rule', momentum: 'rule', rsi_rule: 'rule', sox_gap_fade: 'rule' }
export function ModelFlow(props: { model: string; flow: F | null; loading: boolean; error: string | null; value: (k: string) => any; hoverParam: string | null; onHoverParam: (k: string | null) => void; inHelp?: boolean; config?: any }) {
  const { flow: f } = props
  if (props.error) return <div className="verdict warn">흐름도를 그리지 못했습니다: {props.error}</div>
  if (!f || f.model !== props.model) return <p className="hint">{props.loading ? '모델 구조를 재는 중…' : ''}</p>
  const c: Ctx = { f, v: props.value, hp: props.hoverParam, onHp: props.onHoverParam }
  const body = f.model === 'lstm' ? <RnnFlow c={c} /> : f.model === 'gru' ? <RnnFlow c={c} gru /> : f.model === 'biva' ? <BivaFlow c={c} />
    : f.model === 'ridge' ? <RidgeFlow c={c} /> : f.model === 'lightgbm_model' ? <LgbmFlow c={c} /> : f.model === 'arima' ? <SeriesFlow c={c} /> : f.model === 'hmm' ? <HmmFlow c={c} />
    : f.model === 'sarima' ? <SeriesFlow c={c} seasonal /> : f.model === 'vecm' ? <VecmFlow c={c} /> : <RuleFlow c={c} />
  return <HP.Provider value={props.hoverParam}><div className={'mflow-wrap' + (props.loading ? ' loading' : '')}>
    <p className="hint mf-legend"><span className="lg lg-data" />데이터 <span className="lg lg-core" />계산 <span className="lg lg-filter" />거르는 단계(무엇을 남기고 버리나) <span className="lg lg-out" />출력 <span className="lg lg-opt" />선택·학습 때만 · 모양 [B×…] 의 B = 배치(한 번에 넣는 창 수). 숫자는 지금 설정으로 실제 모델을 만들어 잰 값.
      {f.input?.n_rows != null && <> · 이 설정의 데이터: {f.input.period?.join(' ~ ')} ({f.input.n_rows}행)</>}</p>
    <p className="hint mf-legend2"><b className="mf-pk">✎ 파란 글자</b> = {props.inHelp ? '실험 화면의 파라미터 칸에서 바꾸는 값(마우스를 올리면 뜻이 보임)' : '아래 칸에서 바꾸는 값(누르면 그 칸으로 이동, 칸에 마우스를 올리면 그림의 해당 상자가 진하게)'} · <b>검은 숫자</b> = 고른 변수 수·창 길이·파라미터로 <b>계산된 결과</b>라 직접 바꾸는 칸이 없음(예: 채널 {f.input?.n_features} → {Math.max(1, Math.floor((f.input?.n_features ?? 0) / 4))} 은 원본 코드의 '변수 수÷4' 규칙) {props.inHelp ? '' : '· 값을 바꾸면 0.5초 뒤 다시 잼'}{props.loading ? ' — 재는 중…' : ''}</p>
    {WALKS[f.model] && <p className="hint mf-walk">🎞 <a href={`/api/guide/walk/${WALKS[f.model]}.html`} target="_blank" rel="noopener">코드 따라가기 열기</a> · <a href={`/api/guide/walk/${WALKS[f.model]}.mp4`} target="_blank" rel="noopener">영상(MP4)</a> — {WALKS[f.model] === 'rule' ? '규칙 신호와 모든 매매 모드 모델이 함께 쓰는 백테스트·성적표·검정이' : '원본 코드가'} 실제로 실행되는 순서: 어느 변수가 어디서 잘리고 합쳐지는지, 실제 값·손 계산 검증·음성 설명(브라우저 한국어 음성). 새 창에서 열립니다.</p>}
    {props.config && <details className="easy-wrap" open><summary>쉽게 보기 — 실제 데이터 하루로 끝까지 따라가기</summary>{f.model === 'ridge' ? <RidgeEasy f={f} config={props.config} manual={props.inHelp} /> : <ModelEasy model={f.model} config={props.config} manual={props.inHelp} />}</details>}
    {props.config ? <details className="easy-wrap" open><summary>수식으로 보기 — 실제 모양·개수</summary>{body}<FormulaGuide keys={FLOW_KEYS[f.model] ?? []} title="그림에 나온 식 읽는 법 — 기호 · 문장으로 읽기 · 작은 예로 직접 풀기" /></details> : body}
    {f.notes?.length > 0 && <div className="verdict warn">{f.notes.join(' / ')}</div>}
  </div></HP.Provider>
}
