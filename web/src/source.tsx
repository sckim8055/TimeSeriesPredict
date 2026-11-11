import type { Variable } from './api'

// 데이터가 어디서 왔는지: 모양 = 종류(■ 파일 / ● API / ◆ 파생), 색 = 기관. 색만으로 구분하지 않도록 모양과 글자를 함께 쓴다.
export const SOURCE_INFO: Record<string, { kind: 'file' | 'api' | 'derived'; label: string; org: string; cls: string }> = {
  XLSX: { kind: 'file', label: '엑셀', org: '엑셀 업로드 파일', cls: 'src-xlsx' },
  ECOS: { kind: 'api', label: 'ECOS', org: '한국은행 ECOS API', cls: 'src-ecos' },
  FRED: { kind: 'api', label: 'FRED', org: '미 세인트루이스 연준 FRED API', cls: 'src-fred' },
  YF: { kind: 'api', label: 'Yahoo', org: 'Yahoo Finance', cls: 'src-yf' },
  KRX: { kind: 'api', label: 'KRX', org: '한국거래소 정보데이터시스템(pykrx)', cls: 'src-krx' },
  DATAGO: { kind: 'api', label: '공공데이터', org: '공공데이터포털(금융위원회)', cls: 'src-datago' },
  DERIVED: { kind: 'derived', label: '파생', org: '다른 변수로 계산', cls: 'src-derived' },
}
export const KIND_SYMBOL = { file: '■', api: '●', derived: '◆' } as const
export const KIND_LABEL = { file: '파일(엑셀 업로드)', api: 'API 자동 수집', derived: '파생(계산)' } as const

export function sourceWhere(v: Variable): string {
  const si = SOURCE_INFO[v.source] ?? { kind: 'api', label: v.source, org: v.source, cls: '' }
  const st = v.stored
  let where = ''
  if (v.source === 'XLSX') {
    const [sheet, col] = String(v.source_code || '').split('!')
    where = `${st?.upload ?? 'data/variable_master.xlsx'} · 시트 ${sheet ?? '?'} · 열 ${col ?? '?'}`
  } else where = `${si.org} · 코드 ${v.source_code || '-'}`
  const span = st ? `\n저장: ${st.file}\n기간: ${st.first} ~ ${st.last} (${st.n.toLocaleString()}개)` : '\n데이터 없음(아직 수집되지 않음)'
  return `${KIND_LABEL[si.kind]} — ${where}${span}`
}

export function SourceBadge({ v }: { v: Variable }) {
  const si = SOURCE_INFO[v.source] ?? { kind: 'api' as const, label: v.source, org: v.source, cls: '' }
  return <span className={'badge srcb ' + si.cls} title={sourceWhere(v)}>{KIND_SYMBOL[si.kind]} {si.label}</span>
}
