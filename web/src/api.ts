export type Mode = 'trading' | 'forecast'
export type Meta = {
  models: { file: string; name: string; input_shape: string; target_types: string[]; params: Record<string, any>; param_groups?: string[] | null; uses_validation?: boolean; doc?: string; seq_len_default?: Record<string, number> | null }[]
  variables: Variable[]; groups: string[]; master_warnings: string[]; lag_defaults: Record<string, number>
  variable_sets: VariableSet[]; forecast_targets: { id: string; name: string; freq: string; group: string; domain: string }[]; experiment_counters: Record<string, number>; sealed_evaluations_by_mode: Record<string, any[]>; data_versions: { id: string; created_utc?: string; tickers: string[]; variables?: number }[]
  tickers: { id: string; name: string }[]; defaults: any; experiment_counter: number; sealed_evaluations: any[]; device: string; transforms: string[]; price_api?: Record<string, any[]>; data_version_collect?: { start: string; end: string | null; created_utc: string | null }
}
export type VariableSet = { id: string; name: string; description: string; variable_ids: string[]; transforms: Record<string, string>; lag_override: Record<string, number>; recommended_for: string[] }
export type Variable = { id: string; name: string; description: string; group: string; domain: string; freq: string; lag_days: number; lag_confirmed: boolean; source: string; transform: string; active: boolean; duplicate_of: string; usable: boolean; has_data: boolean; note: string; source_code: string; api?: any; stored: { file: string; first: string; last: string; n: number; upload: string | null } | null }
export type Job = { id: string; name: string; model: string; status: string; sealed: boolean; submitted_at: string; started_at?: string; finished_at?: string; returncode?: number; result_path?: string; log_tail?: string }
export type ResultRow = Record<string, any>

/** 개발 로그: 브라우저 콘솔(F12)에 요청·응답을 남긴다. 끄려면 콘솔에서 localStorage.tspLog = '0' */
const LOG_ON = (() => { try { return localStorage.getItem('tspLog') !== '0' } catch { return true } })()
async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const t0 = performance.now()
  const method = init?.method ?? 'GET'
  if (LOG_ON && init?.body) console.debug(`[api] → ${method} ${url}`, (() => { try { return JSON.parse(String(init.body)) } catch { return init.body } })())
  const r = await fetch(url, init)
  const ms = Math.round(performance.now() - t0)
  if (!r.ok) {
    let msg = r.statusText; try { msg = (await r.json()).detail ?? msg } catch {}
    console.error(`[api] ✖ ${method} ${url} → ${r.status} (${ms}ms)`, msg)
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  const data = await r.json()
  if (LOG_ON && (method !== 'GET' || !/\/api\/(jobs|logs)/.test(url))) console.debug(`[api] ← ${method} ${url} ${r.status} (${ms}ms)`, data)
  return data
}
export const api = {
  meta: () => j<Meta>('/api/meta'),
  submit: (config: any, sealed = false) => j<{ job: Job; config_hash: string }>('/api/experiments', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ config, sealed }) }),
  jobs: () => j<Job[]>('/api/jobs'),
  explain: (config: any, date?: string, epochs?: number) => j<any>('/api/explain?' + new URLSearchParams({ ...(date ? { date } : {}), ...(epochs ? { epochs: String(epochs) } : {}) }).toString(), { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ config, sealed: false }) }),
  flow: (config: any) => j<any>('/api/flow', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ config, sealed: false }) }),
  logs: (q: { date?: string; tail?: number; run?: string; level?: string; q?: string }) => j<{ files: string[]; file: string | null; lines: string[]; total: number }>('/api/logs?' + new URLSearchParams(Object.entries(q).filter(([, v]) => v != null && v !== '').map(([k, v]) => [k, String(v)])).toString()),
  job: (id: string) => j<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => j<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  results: (mode: Mode = 'trading') => j<ResultRow[]>(`/api/results?mode=${mode}`),
  result: (stem: string) => j<any>(`/api/results/${stem}`),
  series: (stem: string) => j<any>(`/api/results/${stem}/series`),
  saveSet: (set: VariableSet) => j<any>('/api/variable_sets', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(set) }),
  deleteSet: (id: string) => j<any>(`/api/variable_sets/${id}`, { method: 'DELETE' }),
  ensemble: (stems: string[], method: 'mean' | 'median') => j<{ saved: string }>('/api/ensemble', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ stems, method }) }),
  remove: (stem: string) => j<any>(`/api/results/${stem}`, { method: 'DELETE' }),
  sealedPreview: (stem: string) => j<any>(`/api/sealed/preview/${stem}`),
  sealedRun: (stem: string, note: string, allowReopen = false) => j<{ job: Job; config_hash: string }>('/api/sealed/run', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ stem, note, confirm: true, allow_reopen: allowReopen }) }),
  sealedLog: (mode: Mode = 'trading') => j<any[]>(`/api/sealed/log?mode=${mode}`),
}
export const pct = (x?: number | null, d = 1) => x == null || Number.isNaN(x) ? '' : (x * 100).toFixed(d) + '%'
export const num = (x?: number | null, d = 3) => x == null || Number.isNaN(x) ? '' : x.toFixed(d)
