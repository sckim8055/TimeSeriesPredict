"""백테스트 엔진 — 모델이 낸 신호대로 실제로 사고팔았다면 돈이 어떻게 됐을지 계산한다.

═══════════════════════════════════════════════════════════════════════════════
 한 줄 요약
   "예측 → 신호(살까 말까) → 포지션(실제로 들고 있는 양) → 체결·비용 → 날마다 번 돈 → 성적표"
═══════════════════════════════════════════════════════════════════════════════

 쉬운 비유
   매일 저녁 장이 끝난 뒤 "내일 아침에 ETF 를 들고 있을까, 현금으로 있을까?"를 정하는 놀이다.
   저녁에 정한 대로 다음날 아침 첫 가격(시가)에 사거나 판다. 사고팔 때마다 수수료를 낸다.
   이걸 수천 일 반복하고, 처음 1원이 마지막에 얼마가 됐는지 본다.

 왜 '다음날 시가'인가 (고정 규칙, 바꿀 수 없음)
   오늘 종가를 보고 결정했는데 오늘 종가에 산다는 건 불가능하다(이미 장이 끝남).
   그래서 결정일 t 의 신호는 항상 t+1 시가에 체결한다. 이 규칙을 어기면 결과가 실제보다 좋아 보이는 '미래 엿보기'가 된다.

 이 파일에 있는 것
   CostModel              사고팔 때 드는 비용(수수료·슬리피지·세금)
   run_backtest           신호 → 자산곡선 (핵심)
   performance            수익률 → 성적표(총수익·연복리·샤프·최대낙폭 …)
   bootstrap_excess_pvalue  "B&H 보다 나은 게 우연인가?" 를 숫자로 (부트스트랩 p값)
   deflated_sharpe_pvalue   "여러 번 시도해서 운 좋게 나온 최고 성적 아닌가?" 보정
   return_moments           수익률의 치우침(왜도)·두꺼운 꼬리(첨도)
   diebold_mariano(_brier)  "두 예측 중 어느 쪽 오차가 정말로 작은가?" 검정

 포지션 단계
   1단계(지금): 0 = 현금, 1 = 전부 ETF(롱). 중간값(0.5 = 절반)도 계산은 되지만 신호는 0/1 로 만든다.
   2단계(미정): −1 인버스, 2 레버리지 등.

 입력 (run_backtest)
   prices : 날짜별 OHLCV 표. 열 이름 open·close 가 있어야 한다(대소문자 무관). 주간·월간으로 묶은 표도 되며,
            그때 '하루'는 '한 주기'로 읽는다.
   signal : 결정일 t 를 인덱스로 하는 숫자 시리즈. 0=현금, 1=롱. 정하지 못한 날(NaN)은 '어제 포지션 그대로'.
 출력
   (daily, summary) — daily 는 결정일 t 의 행마다 체결·손익 기록, summary 는 성적표 dict.
   결과 파일 *_daily.parquet 의 열과 같으니, 저장된 결과로 한 줄씩 손으로 다시 계산해 볼 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)  # frozen = 만든 뒤 값을 바꿀 수 없음(실수로 비용이 중간에 바뀌는 것을 막음)
class CostModel:
    """한 번 사거나 팔 때(편도) 드는 비용.

    예) 100만 원어치를 사면 비용 = 100만 × (0.00015 + 5/10,000) = 100만 × 0.00065 = 650원.
        다시 팔 때 또 650원 → 한 번 왕복에 약 1,300원(0.13%).

    항목          기본값     뜻                                   추천 범위·근거
    fee_rate      0.00015    증권사 수수료 0.015%                 0.00001~0.0005 (온라인 0.001%~0.015%, 지점 거래는 더 큼)
    slippage_bp   5.0        원하는 가격과 실제 체결 가격의 차이   1~20bp. 거래량 많은 ETF·소액이면 1~5, 큰 금액·변동 큰 날이면 10 이상
                             (bp = 0.01%, 5bp = 0.05%)
    tax_rate      0.0        세금                                  국내 주식형 ETF(091160)는 매매차익 비과세 → 0.
                                                                   ※ 지금 계산에는 쓰이지 않는다(one_way 에 안 들어감). 과세 상품을 넣을 때 연결해야 함.
    """

    fee_rate: float = 0.00015  # 편도 수수료 (0.015%)
    slippage_bp: float = 5.0  # 편도 슬리피지 (bp)
    tax_rate: float = 0.0  # 국내주식형 ETF 매매차익 비과세

    @property
    def one_way(self) -> float:
        """편도 비용 비율 = 수수료 + 슬리피지(bp → 비율). 기본값이면 0.00015 + 0.0005 = 0.00065 (0.065%)."""
        return self.fee_rate + self.slippage_bp / 10_000.0


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    *,
    cost: CostModel = CostModel(),
    initial_capital: float = 1.0,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """신호를 체결해 자산곡선을 만든다.

    매개변수
      prices           OHLCV 표(위 설명). 적어도 open·close 열.
      signal           목표 포지션(0~1). NaN = 어제 그대로.
      cost             CostModel. 기본 = 수수료 0.015% + 슬리피지 5bp.
      initial_capital  시작 금액. 기본 1.0 → 결과 equity 가 1.35 면 35% 번 것. (금액이 아니라 '배수'로 보면 됨)
      periods_per_year 1년에 몇 기간인가. 일간 252(한국 거래일 수), 주간 52, 월간 12. 연율화(샤프·연복리)에만 쓰임.

    회계(결정일 t 행에 기록):
    - pos_t: t 마감 후 결정한 목표 포지션. 실제 보유는 t+1 시가부터.
    - 보유 수익률은 체결가(시가) 기준: 포지션이 t+1 시가에 들어가 t+2 시가에 나오면 수익률 = open[t+2]/open[t+1] − 1.
      즉 t 결정의 1기간 수익 = pos_t × (open[t+2]/open[t+1] − 1) − 비용.
    - 비용: 포지션이 바뀔 때마다 변동량 × 편도 비용.
    마지막 두 행은 체결·청산가가 없어 NaN으로 남는다.

    손으로 따라 해 보는 예 (비용 0.00065, 시가 100 → 102 → 101 → 103)
      날짜  시가  신호  포지션  보유수익(다음날 시가→그다음날 시가)  비용             순수익
      t0    100   1     1       102→101: −0.98%                    1×0.00065        −0.98% − 0.065% = −1.045%
      t1    102   1     1       101→103: +1.98%                    0 (안 바꿈)       +1.98%
      t2    101   0     0       (103→?) 보유 안 함                  1×0.00065        −0.065%
      → 같은 신호를 유지하면 비용이 없고, 바꿀 때만 낸다.

    알고 있는 한계(분석할 때 주의)
      - 타겟이 k 기간 뒤(k>1)여도 여기서는 매 기간 다시 판단하고 1기간씩만 보유한다(타겟과 보유 기간이 다름).
      - 마지막 날 들고 있던 것을 파는 비용은 넣지 않는다.
    """
    p = prices.copy()  # 원본 표를 건드리지 않도록 복사
    p.columns = [c.lower() for c in p.columns]  # 'Open'·'OPEN' 도 'open' 으로 통일
    idx = p.index  # 날짜 목록(결정일)
    sig = signal.reindex(idx).astype(float)  # 신호를 가격 날짜에 맞춤. 신호가 없는 날은 NaN
    # 포지션 = 신호를 앞으로 채움(정하지 못한 날은 어제 그대로) → 맨 처음 신호 전은 0(현금) → 0~1 로 자름(1단계: 롱/현금)
    pos = sig.ffill().fillna(0.0).clip(0.0, 1.0)  # 1단계: 롱/현금
    entry = p["open"].shift(-1)  # 진입 가격 = 다음날(t+1) 시가.  shift(-1) = 한 칸 '뒤 날짜' 값을 지금 행으로 당겨 옴
    exit_ = p["open"].shift(-2)  # 청산 가격 = 그다음날(t+2) 시가
    hold_ret = exit_ / entry - 1.0  # 한 기간 들고 있었을 때의 수익률. 예) 101/102 − 1 = −0.98%
    turnover = pos.diff().abs().fillna(pos.abs())  # 포지션이 바뀐 양. 0→1 이면 1, 1→1 이면 0. 첫날은 0 에서 시작한 것으로 봄
    gross = pos * hold_ret  # 비용 빼기 전 수익 = 들고 있던 비율 × 그 기간 수익률
    costs = turnover * cost.one_way  # 비용 = 바꾼 양 × 편도 비용
    net = gross - costs  # 순수익률
    equity = initial_capital * (1.0 + net.fillna(0.0)).cumprod()  # 자산곡선: (1+r1)×(1+r2)×… 누적 곱. NaN(마지막 두 날)은 0 수익으로
    bench = initial_capital * (1.0 + hold_ret.fillna(0.0)).cumprod()  # 비교 기준: 처음부터 끝까지 계속 들고 있기(B&H, 비용 없음)
    daily = pd.DataFrame(
        {
            "open": p["open"],                  # 결정일의 시가(참고용)
            "close": p["close"],                # 결정일의 종가 — 결정에 쓸 수 있는 마지막 가격
            "signal": sig,                      # 모델이 낸 신호(NaN 포함, 원본 그대로)
            "position": pos,                    # 실제로 들고 갈 비율(0~1)
            "hold_return": hold_ret,            # t+1 시가 → t+2 시가 수익률
            "gross_return": gross,              # 포지션 × 보유수익
            "cost": costs,                      # 그날 낸 비용
            "net_return": net,                  # 순수익
            "equity": equity,                   # 전략 자산(시작 = initial_capital)
            "benchmark_equity": bench,          # B&H 자산
        }
    )
    summary = performance(daily["net_return"], daily["hold_return"], periods_per_year=periods_per_year)
    summary.update(
        n_trades=int((turnover > 0).sum()),  # 포지션을 바꾼 날 수(사거나 판 횟수)
        turnover=float(turnover.sum()),  # 바꾼 양의 합. 0↔1 왕복 한 번 = 2
        exposure=float(pos.mean()),  # 평균적으로 얼마나 들고 있었나. 0.6 = 기간의 60% 동안 보유
        total_cost=float(costs.sum()),  # 낸 비용의 합(수익률 단위). 0.01 = 1%
    )
    return daily, summary


def performance(r: pd.Series, bench: pd.Series | None = None, *, periods_per_year: int = 252) -> dict:
    """수익률 시리즈의 성과 지표. 모든 지표 정의는 여기 한 곳 (CLAUDE.md 규약: 화면은 계산하지 않는다).

    입력  r = 기간별 수익률(0.01 = 1%), bench = 비교 기준 수익률(선택), periods_per_year = 연율화 기준(일간 252)
    출력 (dict)
      지표                    뜻                                                     읽는 법(대략)
      n_periods               계산에 쓴 기간 수
      total_return            총수익률. (1+r1)(1+r2)… − 1                            0.35 = 35%
      cagr                    연복리 수익률. 총수익을 '해마다 같은 비율로 벌었다면'으로   예금 금리와 비교 가능
      ann_vol                 연 변동성 = 표준편차 × √252                             반도체 ETF 는 보통 0.3~0.5
      sharpe                  위험 한 단위당 수익 = 평균 / 표준편차 × √252 (무위험금리 0 가정)  0.5 보통 · 1 좋음 · 2 이상은 의심부터
      sortino                 샤프와 같지만 '떨어진 날의 흔들림'만 위험으로 봄
      mdd                     최대 낙폭: 가장 높았던 때에서 가장 많이 떨어진 비율       −0.4 = 꼭대기에서 40% 잃은 적 있음
      win_rate                수익 난 기간 ÷ 손익이 0 이 아니었던 기간                 현금이던 날(0)은 빼고 셈
      benchmark_*             같은 기간 B&H 의 총수익·연복리·최대낙폭
      excess_total_return     전략 총수익 − B&H 총수익
      excess_mean_per_period  기간마다 (전략 − B&H) 의 평균
    예) r = [0.01, −0.02, 0.03] → total = 1.01×0.98×1.03 − 1 = 1.95%
    """
    r = r.dropna()  # 계산할 수 없는 날(NaN) 제외
    n = len(r)
    if n == 0:
        return {}
    eq = (1 + r).cumprod()  # 자산곡선(시작 1)
    years = n / periods_per_year  # 몇 년치인가
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else float("nan")  # 최종배수^(1/년수) − 1
    mu, sd = r.mean(), r.std(ddof=1)  # 평균, 표본 표준편차(ddof=1: n−1 로 나눔)
    sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")  # √252 를 곱해 1년 단위로
    downside = r[r < 0].std(ddof=1)  # 손실 난 날만의 표준편차
    sortino = float(mu / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0  # 그때까지의 최고점 대비 지금 몇 % 아래인가(0 이하)
    out = {
        "n_periods": int(n),
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": cagr,
        "ann_vol": float(sd * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": float(dd.min()),  # 가장 깊었던 낙폭
        "win_rate": float((r > 0).sum() / max((r != 0).sum(), 1)),  # max(…,1): 0 으로 나누기 방지
    }
    if bench is not None:
        b = bench.reindex(r.index).dropna()  # 전략과 같은 날짜만
        rr = r.reindex(b.index)
        beq = (1 + b).cumprod()
        out["benchmark_total_return"] = float(beq.iloc[-1] - 1)
        out["benchmark_cagr"] = float(beq.iloc[-1] ** (1 / (len(b) / periods_per_year)) - 1) if len(b) else float("nan")
        out["benchmark_mdd"] = float((beq / beq.cummax() - 1).min())
        out["excess_total_return"] = out["total_return"] - out["benchmark_total_return"]
        out["excess_mean_per_period"] = float((rr - b).mean())
    return out


# ---------------------------------------------------------------------------
# 신뢰도: 부트스트랩 p값, Deflated Sharpe, Diebold-Mariano
#   p값 읽는 법: "정말로는 효과가 없는데 우연히 이만큼 좋게 나올 확률".
#   작을수록(보통 0.05 미만) 우연이 아니라고 볼 근거가 된다. 0.3 이면 열 번 중 세 번은 운으로도 나오는 성적.
# ---------------------------------------------------------------------------
def bootstrap_excess_pvalue(strategy: pd.Series, bench: pd.Series, *, n_boot: int = 2000, block: int = 20, seed: int = 0) -> float:
    """H0: 전략의 평균 초과수익 ≤ 0. 블록 부트스트랩(자기상관 보존)으로 p값.

    쉬운 설명
      "B&H 보다 하루 평균 조금 더 벌었다"가 운인지 보려고, 초과수익 기록을 20일 묶음(블록)으로 잘라
      무작위로 다시 이어 붙인 가짜 역사 2,000개를 만든다. 진짜 효과가 0 이라고 가정한 가짜 역사들 중
      실제 성적만큼(이상) 좋은 것이 몇 % 나오는지가 p값이다.
      묶음으로 자르는 이유: 오늘과 내일의 수익은 서로 닮아 있어서(자기상관) 하루씩 섞으면 그 성질이 사라진다.

    매개변수  n_boot 2000(기본, 1000~10000 — 늘리면 정확하지만 느림) · block 20(일간 ≈ 한 달, 주간 4~8, 월간 3~6)
              seed 0(같은 결과 재현용)
    반환      p값(0~1). 기간이 block×2 보다 짧으면 NaN.
    """
    d = (strategy - bench).dropna().values  # 날마다 초과수익
    n = len(d)
    if n < block * 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = d.mean()  # 실제 평균 초과수익
    centered = d - obs  # 평균을 0 으로 옮김 = "효과가 없다"는 가정의 세계
    nb = int(np.ceil(n / block))  # 블록 몇 개를 이어야 원래 길이가 되나
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb)  # 블록 시작 위치를 무작위로
        sample = np.concatenate([centered[s : s + block] for s in starts])[:n]  # 이어 붙이고 원래 길이로 자름
        means[i] = sample.mean()
    return float((means >= obs).mean())  # 가짜 세계에서 실제 이상으로 좋게 나온 비율


def deflated_sharpe_pvalue(
    sharpe_annual: float,
    n_periods: int,
    n_trials: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    periods_per_year: int = 252,
    trial_sr_var: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014) 의 p값.

    쉬운 설명
      동전을 100번 던지는 사람을 1,000명 모으면, 그중 1등은 우연만으로도 앞면이 60번 넘게 나온다.
      전략을 여러 번 시험하고 제일 좋은 것만 보면 같은 일이 생긴다. 이 함수는 "N번 시도했다면 운으로도 나올 만한
      최고 샤프"(SR0)를 먼저 구하고, 내 샤프가 그보다 확실히 높은지를 p값으로 낸다.
      n_trials 는 지금까지 이 데이터로 돌려 본 실험 수(results/experiment_counter.json)다 — 많이 돌릴수록 기준이 높아진다.

    1) 기대 최대 샤프: 서로 독립인 N번 시도의 샤프가 분산 V 인 정규분포를 따를 때, 그 최댓값의 기대치
         SR0 = sqrt(V) · [ (1−γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e)) ],  γ = 오일러 상수
       V = trial_sr_var (실제 시도들의 기간 샤프 분산). 주어지지 않으면 표본 표준오차² 1/(n−1) 로 대체(근사).
    2) 확률적 샤프(PSR): 관측 기간 샤프 SR 이 SR0 를 넘을 확률
         z = (SR − SR0) · sqrt(n−1) / sqrt(1 − γ3·SR + (γ4−1)/4 · SR²),  γ3 = 왜도, γ4 = 첨도(정규=3)
       반환 p = 1 − Φ(z). 작을수록 "N번 시도 중 우연히 나온 최대치가 아니다"에 가깝다.
    샤프는 모두 기간 단위(연율 ÷ sqrt(periods_per_year)). n_trials ≤ 1 이면 SR0 = 0 (보정 없음).

    예) 연 샤프 1.0, 일간 1,000일, 시도 1번 → p ≈ 0.001 / 같은 성적인데 시도 200번 → p 가 크게 올라감(우연일 수 있음)
    """
    from math import log, sqrt  # noqa: F401

    from scipy.stats import norm

    if not np.isfinite(sharpe_annual) or n_periods < 2:
        return float("nan")
    sr = sharpe_annual / sqrt(periods_per_year)  # 연 샤프 → 하루(기간) 샤프
    sr0 = 0.0
    if n_trials and n_trials > 1:
        gamma = 0.5772156649015329  # 오일러-마스케로니 상수
        z1 = norm.ppf(1 - 1 / n_trials)
        z2 = norm.ppf(1 - 1 / (n_trials * np.e))
        v = trial_sr_var if (trial_sr_var is not None and np.isfinite(trial_sr_var) and trial_sr_var > 0) else 1.0 / (n_periods - 1)
        sr0 = sqrt(v) * ((1 - gamma) * z1 + gamma * z2)  # N번 시도했을 때 운으로도 기대되는 최고 샤프
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2  # 수익률 분포가 치우치거나 꼬리가 두꺼우면 불확실성이 커짐
    if denom <= 0:
        return float("nan")
    z = (sr - sr0) * sqrt(n_periods - 1) / sqrt(denom)
    return float(1 - norm.cdf(z))


def return_moments(r: pd.Series) -> tuple[float, float]:
    """(왜도, 첨도[정규=3]) — Deflated Sharpe 입력.

    왜도: 0 이면 좌우 대칭, 음수면 '가끔 크게 잃는' 쪽으로 치우침(주식에 흔함).
    첨도: 정규분포 = 3. 3 보다 크면 아주 큰 상승·하락이 정규분포보다 자주 나옴(꼬리가 두꺼움).
    자료가 10개 미만이면 계산이 불안정해 정규분포 값(0, 3)을 돌려준다.
    """
    from scipy.stats import kurtosis, skew as _skew

    x = pd.Series(r).dropna().values
    if len(x) < 10:
        return 0.0, 3.0
    return float(_skew(x)), float(kurtosis(x, fisher=False))  # fisher=False: 정규=3 기준(True 면 정규=0)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, *, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold–Mariano(1995) 검정. e1, e2 = 두 모델의 예측 오차(같은 구간). 손실 = |e|^power.
    H0: 평균 손실 차이 = 0. 반환 (DM 통계량, 양측 p값). h>1이면 Newey-West(h−1 lag) 분산.
    통계량 < 0 이면 모델1 손실이 작다(모델1이 낫다).

    쉬운 설명
      같은 날들에 대해 두 예측의 '틀린 정도'를 날마다 빼 보고(d = 손실1 − 손실2), 그 평균이 0 과 확실히 다른지 본다.
      평균이 음수 = 모델1 이 덜 틀림. p값이 작을수록 그 차이가 우연이 아님.
    매개변수
      h      몇 기간 앞 예측인가(타겟의 k). k 일 앞을 예측하면 이웃한 날들의 오차가 서로 겹쳐 닮으므로 분산을 보정한다.
      power  2 = 제곱오차(큰 실수를 더 무겁게, 기본·사전 등록 기준) · 1 = 절대오차(MAPE 계열 비교용)
    예) e1 = 모델 오차, e2 = 'naive(변화 없음)' 오차 → DM < 0 이고 p < 0.05 면 모델이 naive 보다 유의하게 낫다.
    자료가 10개 미만이면 NaN.
    """
    from scipy.stats import norm

    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power  # 날마다 손실 차이
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    mean = d.mean()
    dc = d - mean
    var = dc @ dc / n  # 손실 차이의 분산(자기 자신)
    for lag in range(1, h):  # h>1: 겹치는 기간만큼 이웃 날과의 공분산을 더함(Newey-West, 가중치 1 − lag/h)
        w = 1 - lag / h
        var += 2 * w * (dc[lag:] @ dc[:-lag]) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = mean / np.sqrt(var / n)  # 평균 ÷ 평균의 표준오차 = 표준정규 통계량
    return float(dm), float(2 * (1 - norm.cdf(abs(dm))))  # 양측 p값


def clark_west(y: np.ndarray, f_model: np.ndarray, f_bench: np.ndarray | float = 0.0, *, h: int = 1) -> tuple[float, float]:
    """Clark–West(2007) 검정 — 기준 예측(f_bench)이 모델에 '내포'될 때(모델의 계수가 모두 0 이면 기준과 같아질 때) 쓰는 검정.
    반환 (통계량, 한쪽 p값). 통계량이 클수록 '모델이 기준보다 낫다'. H0: 모집단에서 모델이 기준보다 나을 것이 없다(변수에 예측 정보가 없다).

    왜 DM 과 따로 두는가
      변수에 정보가 전혀 없어도, 모델은 표본에서 계수를 추정하므로 예측이 0 주변에서 흔들리고 그만큼 제곱오차가 기준보다 '커진다'.
      DM 은 이 추정 잡음을 포함한 오차를 그대로 비교한다 → "이 방법을 이 학습 창으로 쓰면 기준보다 정확한가"(Giacomini–White 2006 의 해석, 실용 질문)에 답한다.
      CW 는 그 잡음만큼(Σ(f_bench − f_model)²) 모델의 제곱오차를 깎아 비교한다 → "변수에 예측 정보가 있는가"(과학 질문)에 답한다.
      그래서 정보 없는 변수를 넣은 모델은 DM 으로 '기준보다 유의하게 나쁨'이 나올 수 있고(사실: 그 방법은 더 부정확하다),
      그것을 '변수에 정보가 없다'의 증거로 읽으면 안 된다. 반대로 CW 가 유의해도 그 방법이 기준보다 정확하다는 뜻은 아니다.

    계산
      날마다  adj_t = (y_t − f_bench_t)² − [ (y_t − f_model_t)² − (f_bench_t − f_model_t)² ]
      기준이 0 예측이면 adj_t = 2·y_t·f_model_t  (예측과 실제가 같은 방향으로 함께 움직이는가)
      통계량 = adj 의 평균 ÷ 평균의 표준오차. h>1 이면 겹치는 기간만큼 Newey–West(h−1 lag, 가중치 1 − lag/h) 분산.
      p값은 한쪽(통계량이 클수록 작음). 표준정규 근사 — Clark·West 는 이 근사가 약간 보수적(실제 기각률이 명목보다 조금 낮음)이라고 보고했다.
    한계: 굴리는 창(rolling window)에서 근사가 가장 잘 맞고, 기준이 모델에 내포되지 않으면(예: 서로 다른 변수를 쓰는 두 모델) 쓰지 않는다 — 그때는 DM.
    분류(확률) 예측은 y·f 를 0.5 만큼 옮겨 같은 식을 쓴다(clark_west(y, p, 0.5)).
    자료가 10개 미만이거나 분산이 0 이면 NaN.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    fm = np.asarray(f_model, dtype=float)
    fb = np.broadcast_to(np.asarray(f_bench, dtype=float), y.shape)
    adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    adj = adj[np.isfinite(adj)]
    n = len(adj)
    if n < 10:
        return float("nan"), float("nan")
    mean = adj.mean()
    dc = adj - mean
    var = dc @ dc / n
    for lag in range(1, h):
        w = 1 - lag / h
        var += 2 * w * (dc[lag:] @ dc[:-lag]) / n
    if var <= 0:
        return float("nan"), float("nan")
    t = mean / np.sqrt(var / n)
    return float(t), float(1 - norm.cdf(t))


def diebold_mariano_brier(prob: np.ndarray, y: np.ndarray, *, h: int = 1, baseline: float = 0.5) -> tuple[float, float]:
    """분류 타겟용 DM 검정. 손실 = Brier(제곱오차) — 모델 확률 vs 무정보 기준(baseline 확률, 기본 0.5).
    통계량 < 0 이면 모델이 무정보보다 낫다. 회귀의 'DM vs 0예측' 과 같은 자리.

    예) 오를 확률을 0.7 로 냈는데 실제로 올랐다(y=1) → 손실 (0.7−1)² = 0.09. 무정보 0.5 → 0.25. 모델이 더 작음.
    baseline: 기본 0.5(반반). 학습 구간에서 오른 날의 비율(예 0.53)을 넣으면 더 엄격한 기준이 된다.
    """
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=float)
    return diebold_mariano(prob - y, baseline - y, h=h, power=2)
