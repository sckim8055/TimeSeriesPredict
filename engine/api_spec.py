"""변수 사전용 '데이터 받는 법' 명세 — 같은 값을 다른 사람이 다시 받을 수 있도록, 어댑터(adapters/*.py)가 실제로 보내는 요청을 그대로 적는다.

원칙: 여기 적는 주소·매개변수는 adapters/*.py 와 collect.py 의 코드와 같아야 한다(시험 tests/test_api_spec.py 가 어댑터 상수와 대조).
API 키는 절대 적지 않는다 — 자리표시자 <ECOS_API_KEY> 처럼만.
"""
from __future__ import annotations

import importlib.metadata as md
from urllib.parse import urlencode

import pandas as pd

from adapters import ecos_adapter, fred_adapter


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def ecos_period(ts: pd.Timestamp, freq: str) -> str:
    """ECOS 가 받는 기간 문자열(collect.py 와 같은 규칙). D: 20060101, M: 200601, Q: 2006Q1, A: 2006."""
    if freq == "Q":
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    return ts.strftime({"D": "%Y%m%d", "M": "%Y%m", "A": "%Y"}[freq])


def request_spec(source: str, source_code: str, start: str, end: str | None, collected_utc: str | None = None) -> dict | None:
    """변수 하나를 받는 요청. start/end = 수집 때 쓴 기간(데이터 버전 manifest). end 가 None 이면 '수집일까지'."""
    end_txt = end or (f"수집일({str(collected_utc)[:10]})까지 — 다시 받을 때는 end 를 그 날짜로 고정" if collected_utc else "오늘까지")
    s = {"source": source, "source_code": source_code, "start": start, "end": end_txt}
    if source == "FRED":
        params = {"series_id": source_code, "api_key": "<FRED_API_KEY>", "file_type": "json", "observation_start": start}
        if end:
            params["observation_end"] = end
        s.update(method="GET", url=fred_adapter.URL, params=params, full_url=f"{fred_adapter.URL}?{urlencode(params, safe='<>')}", key_env="FRED_API_KEY",
                 response="observations[].date → 날짜, observations[].value → 값 (숫자가 아닌 값 '.' 은 버림)",
                 code="adapters/fred_adapter.py fetch_series",
                 python=f'requests.get("{fred_adapter.URL}", params={params!r}).json()["observations"]',
                 doc="https://fred.stlouisfed.org/docs/api/fred/series_observations.html")
    elif source == "ECOS":
        stat, item, fq = source_code.split("/")
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end) if end else (pd.Timestamp(collected_utc) if collected_utc else pd.Timestamp.today())
        a, b = ecos_period(t0, fq), ecos_period(t1, fq)
        url = ecos_adapter.URL.format(key="<ECOS_API_KEY>", stat=stat, freq=fq, start=a, end=b, item=item)
        s.update(method="GET", url=url, params={"서비스": "StatisticSearch", "인증키": "<ECOS_API_KEY>", "형식": "json", "언어": "kr", "시작 행": 1, "끝 행": 100000,
                                               "통계표 코드": stat, "주기": fq, "시작": a, "끝": b, "항목 코드1": item},
                 full_url=url, key_env="ECOS_API_KEY",
                 response="StatisticSearch.row[].TIME → 날짜(D: YYYYMMDD, M: YYYYMM, Q: YYYYQn, A: YYYY), row[].DATA_VALUE → 값 (숫자가 아니면 버림)",
                 code="adapters/ecos_adapter.py fetch_series (collect.py fetch_ecos 가 기간 문자열을 만듦)",
                 python=f'requests.get("{url}").json()["StatisticSearch"]["row"]',
                 doc="https://ecos.bok.or.kr/api/ (한국은행 ECOS Open API — 개발 가이드의 StatisticSearch)")
    elif source == "YF":
        call = f'yfinance.download("{source_code}", start="{start}", end={end!r}, progress=False, auto_adjust=False, multi_level_index=False)["Close"]'
        s.update(method="라이브러리(yfinance)", url=f"https://finance.yahoo.com/quote/{source_code}", params={"symbol": source_code, "start": start, "end": end, "auto_adjust": False, "열": "Close(원 종가, 배당·분할 보정 없음)"},
                 full_url=None, key_env=None, library=f"yfinance {_ver('yfinance') or '(설치 안 됨)'}",
                 response="일별 OHLCV 표 → Close 열",
                 code="adapters/yf_adapter.py fetch_series", python=call,
                 note="Yahoo 의 공식 API 가 아니라 yfinance 가 Yahoo 웹 데이터를 읽는 방식(비공식). Yahoo 가 과거 값을 고치면 다시 받은 값이 달라질 수 있으므로, 같은 값이 필요하면 데이터 버전 폴더의 원본(raw) 파일을 함께 넘기세요.")
    elif source == "KRX":
        parts = source_code.split(":")
        if len(parts) >= 3 and parts[0] == "flow":
            on = parts[3] if len(parts) > 3 else "순매수"
            call = (f'pykrx.stock.get_etf_trading_volume_and_value("YYYYMMDD", "YYYYMMDD", "{parts[1]}", "거래대금", "{on}")  # ETF'
                    f'\npykrx.stock.get_market_trading_value_by_date("YYYYMMDD", "YYYYMMDD", "{parts[1]}", on="{on}")  # 주식')
            s.update(method="라이브러리(pykrx, KRX 로그인)", url="https://data.krx.co.kr (pykrx 가 KRX 정보데이터시스템 화면 요청을 대신 보냄)",
                     params={"종목": parts[1], "투자자": parts[2], "값": on, "단위": "거래대금(원)", "기간": "연 단위로 나눠 요청(1월 1일~12월 31일)"},
                     full_url=None, key_env="KRX_ID / KRX_PW", library=f"pykrx {_ver('pykrx') or '(설치 안 됨)'}",
                     response=f"일별 투자자별 거래대금 표 → '{parts[2]}' 열 (ETF 경로는 '외국인'→'외국인합계', '기관'→'기관합계' 로 이름 바꿈)",
                     code="adapters/krx_adapter.py fetch_flow", python=call,
                     note="주식 종목표에 없는 티커는 ETF 경로로 요청합니다. KRX 는 요청이 많으면 일시 차단할 수 있어 요청 사이 1초 쉬고, 지난 해 결과는 파일 캐시를 다시 씁니다.")
    elif source == "DERIVED":
        s.update(method="계산(다른 변수로)", url=None, params={"식": source_code}, full_url=None, key_env=None,
                 response="식에 나온 변수들의 같은 날짜 값으로 계산(날짜 교집합)", code="engine/derived.py compute_derived",
                 python=f"compute_derived: {source_code}")
    else:
        return None
    return s


def price_specs(ticker: str, start: str, end: str | None, collected_utc: str | None = None) -> list[dict]:
    """ETF 가격(실험의 기본 데이터) — 세 출처를 받아 조립한다(collect.py 2단계)."""
    from adapters import datago_adapter
    end_txt = end or f"수집일({str(collected_utc)[:10]})까지"
    dg_params = {"serviceKey": "<DATAGO_API_KEY>", "resultType": "json", "numOfRows": 500, "pageNo": 1, "likeSrtnCd": ticker,
                 "beginBasDt": "YYYY0101", "endBasDt": "(YYYY+1)0101"}
    yf_sym = ticker if "." in ticker or not ticker.isdigit() else f"{ticker}.KS"
    return [
        {"source": "DATAGO(공공데이터포털, 공식)", "role": "주 출처(2020~)", "method": "GET", "url": datago_adapter.URL, "params": dg_params,
         "full_url": f"{datago_adapter.URL}?{urlencode(dg_params, safe='<>()+')}", "key_env": "DATAGO_API_KEY",
         "response": "response.body.items.item[] 의 basDt(날짜)·mkp(시가)·hipr(고가)·lopr(저가)·clpr(종가)·trqu(거래량), srtnCd 가 종목코드와 같은 행만",
         "code": "adapters/datago_adapter.py fetch_prices", "note": "연 단위로 나눠 요청(endBasDt 는 끝 날짜 미포함), 500행씩 페이지. 서버 오류(5xx)는 최대 4번 재시도.", "start": start, "end": end_txt},
        {"source": "Yahoo(yfinance)", "role": "공식 데이터 이전 기간(2009~2019)", "method": "라이브러리(yfinance)", "url": f"https://finance.yahoo.com/quote/{yf_sym}",
         "params": {"symbol": yf_sym, "start": start, "end": end, "auto_adjust": False}, "full_url": None, "key_env": None, "library": f"yfinance {_ver('yfinance') or '(설치 안 됨)'}",
         "python": f'yfinance.download("{yf_sym}", start="{start}", end={end!r}, progress=False, auto_adjust=False, multi_level_index=False)',
         "response": "Open·High·Low·Close·Volume (원 주가, 수정 없음)", "code": "adapters/yf_adapter.py fetch_prices", "start": start, "end": end_txt},
        {"source": "네이버(pykrx, 수정주가)", "role": "빠진 거래일 비율 보정·교차검증", "method": "라이브러리(pykrx)", "url": "https://finance.naver.com (pykrx 가 차트 데이터 요청)",
         "params": {"ticker": ticker, "adjusted": True}, "full_url": None, "key_env": None, "library": f"pykrx {_ver('pykrx') or '(설치 안 됨)'}",
         "python": f'pykrx.stock.get_market_ohlcv_by_date("YYYYMMDD", "YYYYMMDD", "{ticker}", adjusted=True)',
         "response": "시가·고가·저가·종가·거래량", "code": "adapters/pykrx_adapter.py fetch_prices · engine/repair.py merge_prices", "start": start, "end": end_txt},
    ]
