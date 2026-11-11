"""변수 사전의 '데이터 받는 법'이 어댑터가 실제로 보내는 요청과 같은지(다른 사람이 같은 값을 받을 수 있게)."""
from __future__ import annotations

import pandas as pd

from adapters import datago_adapter, ecos_adapter, fred_adapter
from engine.api_spec import ecos_period, price_specs, request_spec


def test_fred_spec_matches_adapter_request(monkeypatch):
    sent = {}

    class R:
        def json(self):
            return {"observations": [{"date": "2020-01-02", "value": "1.5"}]}

    monkeypatch.setenv("FRED_API_KEY", "S3CRETVALUE")
    monkeypatch.setattr(fred_adapter.requests, "get", lambda url, params=None, timeout=None: sent.update(url=url, params=params) or R())
    fred_adapter.fetch_series("DGS10", "2006-01-01", None)
    spec = request_spec("FRED", "DGS10", "2006-01-01", None)
    assert spec["url"] == sent["url"] and {k: v for k, v in spec["params"].items() if k != "api_key"} == {k: v for k, v in sent["params"].items() if k != "api_key"}
    assert "S3CRETVALUE" not in str(spec) and "<FRED_API_KEY>" in spec["full_url"]


def test_ecos_spec_matches_adapter_url(monkeypatch):
    sent = {}

    class R:
        def json(self):
            return {"StatisticSearch": {"row": [{"TIME": "20200102", "DATA_VALUE": "1.0"}]}}

    monkeypatch.setenv("ECOS_API_KEY", "SECRET")
    monkeypatch.setattr(ecos_adapter.requests, "get", lambda url, timeout=None: sent.update(url=url) or R())
    ecos_adapter.fetch_series("817Y002", "010210000", "D", ecos_period(pd.Timestamp("2006-01-01"), "D"), ecos_period(pd.Timestamp("2026-11-03"), "D"))
    spec = request_spec("ECOS", "817Y002/010210000/D", "2006-01-01", None, "2026-11-03T14:19:10")
    assert spec["full_url"] == sent["url"].replace("SECRET", "<ECOS_API_KEY>")
    assert ecos_period(pd.Timestamp("2006-05-01"), "Q") == "2006Q2" and ecos_period(pd.Timestamp("2006-05-01"), "M") == "200605"


def test_price_specs_use_adapter_url_and_no_key():
    specs = price_specs("091160", "2006-01-01", None, "2026-11-03T00:00:00")
    assert specs[0]["url"] == datago_adapter.URL and "<DATAGO_API_KEY>" in specs[0]["full_url"]
    assert specs[1]["params"]["symbol"] == "091160.KS"
