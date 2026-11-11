"""BIVA 비교용 기준선: naive(마지막값), 계절 naive(12개월 전 변화율 적용), ARIMA(2,1,2) 워크포워드.
동일 데이터·동일 테스트 구간(60/10/30 분할의 마지막 30%)에서 pred_len 스텝 앞 예측 RMSE/MAE/MAPE.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'biva_orig', 'Base_data', 'MacroEco_S03_20250727_2019_SEOUL_All_Local_34.xlsx')
y = pd.read_excel(DATA, sheet_name='df_M', index_col='date')['F11'].astype(float).loc['2009-01':]
T = len(y); n_train = int(T * 0.6); n_test = int(T * 0.3); seq_len = 6
test_lo = T - n_test - seq_len  # biva_fixed 와 동일한 윈도우 시작

def metrics(p, t):
    p, t = np.array(p), np.array(t)
    return dict(rmse=float(np.sqrt(np.mean((p - t) ** 2))), mae=float(np.mean(np.abs(p - t))), mape=float(np.mean(np.abs((p - t) / t))))

out = {}
for h in [1, 3, 6]:
    naive, arima, true = [], [], []
    for s in range(test_lo, T - seq_len - h + 1):
        e = s + seq_len                      # 입력 마지막 행 = e-1, 정답 = e .. e+h-1
        hist = y.iloc[:e]
        tgt = y.iloc[e:e + h].values
        naive.append(np.repeat(hist.iloc[-1], h))
        try:
            fc = ARIMA(hist.values, order=(2, 1, 2)).fit().forecast(steps=h)
        except Exception:
            fc = np.repeat(hist.iloc[-1], h)
        arima.append(fc); true.append(tgt)
    out[f'pred_len_{h}'] = dict(n=len(true), naive_last_value=metrics(naive, true), arima_212_walkforward=metrics(arima, true))
    print(h, json.dumps(out[f'pred_len_{h}']))
json.dump(out, open(os.path.join(HERE, 'baselines_summary.json'), 'w'), indent=1)
