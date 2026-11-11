"""BIVA 수정본 실행.
원본(biva_orig_run)과의 차이 — 세 가지 수정 + 파라미터화:
  [F1] 타겟 정렬: 정답 = 입력 윈도우 '다음' pred_len개 시점 (원본은 입력 마지막 행 = 정답 → 누출)
  [F2] 스케일러: 학습 구간에만 fit (원본은 전체 기간에 fit_transform)
  [F3] 손실: 타겟도 스케일하고 가중합 w_pred*MSE + w_vae*VAE + w_imp*Imputation (원본은 원단위 MSE가 VAE·보간 손실을 압도)
  [P]  경로·장치·pred_len·seq_len 전부 인자. 모델 신경망 코드(model/)는 원본 그대로(장치 패치만).
학습 루프(Adam, lradj '3', 500 epoch, best-val 체크포인트)는 원본과 동일하게 유지.
"""
import sys, os, json, time, random, argparse, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.set_num_threads(1)
from sklearn.preprocessing import MinMaxScaler
from model import BIVA

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join(HERE, '..', 'biva_orig', 'Base_data', 'MacroEco_S03_20250727_2019_SEOUL_All_Local_34.xlsx'))
ap.add_argument('--target', default='F11'); ap.add_argument('--start', default='2009-01')
ap.add_argument('--seq_len', type=int, default=6); ap.add_argument('--pred_len', type=int, default=1)
ap.add_argument('--epochs', type=int, default=500); ap.add_argument('--batch', type=int, default=3); ap.add_argument('--lr', type=float, default=0.01)
ap.add_argument('--w_pred', type=float, default=1.0); ap.add_argument('--w_vae', type=float, default=0.1); ap.add_argument('--w_imp', type=float, default=0.05)
ap.add_argument('--leak', action='store_true', help='[진단용] 원본처럼 정답=입력 마지막 행 (누출 재현)')
ap.add_argument('--diff', action='store_true', help='[F4 실험] 타겟을 수준이 아닌 변화량(y[t+h]-y[t_last])으로 학습, 예측 후 수준 복원')
ap.add_argument('--seed', type=int, default=2025); ap.add_argument('--tag', default='')
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 데이터 (원본 Dataset_BIVA 의 결합 방식 유지: df_M + df_Q, 타겟을 마지막 열로) ----------
dM = pd.read_excel(a.data, sheet_name='df_M', index_col='date').astype(float).loc[a.start:]
dQ = pd.read_excel(a.data, sheet_name='df_Q', index_col='date').astype(float).loc[a.start:]
cols = [c for c in dM.columns if c != a.target] + [a.target]
df = pd.concat([dM[cols], dQ], axis=1).loc[dM.index]      # 월간 인덱스 기준, 분기값은 분기말 월에만 존재(그 외 NaN → BRITS가 보간)
cols = [c for c in df.columns if c != a.target] + [a.target]; df = df[cols]
C = df.shape[1]; T = len(df); ti = cols.index(a.target)
n_train = int(T * 0.6); n_test = int(T * 0.3); n_val = T - n_train - n_test
b = {'train': (0, n_train), 'val': (n_train - a.seq_len, n_train + n_val), 'test': (T - n_test - a.seq_len, T)}

sc = MinMaxScaler().fit(df.values[:n_train])                       # [F2] 학습 구간에만 fit
X_all = sc.transform(df.values).astype('float32')                  # 입력·타겟 모두 스케일 [F3]
tmin, tmax = sc.data_min_[ti], sc.data_max_[ti]
inv = lambda z: z * (tmax - tmin) + tmin
dscale = float(np.nanstd(np.diff(X_all[:n_train, ti]))) if a.diff else 1.0

def windows(lo, hi):
    xs, ys, ids = [], [], []
    for s in range(lo, hi - a.seq_len - (0 if a.leak else a.pred_len) + 1):
        e = s + a.seq_len
        y_idx = list(range(e - 1, e - 1 + a.pred_len)) if a.leak else list(range(e, e + a.pred_len))   # [F1]
        if y_idx[-1] >= hi: break
        xs.append(X_all[s:e]); ids.append(y_idx)
        ys.append((X_all[y_idx, ti] - X_all[e - 1, ti])[:, None] / dscale if a.diff else X_all[y_idx, ti][:, None])
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys)), ids
data = {k: windows(*v) for k, v in b.items()}
print({k: tuple(v[0].shape) for k, v in data.items()})

# ---------- 모델 (원본 args 그대로) ----------
class args: pass
args.seq_len=a.seq_len; args.pred_len=a.pred_len; args.label_len=1; args.channels=C
args.rnn_hid_size=128; args.vae_hid_size=128; args.vae_latent_size=4; args.infer_hid_size=128
args.moving_avg=6; args.batch_size=a.batch; args.imputed_weight=0.05; args.conv1d=True; args.conv_kernal=1; args.RIN=False; args.combination=True
model = BIVA.Model(args).float().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr); mse = nn.MSELoss()

def run_epoch(X, Y, train):
    model.train(train); tot = []; order = torch.randperm(len(X)) if train else torch.arange(len(X))
    for i in range(0, len(X) - (len(X) % a.batch if train else 0), a.batch):   # drop_last=True (원본 동일) 학습 시
        idx = order[i:i + a.batch]; bx, by = X[idx].to(dev), Y[idx].to(dev)
        with torch.set_grad_enabled(train):
            states, vae_loss, recon, seas, imp_loss, imp_x = model(bx)
            loss = a.w_pred * mse(states, by) + a.w_vae * vae_loss + a.w_imp * imp_loss      # [F3]
        if train: opt.zero_grad(); loss.backward(); opt.step()
        tot.append(loss.item())
    return float(np.mean(tot))

t0 = time.time(); best = (1e9, None)
for ep in range(1, a.epochs + 1):
    tr = run_epoch(*data['train'][:2], True); va = run_epoch(*data['val'][:2], False)
    if va < best[0]: best = (va, {k: v.detach().clone() for k, v in model.state_dict().items()})
    for g in opt.param_groups: g['lr'] = a.lr if ep < 10 else a.lr * 0.1          # lradj '3' (원본 동일)
    if ep % 50 == 0 or ep == 1: print(f'ep {ep} train {tr:.5f} val {va:.5f}')
model.load_state_dict(best[1]); model.eval()
Xt, Yt, ids = data['test']
with torch.no_grad(): P = model(Xt.to(dev))[0].cpu().numpy()
last = Xt.numpy()[:, -1, ti][:, None]
if a.diff:
    pred = inv(P[:, :, 0] * dscale + last); true = inv(Yt.numpy()[:, :, 0] * dscale + last)
else:
    pred = inv(P[:, :, 0]); true = inv(Yt.numpy()[:, :, 0])
# 기준선: naive = 입력 마지막 행의 타겟값을 그대로 예측
naive = np.repeat(inv(Xt.numpy()[:, -1, ti])[:, None], a.pred_len, axis=1)
m = lambda p: dict(rmse=float(np.sqrt(np.mean((p - true) ** 2))), mae=float(np.mean(np.abs(p - true))), mape=float(np.mean(np.abs((p - true) / true))))
out = dict(variant='leak-repro' if a.leak else ('fixed-diff' if a.diff else 'fixed'), pred_len=a.pred_len, seq_len=a.seq_len, epochs=a.epochs,
           loss_w=[a.w_pred, a.w_vae, a.w_imp], n_test=int(len(Xt)), biva=m(pred), naive_last_value=m(naive),
           test_period=[str(df.index[ids[0][0]].date()), str(df.index[ids[-1][-1]].date())], seconds=round(time.time() - t0, 1), device=str(dev))
print(json.dumps(out, ensure_ascii=False))
tag = f"{out['variant']}-seq{a.seq_len}-pred{a.pred_len}{a.tag}"
json.dump(out, open(os.path.join(HERE, 'exp', f'summary_{tag}.json'), 'w'), ensure_ascii=False, indent=1)
np.savez(os.path.join(HERE, 'exp', f'pred_{tag}.npz'), pred=pred, true=true, naive=naive, dates=np.array([str(df.index[i[0]].date()) for i in ids]))
