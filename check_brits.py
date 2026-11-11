"""BRITS 확인 도구 — 빈 칸이 없는 입력이면 BRITS 출력이 입력과 똑같은지, 그리고 BRITS 가 시간을 얼마나 쓰는지 잰다.

    python check_brits.py

왜 필요한가
  BRITS 는 빈 칸만 채우고 관측된 칸은 그대로 둔다(출력 = m·x + (1−m)·추정, m=1 이면 x).
  그래서 빈 칸이 없으면 출력이 입력과 같아야 한다. 이 도구는 그것을 원본 코드(models/biva_src)로 직접 확인한다.
  (2026-11-06 원본 복원 뒤에는 '빈 칸이 없으면 BRITS 건너뛰기'를 하지 않는다 — 원본처럼 매번 실행)
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.biva_src.BIVA import Model as BIVANet  # noqa: E402  원본 BIVA 신경망


def main() -> None:
    # 원본 노트북과 같은 크기: 배치 3 · 창 6개월 · 변수 34개
    a = SimpleNamespace(seq_len=6, pred_len=1, label_len=1, channels=34, rnn_hid_size=128, vae_latent_size=4, infer_hid_size=128,
                        conv_kernal=1, moving_avg=6, batch_size=3, conv1d=True, RIN=False, combination=True, imputed_weight=0.05)
    torch.manual_seed(0)
    net = BIVANet(a).eval()
    x = torch.rand(3, 6, 34)                 # 빈 칸 없는 입력
    with torch.no_grad():
        net.BRITS(x); net(x)                 # 처음 한 번은 준비 시간이 섞이므로 버림(워밍업)
        t = time.time(); [net.BRITS(x) for _ in range(5)]; t_brits = (time.time() - t) / 5
        out = net.BRITS(x)
        same = torch.equal(out["imputations"], x)
        t = time.time(); [net(x) for _ in range(5)]; t_all = (time.time() - t) / 5
    print(f"빈 칸 없는 입력에서 BRITS 출력 == 입력: {same}")
    print(f"BRITS 한 번 {t_brits * 1000:.1f}ms · BIVA 전체 한 번 {t_all * 1000:.1f}ms (BRITS 비중 {t_brits / t_all:.0%})")


if __name__ == "__main__":
    main()
