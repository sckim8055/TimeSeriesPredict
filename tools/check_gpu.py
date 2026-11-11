"""GPU 점검 — 노트북에서 실행해 결과를 그대로 붙여 주세요.  python tools/check_gpu.py"""
import platform
import subprocess
import sys

print("python", sys.version.split()[0], platform.platform())
try:
    import torch

    print("torch", torch.__version__, "| CUDA 빌드", torch.version.cuda)
    print("cuda.is_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| VRAM %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
        x = torch.randn(4096, 4096, device="cuda"); y = (x @ x).sum().item(); print("GPU 계산 테스트 OK")
    else:
        print("→ CPU 전용 torch 입니다. LSTM·BIVA 가 매우 느립니다.")
except Exception as e:  # noqa: BLE001
    print("torch 없음/오류:", e)
try:
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]).decode().strip())
except Exception as e:  # noqa: BLE001
    print("nvidia-smi 실행 불가:", e)
