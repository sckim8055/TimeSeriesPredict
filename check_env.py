"""실행 환경 점검.  python check_env.py
Python·PyTorch·CUDA·GPU 상태와, GPU 를 못 쓰는 경우 해결 방법을 출력한다."""
import platform
import sys


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"실행파일 : {sys.executable}")
    ok = True
    try:
        import torch
    except ImportError:
        print("PyTorch : 설치 안 됨 → pip install torch")
        return 1
    print(f"PyTorch : {torch.__version__}")
    cuda_build = torch.version.cuda
    if cuda_build is None:
        print("CUDA    : 이 PyTorch 는 CPU 전용 빌드입니다 (PyPI 기본 Windows wheel).")
        print("  해결 → CUDA 빌드로 교체 (CUDA 12.x 드라이버 기준):")
        print("    python -m pip uninstall -y torch")
        print("    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("  (드라이버가 오래됐으면 cu126, 최신이면 cu130. `nvidia-smi` 우상단 CUDA Version 이 그 이상이어야 함)")
        ok = False
    else:
        print(f"CUDA    : 빌드 {cuda_build}, 사용 가능 = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            print(f"GPU     : {props.name}, 메모리 {props.total_memory / 2**30:.1f} GB, 연산능력 {props.major}.{props.minor}")
            x = torch.randn(2000, 2000, device="cuda")
            torch.cuda.synchronize()
            import time

            t = time.time()
            for _ in range(20):
                x @ x
            torch.cuda.synchronize()
            print(f"GPU 행렬곱 20회: {time.time() - t:.3f}s (정상이면 0.1s 안팎)")
        else:
            print("  CUDA 빌드인데 GPU 를 못 찾음 → NVIDIA 드라이버 설치/업데이트 확인 (nvidia-smi 실행)")
            ok = False
    for name in ("lightgbm", "pandas", "statsmodels", "sklearn", "pyarrow", "pydantic"):
        try:
            m = __import__(name)
            print(f"{name:11s}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{name:11s}: 없음")
            ok = False
    print("\n결과:", "GPU 사용 가능 — BIVA·LSTM 은 자동으로 GPU 에서 돈다" if ok else "위 항목을 해결한 뒤 다시 실행")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
