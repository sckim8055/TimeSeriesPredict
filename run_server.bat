@echo off
REM Start the web UI server. Open http://127.0.0.1:8000 in a browser.
REM First time only: %PY% -m pip install -r requirements.txt
REM PY = the Python that has torch/pykrx installed (not anaconda).
set PY=C:\Users\test\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PY%" set PY=python
cd /d %~dp0
"%PY%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
