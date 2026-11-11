#!/bin/bash
# 클라우드 작업용: API 재시작 (노트북에서는 run_server.bat)
cd "$(dirname "$0")"
for pid in $(pgrep -f "uvicorn api.main"); do [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null; done
sleep 1
nohup python -m uvicorn api.main:app --port 8000 > /tmp/api.log 2>&1 &
sleep 8
curl -s -o /dev/null -w "API %{http_code}\n" http://127.0.0.1:8000/api/results
