@echo off
REM Make narration audio for the code walkthrough screens (ElevenLabs). See tools\narrate.py.
REM First time: put two lines in .env -> ELEVENLABS_API_KEY=...  and  ELEVENLABS_VOICE_ID=...
REM Usage: narrate.bat ridge          (several: narrate.bat ridge lgbm)
REM        narrate.bat --list-voices  (your voice names and IDs)
REM        narrate.bat ridge --dry-run (characters needed and credits left, nothing is made)
chcp 65001 >nul
set PY=C:\Users\test\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PY%" set PY=python
cd /d %~dp0
"%PY%" tools\narrate.py %*
pause
