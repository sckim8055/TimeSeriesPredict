@echo off
rem ======================================================================
rem  TimeSeriesPredict - first upload to GitHub (run ONCE)
rem  Before running:
rem    1) On github.com create an EMPTY PRIVATE repository (no README).
rem    2) Edit the two lines below: GH_USER and GH_REPO.
rem  Login: a browser window (Git Credential Manager) opens on first push.
rem         Never paste a token into chat.
rem ======================================================================
set GH_USER=YOUR_GITHUB_ID
set GH_REPO=TimeSeriesPredict

cd /d %~dp0
where git >nul 2>nul || (echo [ERROR] git is not installed. Install from https://git-scm.com/download/win & pause & exit /b 1)
if "%GH_USER%"=="YOUR_GITHUB_ID" (echo [ERROR] edit GH_USER in this file first & pause & exit /b 1)

rem --- safety: .env must never be committed ---
if not exist .gitignore (echo [ERROR] .gitignore missing & pause & exit /b 1)

if not exist .git git init -b main
git config user.name >nul 2>nul || git config user.name "%GH_USER%"
git config user.email >nul 2>nul || git config user.email "%GH_USER%@users.noreply.github.com"

git add -A
git status --short | findstr /r /c:"\.env$" >nul && (echo [ERROR] .env is staged - aborting & git reset & pause & exit /b 1)
git commit -m "Initial import: engine, models, web, docs, data versions, results"
git remote remove origin >nul 2>nul
git remote add origin https://github.com/%GH_USER%/%GH_REPO%.git
git push -u origin main
echo.
echo Done. Check https://github.com/%GH_USER%/%GH_REPO% (must say Private).
pause
