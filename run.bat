@echo off
chcp 65001 >nul
setlocal
set PYTHONIOENCODING=utf-8
set "ROOT=%~dp0"
set "PY=%ROOT%.venv-cuda\Scripts\python.exe"
if not exist "%PY%" set "PY=E:\Anaconda\envs\showo2\python.exe"
cd /d "%ROOT%"
set "SB_MODEL_DIR=%ROOT%models\Z-Image-Turbo"
if exist "%ROOT%models\Z-Image-Turbo-bf16\model_index.json" set "SB_MODEL_DIR=%ROOT%models\Z-Image-Turbo-bf16"

echo [StoryBoard Agent] Python: %PY%
echo [StoryBoard Agent] 权重目录: %SB_MODEL_DIR%
"%PY%" main.py --mode sequential %*
echo.
pause
endlocal
