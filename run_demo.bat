@echo off
start "CropGuard Backend" "C:\Users\ANURAG BANERJEE\Documents\ComfyUI\.venv\Scripts\python.exe" -m uvicorn server.main:app --host 0.0.0.0 --port 8000
timeout /t 2 /nobreak >nul
start "" "frontend\index.html"