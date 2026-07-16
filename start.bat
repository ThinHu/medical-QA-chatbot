@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo ========================================================
echo       Khoi dong he thong ViMQ Chatbot (Local)
echo ========================================================
echo.

echo [1/3] Dang khoi dong AI Backend (gRPC Server)...
start "AI Backend" cmd /k "set HF_ENDPOINT=https://hf-mirror.com && .\venv\Scripts\activate.bat && python backend\grpc_server.py"

timeout /t 3 /nobreak > nul

echo [2/3] Dang khoi dong API Gateway (FastAPI)...
start "API Gateway" cmd /k ".\venv\Scripts\activate.bat && uvicorn backend.fastapi_app:app --port 8080"

timeout /t 2 /nobreak > nul

echo [3/3] Dang khoi dong Giao dien Nguoi dung (Chainlit)...
start "Chainlit UI" cmd /k ".\venv\Scripts\activate.bat && chainlit run frontend\app.py -w"

echo.
echo ========================================================
echo Hoan tat! 3 cua so moi da duoc mo cho cac dich vu.
echo Trinh duyet se tu dong mo http://localhost:8000
echo ========================================================
pause
