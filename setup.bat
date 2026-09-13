@echo off
cd /d "%~dp0"

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  set PY=py -3
) else (
  set PY=python
)

echo Creating virtual environment...
%PY% -m venv .venv
if errorlevel 1 goto :fail
call .venv\Scripts\activate.bat

echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo Dataset...
python dataset.py
if errorlevel 1 goto :fail

echo Building RAG indexes. First time downloads all-MiniLM-L6-v2...
python build_index.py
if errorlevel 1 goto :fail

echo Running demos...
python run_demos.py
if errorlevel 1 goto :fail

echo Tests...
python -m pytest -q
if errorlevel 1 goto :fail

echo Ready check...
python check_ready.py
if errorlevel 1 goto :fail

echo.
echo Setup finished.
echo Port is 8001
echo Start the API with:  python run_server.py
echo Then open http://127.0.0.1:8001
echo Docs http://127.0.0.1:8001/docs
echo MCP  http://127.0.0.1:8001/mcp
echo In a second terminal:  python mcp_svc\client.py
goto :eof

:fail
echo Setup failed.
exit /b 1
