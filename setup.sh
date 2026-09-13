#!/bin/sh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dataset.py
python build_index.py
python run_demos.py
python -m pytest -q
python check_ready.py
echo
echo Setup finished.
echo Port is 8001
echo Start the API with: python run_server.py
echo Then open http://127.0.0.1:8001
