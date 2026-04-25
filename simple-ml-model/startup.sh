#!/bin/bash
set -e

PORT="${PORT:-8000}"

# Fallback dependency install in case build automation did not run.
if ! python -c "import streamlit" >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

exec python -m streamlit run app.py --server.port "${PORT}" --server.address 0.0.0.0 --server.headless true
