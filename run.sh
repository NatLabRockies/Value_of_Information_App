#!/usr/bin/env bash
# Launch the NLR VOI Streamlit app on http://localhost:8600
# Two workflows in one, toggled in the sidebar:
#   Main VOI App         -> app.py          (binary decision / Naive Bayes)
#   Sequential Decisions -> eig_app.py      (Expected Information Gain planner)
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

# Needs Python 3.11+; set PYTHON=/path/to/python3 to pick a specific interpreter.
PYTHON="${PYTHON:-python3}"

if [ ! -x .venv/bin/streamlit ]; then
  echo "No virtualenv found. Creating one and installing dependencies..."
  "$PYTHON" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/streamlit run voi_app.py \
  --server.port 8600 \
  --server.enableXsrfProtection false \
  --server.maxUploadSize 400
