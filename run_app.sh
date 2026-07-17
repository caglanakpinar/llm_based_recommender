#!/usr/bin/env bash
# Launch Streamlit app with a clean Poetry environment.
set -euo pipefail
cd "$(dirname "$0")"

# Stale VIRTUAL_ENV (e.g. old .venv311) makes Poetry recreate an empty .venv.
unset VIRTUAL_ENV

if command -v deactivate >/dev/null 2>&1; then
  deactivate 2>/dev/null || true
fi

poetry env use python3.11 2>/dev/null || poetry env use /usr/local/bin/python3.11
poetry install
exec poetry run streamlit run app.py "$@"
