#!/usr/bin/env bash
# Run with: bash run.sh   (first run creates a local Python environment and installs the packages)
set -e
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else
    echo "Python 3.9 or newer is required: https://www.python.org/downloads/"
    exit 1
fi

if [ ! -f .venv/setup_done ]; then
    echo "First run: setting up the environment. This takes a few minutes..."
    [ -x .venv/bin/python ] || "$PY" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip --quiet
    .venv/bin/python -m pip install -r requirements.txt --quiet
    touch .venv/setup_done
fi

.venv/bin/python main.py "$@"
echo
echo "Tables are in 'reports/results' and figures in 'reports/figures'."
