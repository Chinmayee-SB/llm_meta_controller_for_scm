#!/bin/sh
# Starts the live demo. Open http://localhost:8765 once it says "Live demo".
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Setting up the environment (first run only)..."
  python3 -m venv .venv && .venv/bin/pip install -q -r demo/requirements-demo.txt || exit 1
fi
exec .venv/bin/python demo/server.py
