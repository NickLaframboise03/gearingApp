#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  pip install -r requirements.txt
fi

PORT="${PORT:-25564}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"

if python -c "import gunicorn" >/dev/null 2>&1; then
  exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers "${WEB_CONCURRENCY}" \
    --threads "${GUNICORN_THREADS}" \
    --worker-class gthread \
    --timeout "${TIMEOUT_SECONDS}" \
    --keep-alive 5 \
    --access-logfile - \
    app:server
fi

echo "gunicorn is not installed; falling back to python app.py" >&2
exec python app.py
