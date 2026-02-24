#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  local code=$?
  trap - INT TERM EXIT

  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi

  wait 2>/dev/null || true
  exit "${code}"
}

trap cleanup INT TERM EXIT

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "No python found. Activate your conda/venv environment first."
  exit 1
fi
echo "Using Python: ${PYTHON}"

if ! "${PYTHON}" -c "import uvicorn" >/dev/null 2>&1; then
  echo "Missing 'uvicorn' in: ${PYTHON}"
  echo "Install backend deps first: ${PYTHON} -m pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "${ROOT_DIR}/frontend/node_modules" ]]; then
  echo "Missing frontend/node_modules"
  echo "Install frontend deps first: cd frontend && npm install"
  exit 1
fi

echo "Starting backend on http://localhost:${BACKEND_PORT} ..."
(
  cd "${ROOT_DIR}"
  exec "${PYTHON}" -m uvicorn api.main:app --host 0.0.0.0 --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

sleep 1
if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
  echo "Backend failed to start."
  exit 1
fi

echo "Starting frontend on http://localhost:${FRONTEND_PORT} ..."
(
  cd "${ROOT_DIR}/frontend"
  exec npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

sleep 1
if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
  echo "Frontend failed to start."
  exit 1
fi

echo
echo "All services are up:"
echo "  Backend : http://localhost:${BACKEND_PORT}"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo "Press Ctrl+C to stop both."
echo

wait
