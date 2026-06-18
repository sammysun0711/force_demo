#!/usr/bin/env bash
# Start Jupyter so hipblaslt_offline_tuning.ipynb opens in a browser (local or over the network).
# Usage: ./launch_hipblaslt_demo.sh
#        NOTEBOOK_PORT=9999 ./launch_hipblaslt_demo.sh
set -euo pipefail

cd "$(dirname "$0")"
: "${NOTEBOOK_PORT:=9999}"
: "${NOTEBOOK_IP:=0.0.0.0}"

NOTEBOOK="hipblaslt_offline_tuning.ipynb"
if [[ ! -f "${NOTEBOOK}" ]]; then
  echo "error: ${NOTEBOOK} not found in $(pwd)" >&2
  exit 1
fi

if command -v jupyter >/dev/null 2>&1; then
  JPY=(jupyter notebook)
elif command -v python3 >/dev/null 2>&1 && python3 -m jupyter notebook --help >/dev/null 2>&1; then
  JPY=(python3 -m jupyter notebook)
else
  echo "error: install Jupyter first, e.g. pip install notebook" >&2
  exit 1
fi

echo "Starting Jupyter Notebook…"
echo "  directory: $(pwd)"
echo "  notebook:  ${NOTEBOOK}"
echo "  listen:    http://${NOTEBOOK_IP}:${NOTEBOOK_PORT}/"
echo "  (open the URL below in your browser; token auth is disabled for this launcher)"
echo

exec "${JPY[@]}" \
  --ip="${NOTEBOOK_IP}" \
  --port="${NOTEBOOK_PORT}" \
  --no-browser \
  --allow-root \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  "${NOTEBOOK}"
