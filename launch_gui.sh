#!/usr/bin/env bash
# Launch the StructureForge web GUI (Linux / macOS).
# Opens http://127.0.0.1:8000 in the default browser once the server is ready.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- ensure the [api] extras are installed -----------------------------------
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "[structureforge] installing API dependencies..."
    pip install -e ".[api]" --quiet
fi

# --- open browser once the server accepts connections ------------------------
(
    for i in $(seq 1 30); do
        sleep 0.5
        if curl -sf http://127.0.0.1:8000 >/dev/null 2>&1; then
            if command -v xdg-open >/dev/null 2>&1; then
                xdg-open http://127.0.0.1:8000
            elif command -v open >/dev/null 2>&1; then
                open http://127.0.0.1:8000
            fi
            break
        fi
    done
) &

echo "[structureforge] starting server at http://127.0.0.1:8000  (Ctrl+C to stop)"
python -m structureforge.api.cli "$@"
