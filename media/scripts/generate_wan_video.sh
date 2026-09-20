#!/usr/bin/env bash
set -euo pipefail

NOVEL_DIR=$(cd "$(dirname "$0")/../.." && pwd)
COMFY_DIR=${COMFY_DIR:-/home/ganlingwen/repos/ComfyUI}
PYTHON=${PYTHON:-/home/ganlingwen/venv/bin/python}
MODEL_PATHS="$NOVEL_DIR/media/config/extra_model_paths.yaml"
CLIENT=$NOVEL_DIR/media/scripts/wan_t2v.py

if curl -fsS --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    exec "$PYTHON" "$CLIENT" "$@"
fi

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 30 * 1024 * 1024 )); then
    echo "Refusing to start Wan: less than 30 GiB system memory is available." >&2
    exit 1
fi

unit="wan-api-$$"
cleanup() {
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

systemd-run --user --unit="$unit" \
    --property=MemoryMax=24G \
    --property=MemorySwapMax=4G \
    --working-directory="$COMFY_DIR" \
    "$PYTHON" main.py --listen 127.0.0.1 --port 8188 --novram --cpu-vae --disable-smart-memory \
        --output-directory "$NOVEL_DIR/generated/videos" \
        --extra-model-paths-config "$MODEL_PATHS"

for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
        "$PYTHON" "$CLIENT" "$@"
        exit
    fi
    sleep 1
done

echo "ComfyUI worker did not become ready." >&2
exit 1
