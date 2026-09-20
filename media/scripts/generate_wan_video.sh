#!/usr/bin/env bash
set -euo pipefail

NOVEL_DIR=$(cd "$(dirname "$0")/../.." && pwd)
COMFY_DIR=${COMFY_DIR:-/home/ganlingwen/repos/ComfyUI}
PYTHON=${PYTHON:-/home/ganlingwen/venv/bin/python}
MODEL_PATHS="$NOVEL_DIR/media/config/extra_model_paths.yaml"
CLIENT=$NOVEL_DIR/media/scripts/wan_t2v.py

DEFAULT_PROMPT="一个天才青年，名字叫干灵文。在破败的街道上，灰暗的暴雨云压在城市上空，濛濛细雨落在湿漉漉的路面上。前景只有一名成年男子，他有短黑发，穿着鲜红色防水夹克，清晰可辨的脸正面朝向镜头，正在向镜头奔跑；四名丧尸在他身后追赶。HDR电影画面，正常曝光，主体脸部和衣服清晰锐利，冷色调但明亮通透，丰富鲜明而自然的色彩，雨水反光，动态跟拍，真实自然的人体动作，高质量。"
PROMPT=$DEFAULT_PROMPT
WIDTH=832
HEIGHT=480
FRAMES=33
STEPS=30

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --prompt TEXT   Text prompt (default: built-in HDR rainy chase prompt)
  --width N       Video width, divisible by 16 (default: 832)
  --height N      Video height, divisible by 16 (default: 480)
  --frames N      Number of frames (default: 33)
  --steps N       Sampling steps (default: 30)
  -h, --help      Show this help
EOF
}

while (($#)); do
    case "$1" in
        --prompt)
            (($# >= 2)) || { echo "--prompt requires a value" >&2; exit 2; }
            PROMPT=$2
            shift 2
            ;;
        --width|--height|--frames|--steps)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            value=$2
            [[ $value =~ ^[0-9]+$ ]] || { echo "$1 must be a positive integer" >&2; exit 2; }
            case "$1" in
                --width) WIDTH=$value ;;
                --height) HEIGHT=$value ;;
                --frames) FRAMES=$value ;;
                --steps) STEPS=$value ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

(( WIDTH > 0 && WIDTH % 16 == 0 )) || { echo "--width must be a positive multiple of 16" >&2; exit 2; }
(( HEIGHT > 0 && HEIGHT % 16 == 0 )) || { echo "--height must be a positive multiple of 16" >&2; exit 2; }
(( FRAMES > 0 )) || { echo "--frames must be positive" >&2; exit 2; }
(( STEPS > 0 )) || { echo "--steps must be positive" >&2; exit 2; }

if curl -fsS --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    exec "$PYTHON" "$CLIENT" "$PROMPT" "$WIDTH" "$HEIGHT" "$FRAMES" "$STEPS"
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
        "$PYTHON" "$CLIENT" "$PROMPT" "$WIDTH" "$HEIGHT" "$FRAMES" "$STEPS"
        exit
    fi
    sleep 1
done

echo "ComfyUI worker did not become ready." >&2
exit 1
