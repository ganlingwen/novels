#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="qwen38-27b-sglang"
HEALTH_URL="http://127.0.0.1:8426/health"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
        echo "$CONTAINER_NAME is already running."
    else
        echo "Starting existing container: $CONTAINER_NAME"
        docker start "$CONTAINER_NAME" >/dev/null
    fi
else
    echo "Creating and starting: $CONTAINER_NAME"
    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --network host \
        --ipc host \
        --shm-size 16g \
        -e TRITON_CACHE_DIR=/root/.triton \
        -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0 \
        -e HF_HOME=/root/.cache/huggingface \
        -v /home/ganlingwen/.cache/huggingface:/root/.cache/huggingface \
        -v /home/ganlingwen/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/triton:/root/.triton \
        lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2 \
        python3 -m sglang.launch_server \
        --model-path RadixArk/Qwen3.8-27B-NVFP4 \
        --served-model-name nvidia/Qwen3.8-27B-NVFP4 \
        --trust-remote-code \
        --revision 554ebba9b5f1b79dc11246341960360e6ef4 \
        --mem-fraction-static 0.50 \
        --attention-backend flashinfer \
        --chunked-prefill-size 8192 \
        --disable-prefill-cuda-graph \
        --cuda-graph-max-bs 8 \
        --disable-flashinfer-autotune \
        --kv-cache-dtype fp8_e4m3 \
        --mamba-ssm-dtype bfloat16 \
        --mamba-full-memory-ratio 4.21 \
        --mamba-radix-cache-strategy extra_buffer \
        --max-mamba-cache-size 5 \
        --max-running-requests 1 \
        --max-total-tokens 262144 \
        --context-length 262144 \
        --speculative-algorithm DFLASH \
        --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2 \
        --speculative-draft-model-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
        --speculative-draft-model-quantization unquant \
        --speculative-num-draft-tokens 8 \
        --enable-torch-compile \
        --torch-compile-max-bs 4 \
        --num-continuous-decode-steps 2 \
        --sleep-on-idle \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --sampling-defaults model \
        --enable-metrics \
        --enable-cache-report \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        --host 0.0.0.0 \
        --port 8426
fi

echo "Waiting for SGLang health check..."
for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
        echo "Qwen is ready: http://127.0.0.1:8426/v1"
        echo "Model: nvidia/Qwen3.8-27B-NVFP4"
        exit 0
    fi
    sleep 5
done

echo "Timed out waiting for Qwen. Follow logs with:"
echo "docker logs -f $CONTAINER_NAME"
exit 1
