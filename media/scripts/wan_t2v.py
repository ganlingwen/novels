#!/usr/bin/env python3
"""Submit a Wan 2.1 T2V 1.3B text-to-video job to a running ComfyUI instance."""
import json
import urllib.request
import urllib.error
import sys
import time

HOST = "http://127.0.0.1:8188"
DEFAULT_PROMPT = "一个天才青年，名字叫干灵文。在破败的街道上，在灰暗的天空下，在濛濛的细雨中，他被一群丧尸追逐着。他惊慌地向镜头方向奔跑，衣服和头发被雨水打湿，身后的丧尸快速追赶。电影感，写实风格，低饱和度，动态跟拍，人物动作自然，雨雾氛围，细节清晰。"

def build_workflow(prompt, width=832, height=480, frames=33, steps=20):
    return {
        "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "Wan2_1-T2V-1_3B_bf16.safetensors",
            "weight_dtype": "default",
        },
    },
        "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
    },
        "3": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
    },
        "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": ["2", 0]},
    },
        "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted, watermark, text",
            "clip": ["2", 0],
        },
    },
        "6": {
        "class_type": "EmptyHunyuanLatentVideo",
        "inputs": {
            "width": width,
            "height": height,
            "length": frames,
            "batch_size": 1,
        },
    },
        "8": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": 42,
            "steps": steps,
            "cfg": 6.0,
            "sampler_name": "uni_pc",
            "scheduler": "simple",
            "denoise": 1.0,
        },
    },
        "9": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
    },
        "10": {
        "class_type": "SaveWEBM",
        "inputs": {
            "images": ["9", 0],
            "filename_prefix": "wan_t2v",
            "codec": "vp9",
            "fps": 16.0,
            "crf": 32.0,
        },
    },
    }

def generate(prompt, width=832, height=480, frames=33, steps=20):
    workflow = build_workflow(prompt, width, height, frames, steps)
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{HOST}/prompt", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        result = json.load(r)
    prompt_id = result["prompt_id"]
    print("QUEUED prompt_id:", prompt_id)
    print(f"  {width}x{height}, {frames} frames, {steps} steps")
    print(f"  prompt: {prompt}")

    while True:
        with urllib.request.urlopen(f"{HOST}/history/{prompt_id}") as r:
            history = json.load(r).get(prompt_id)
        if history is not None:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status}")
            if status.get("completed"):
                for output in history.get("outputs", {}).values():
                    for video in output.get("images", []):
                        print("OUTPUT:", video.get("filename"))
                break
        time.sleep(2)


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 832
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 33
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    try:
        generate(prompt, width, height, frames, steps)
    except urllib.error.HTTPError as e:
        print("ERROR", e.code)
        print(e.read().decode()[:2000])


if __name__ == "__main__":
    main()
