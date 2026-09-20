#!/usr/bin/env python3
"""Submit a Wan 2.1 T2V 1.3B text-to-video job to a running ComfyUI instance."""
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

try:
    import aiohttp
    from tqdm import tqdm
except ImportError:
    aiohttp = None
    tqdm = None

HOST = "http://127.0.0.1:8188"
DEFAULT_PROMPT = "一个天才青年，名字叫干灵文。在破败的街道上，灰暗的暴雨云压在城市上空，濛濛细雨落在湿漉漉的路面上。前景只有一名成年男子，他有短黑发，穿着鲜红色防水夹克，清晰可辨的脸正面朝向镜头，正在向镜头奔跑；四名丧尸在他身后追赶。HDR电影画面，正常曝光，主体脸部和衣服清晰锐利，冷色调但明亮通透，丰富鲜明而自然的色彩，雨水反光，动态跟拍，真实自然的人体动作，高质量。"
DEFAULT_NEGATIVE_PROMPT = "dark, underexposed, black silhouette, ghost, translucent, spectral, motion blur, out of focus, blurry, low quality, distorted anatomy, deformed face, distorted hands, extra people, watermark, text"

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
            "text": DEFAULT_NEGATIVE_PROMPT,
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

def _print_outputs(history):
    for output in history.get("outputs", {}).values():
        for video in output.get("images", []):
            print("OUTPUT:", video.get("filename"))


def _wait_for_history(prompt_id):
    while True:
        with urllib.request.urlopen(f"{HOST}/history/{prompt_id}") as r:
            history = json.load(r).get(prompt_id)
        if history is not None:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status}")
            if status.get("completed"):
                _print_outputs(history)
                return
        time.sleep(2)


async def _generate_with_progress(prompt, width, height, frames, steps):
    workflow = build_workflow(prompt, width, height, frames, steps)
    client_id = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=None)

    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.ws_connect(f"{HOST}/ws?clientId={client_id}") as socket,
    ):
            async with session.post(
                f"{HOST}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            ) as response:
                response.raise_for_status()
                prompt_id = (await response.json())["prompt_id"]

            print("QUEUED prompt_id:", prompt_id)
            print(f"  {width}x{height}, {frames} frames, {steps} steps")
            print(f"  prompt: {prompt}")

            bar = tqdm(total=steps, unit="step", desc="Waiting", dynamic_ncols=True) if tqdm else None
            try:
                while True:
                    try:
                        message = await socket.receive(timeout=2)
                    except asyncio.TimeoutError:
                        message = None

                    if message is not None and message.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(message.data)
                        data = event.get("data", {})
                        if data.get("prompt_id") not in (None, prompt_id):
                            continue
                        if event.get("type") == "progress" and bar:
                            bar.set_description("Sampling")
                            bar.total = data.get("max", steps)
                            bar.n = data.get("value", 0)
                            bar.refresh()
                        elif event.get("type") == "executing" and bar:
                            node = data.get("node")
                            descriptions = {
                                "9": "Decoding VAE",
                                "10": "Saving video",
                            }
                            if node in descriptions:
                                bar.set_description(descriptions[node])
                            elif node == "8":
                                bar.set_description("Sampling")
                            elif node is None:
                                bar.set_description("Finishing")

                    async with session.get(f"{HOST}/history/{prompt_id}") as response:
                        history = (await response.json()).get(prompt_id)
                    if history is not None:
                        status = history.get("status", {})
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"ComfyUI generation failed: {status}")
                        if status.get("completed"):
                            if bar:
                                bar.set_description("Complete")
                                bar.n = bar.total
                                bar.refresh()
                            _print_outputs(history)
                            return
            finally:
                if bar:
                    bar.close()


def generate(prompt, width=832, height=480, frames=33, steps=20):
    if aiohttp is not None:
        asyncio.run(_generate_with_progress(prompt, width, height, frames, steps))
        return

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
    _wait_for_history(prompt_id)


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
