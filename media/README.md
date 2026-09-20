# Novel media generation

The scripts in `scripts/` use the ComfyUI checkout at
`~/repos/ComfyUI` as a local inference runtime. Wan model files live under
`models/wan2.1/` and generated videos are written to `../generated/videos/`.

`scripts/generate_wan_video.sh` starts a temporary, memory-limited ComfyUI
worker, submits the prompt through its localhost API, waits for completion, and
stops the worker. The browser UI and a persistent server are not required.

Run the local workflow test with:

```bash
python -m unittest media.tests.test_wan_t2v
```

Generate a default-quality example with:

```bash
media/scripts/generate_wan_video.sh
```

Override settings with named options:

```bash
media/scripts/generate_wan_video.sh \
  --prompt "一个天才青年在破败街道上奔跑，前景只有一名穿鲜红色防水夹克的成年男子，脸部清晰可见，四名丧尸在身后追赶，HDR电影画面，正常曝光，明亮通透，丰富鲜明的色彩，雨水反光，真实自然的人体动作" \
  --width 832 --height 480 --frames 33 --steps 30
```

The client displays a live sampling bar when run with the ComfyUI virtual
environment. It also reports the VAE decoding and video-saving phases, which
can take time without changing the sampling counter.

Resolution is a quality/performance tradeoff: width and height increase both
spatial detail and memory use, while the number of frames increases temporal
length and memory use. For a lower-resolution but still useful landscape test,
use `--width 640 --height 368`; keep the dimensions divisible by 16. Lowering
resolution can improve stability and speed, but it cannot recover detail that
the model was never asked to generate. More sampling steps may improve
coherence, at a much higher runtime cost.
