# Stage-2 constant learning-rate comparison

## Results

| Model | Set | Chosen NLL | Rejected NLL | Norm. accuracy | Norm. margin | Raw accuracy |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 | all 151 | 3.1099 | 2.9903 | 44.37% | -0.2492 | 37.75% |
| Stage 2 cosine SFT | all 151 | 3.0732 | 2.9523 | 44.37% | -0.2539 | 37.09% |
| Stage 2 constant SFT | all 151 | 2.0483 | 2.5623 | 71.52% | 0.4350 | 58.28% |
| Stage 2 cosine DPO | all 151 | 3.0667 | 2.9516 | 44.37% | -0.2494 | 37.75% |
| Stage 2 constant DPO | all 151 | 2.7681 | 6.2463 | 98.01% | 4.0538 | 94.04% |
| Stage 1 | valid 15 | 2.9111 | 2.4712 | 26.67% | -0.5923 | 33.33% |
| Stage 2 cosine SFT | valid 15 | 2.8813 | 2.4450 | 20.00% | -0.6032 | 33.33% |
| Stage 2 constant SFT | valid 15 | 1.9940 | 2.2528 | 60.00% | 0.1096 | 60.00% |
| Stage 2 cosine DPO | valid 15 | 2.8808 | 2.4425 | 13.33% | -0.6059 | 33.33% |
| Stage 2 constant DPO | valid 15 | 3.6647 | 4.8074 | 86.67% | 1.3277 | 73.33% |

## Identity

- Date: 2026-09-22
- Branch: `feat/stage2-constant-lr`
- Base model: Stage-1 final from `llm/outputs/qwen3-4b-novel-sft/runs/20260918-180919/final`
- Dataset: 151 usable DPO pairs, deterministic 136/15 train/validation split
- Hardware/runtime: CUDA, bf16, Triton/native JIT disabled with `TORCH_DISABLE_NATIVE_JIT=1`
- Machine-readable output: `llm/constant_lr_comparison.json`

## Variable and commands

The experiment adds a selectable constant scheduler and retrains both stages
with `lr=1e-5`. Batch size, gradient accumulation, beta, step counts, data,
model initialization, and validation split remain unchanged from the current
Stage-2 runs. The baseline uses cosine scheduling with SFT `1e-6` and DPO
`5e-7`.

- SFT: 50 steps, initialized from Stage-1 final, batch 1, accumulation 16.
- DPO: 100 steps, initialized from the new constant-LR SFT final, beta 0.1,
  accumulation 16, with that same SFT final as the frozen reference.
- Both new runs save only one final model.

The new SFT run reported final validation loss `2.868023`; the old cosine SFT
run reported `3.355691`. The new DPO run reported validation DPO loss
`0.600982`, raw preference accuracy `73.33%`, and raw DPO margin `41.310229`.

## Metric definitions and scope

NLL is token-weighted mean negative log-likelihood over the response tokens.
Normalized accuracy and margin compare chosen and rejected mean token
log-probabilities per response; positive margin is preferred. Raw accuracy is
the same comparison using summed response log-probabilities. The all-151 table
is retrospective and overlaps training data, so the 15-pair validation rows
are the relevant generalization check.

The constant-LR run improves every reported validation preference metric over
the cosine baseline. However, 15 validation pairs are too few for a stable
claim about broad writing quality; the result supports adopting this learning
rate/schedule for the next controlled experiment, not a final model-quality
conclusion.

## Reproduction

```bash
TORCH_DISABLE_NATIVE_JIT=1 llm/.venv/bin/python llm/evaluate_stage2.py \
  --base-model llm/outputs/qwen3-4b-novel-sft/runs/20260918-180919/final \
  --data-dir data --output llm/constant_lr_comparison.json \
  stage1=llm/outputs/qwen3-4b-novel-sft/runs/20260918-180919/final \
  cosine_sft=llm/outputs/qwen3-4b-novel-stage2-sft/runs/20260922-173708/final \
  cosine_dpo=llm/outputs/qwen3-4b-novel-stage2-dpo/runs/20260922-181413/final \
  constant_sft=llm/outputs/qwen3-4b-novel-stage2-sft-constant-lr/runs/20260922-191308/final \
  constant_dpo=llm/outputs/qwen3-4b-novel-stage2-dpo-constant-lr/runs/20260922-192040/final
```

## Next experiment

Keep the constant scheduler and `1e-5` as the current promising setting, then
repeat with a larger untouched preference validation set before changing the
objective or adding more training steps.
