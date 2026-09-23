# Initial checkpoint evaluation

## Identity

- Date: 2026-09-22
- Model: Qwen3-4B base and full-parameter SFT checkpoints from run `20260918-180919`
- Checkpoints: 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000
- Evaluation code: temporary evaluator used during this run; Triton disabled with `TORCH_DISABLE_NATIVE_JIT=1`
- Manual editing set: 140 usable SFT records and 125 usable DPO preference records
- Important scope: this is the manually collected editing benchmark (E04), not the Stage-1 continuation validation split (E03).

## Results

| Model | Accepted-response NLL | DPO accuracy | Length-normalized margin |
|---|---:|---:|---:|
| Base | 6.3635 | 41.6% | -0.3195 |
| Step 1000 | 3.1923 | 41.6% | -0.3000 |
| Step 2000 | 3.1848 | 42.4% | -0.2949 |
| Step 3000 | 3.1769 | 43.2% | -0.2961 |
| Step 4000 | 3.1880 | 42.4% | -0.2907 |
| Step 5000 | 3.1910 | 43.2% | -0.2894 |
| Step 6000 | 3.1900 | 42.4% | -0.2893 |
| Step 7000 | 3.1909 | 42.4% | -0.2873 |
| Step 8000 | 3.1901 | 43.2% | -0.2906 |
| Step 9000 | 3.1897 | 42.4% | -0.2904 |

NLL is the mean per-response supervised-token NLL.  DPO accuracy is the
fraction where chosen has higher length-normalized log-probability than
rejected.  A positive margin would be ideal; all observed margins remain
negative.

## Interpretation boundary

The table supports a strong improvement on the accepted editing responses and
a small, noisy DPO improvement relative to base. It does not establish
generalization on unseen Stage-1 data, because the manual benchmark was
collected separately and was not split into an untouched test set. It also has
no per-source continuation/instruction breakdown or bootstrap interval yet.

## Reuse requirements

Future experiments should append a new report rather than overwrite this one,
and keep the same benchmark IDs, tokenizer/template, decoding settings, and
metric definitions. Each report should record the code commit, model paths,
dataset revision, sample counts, command, checkpoint/scheduler state, and any
excluded records.
