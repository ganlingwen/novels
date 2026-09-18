"""Check that omitting right-padding masks preserves supervised loss/gradients."""
import os
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def test_causal_padding():
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, attention_dropout=0.0,
    ))
    model.config._attn_implementation = "sdpa"
    ids = torch.randint(1, 64, (2, 12))
    ids[0, 7:] = 0
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[:, :3] = -100
    labels[mask == 0] = -100
    reference = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    reference.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual = model(input_ids=ids, labels=labels).loss
    actual.backward()
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    for name, param in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(param.grad, grads[name], atol=1e-6, rtol=1e-4)
    print("Right-padding loss and all parameter gradients agree.")


if __name__ == "__main__":
    test_causal_padding()
