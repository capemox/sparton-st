---
library_name: kernels
license: apache-2.0
---

# Sparton

Sparton fuses the SPLADE decoder projection, attention masking and token-wise max reduction into Triton tiles. It then applies the SPLADE activation to the pooled vocabulary scores.

```python
from kernels import get_kernel

kernel = get_kernel("{{ repo_id }}", version={{ version }}, backend="cuda")
scores = kernel.sparton_forward(hidden, weight, bias, attention_mask)
```

`hidden` has shape `[batch, tokens, hidden_size]`, the decoder `weight` has shape `[vocab, hidden_size]`, and `attention_mask` has shape `[batch, tokens]`. `bias` can be `None`. The result has shape `[batch, vocab]`. If your SPLADE model applies a second `log1p`, pass `double_log=True`.

The v1 API supports inference on NVIDIA CUDA devices with compute capability 8.0 or later, using float16, bfloat16 or float32. The attention mask takes integer or Boolean values.

Based on the algorithm in [Sparton: Fast and Memory-Efficient Triton Kernel for Learned Sparse Retrieval](https://arxiv.org/abs/2603.25011).
