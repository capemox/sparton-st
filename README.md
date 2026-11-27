# Sparton for Sentence Transformers

Sparton fuses the masked-language-model vocabulary projection and SPLADE max pooling on NVIDIA GPUs. The kernel reduces sequence tiles before writing results, avoiding the full `[batch, tokens, vocabulary]` logits tensor.

## Usage

Install from the source checkout with `pip install .`.

```python
from sentence_transformers import SparseEncoder
from sparton_st import kernelize

model = SparseEncoder("naver/splade-v3", device="cuda")
model = kernelize(model)
embeddings = model.encode(["What is sparse retrieval?"])
```

Or enable the same conversion during loading with the package's `SparseEncoder` subclass.

```python
from sparton_st import SparseEncoder

model = SparseEncoder("naver/splade-v3", device="cuda", use_sparton=True)
```

`kernelize()` modifies the supplied encoder in place and returns it. The original model owns its parameters, tokenizer, and pooling configuration. `save_pretrained()` produces a regular Sentence Transformers checkpoint. Apply `kernelize()` again after loading it to enable the kernel.

## Compatibility

- Sentence Transformers 5.x and 6.x, including the legacy `MLMTransformer` and `Transformer(transformer_task="fill-mask")`.
- Two-module SPLADE encoders with the built-in max `SpladePooling`, using BERT, RoBERTa, XLM-RoBERTa, DistilBERT, or ModernBERT. The adapter handles both SPLADE activation variants. Other pipeline layouts and sum pooling raise a conversion error.
- The bundled fused path runs with CUDA PyTorch and Triton on Linux, using an NVIDIA SM80+ GPU and float32, float16, or bfloat16 parameters. CPU calls, training, calls with gradients or CUDA autocast enabled, and packed (unpadded) attention use the native modules.
- `encode()`, `encode_query()`, and `encode_document()` use the regular Sentence Transformers API. On fused calls, low-level `forward()` returns `sentence_embedding` without the per-token vocabulary logits.

On an RTX 4060 Laptop GPU, a synthetic decoder-and-pooling benchmark with batch size 8, 128 tokens, hidden size 768, and vocabulary size 30,522 reduced median FP16 latency from 4.25 ms to 1.70 ms and peak temporary CUDA tensor allocation from about 122 MiB to 1.4 MiB. FP32 latency was about 10.2 ms for both paths, with peak temporary allocation falling from about 243 MiB to 4.7 MiB.

## Hugging Face Kernels

The bundled Triton kernel is used by default. To load a published Sparton v1 kernel, install with `pip install '.[hub]'` and pass its repository ID. A trusted publisher is required by default. When you trust another publisher, opt in with `trust_kernel_code=True`.

```python
model = SparseEncoder("naver/splade-v3", device="cuda")
model = kernelize(
    model,
    kernel_repo="your-org/sparton",
    kernel_revision="your-kernel-commit-sha",
)
```

Omitting `kernel_revision` selects the v1 release. Loading happens on the first eligible CUDA forward. Legacy environments pinned to `huggingface-hub<1` can use the bundled kernel. The Hub module exposes `SPARTON_API_VERSION = 1` and `sparton_forward(hidden, weight, bias, attention_mask, *, double_log=False)`. See the [kernel card](kernel/CARD.md) for tensor shapes and dtypes.

The [kernel source](kernel/) is shared by the Python package and Hub artifact. From a committed checkout, build with `kernel-builder build-and-copy kernel`. Publish with `kernel-builder build-and-upload kernel --repo-id your-org/sparton`. See the [Hugging Face publishing guide](https://huggingface.co/docs/kernels/builder/build).

## Tests

Run `pip install -e '.[test]'` and `python -m pytest`. The tests create small local checkpoints. CUDA tests compare the fused output against Sentence Transformers and PyTorch.

Based on [Sparton: Fast and Memory-Efficient Triton Kernel for Learned Sparse Retrieval](https://arxiv.org/abs/2603.25011) by Nguyen et al.
