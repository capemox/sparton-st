"""Sparton projection and max pooling for padded SPLADE inputs."""

from __future__ import annotations

import torch

SPARTON_API_VERSION = 1
__all__ = ["SPARTON_API_VERSION", "sparton_forward"]

_PROJECTION_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
_MASK_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def sparton_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    attention_mask: torch.Tensor,
    *,
    double_log: bool = False,
) -> torch.Tensor:
    """Return SPLADE scores of shape (batch, vocabulary).

    The decoder weight is in PyTorch Linear layout (vocabulary, hidden size).
    Inputs must be on the same NVIDIA CUDA device (SM80+) and may be strided.
    """
    if hidden.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "Expected hidden [batch, tokens, hidden] and weight [vocab, hidden]."
        )
    batch, tokens, width = hidden.shape
    vocab, decoder_width = weight.shape
    if not tokens or not width or not vocab or width != decoder_width:
        raise ValueError("Invalid sequence or decoder dimensions.")
    if (
        attention_mask.shape != (batch, tokens)
        or attention_mask.dtype not in _MASK_DTYPES
    ):
        raise ValueError(
            "Expected an integer or Boolean attention mask of shape [batch, tokens]."
        )
    if hidden.dtype not in _PROJECTION_DTYPES or weight.dtype != hidden.dtype:
        raise ValueError(
            "Hidden states and decoder weight must share an fp16, bf16 or fp32 dtype."
        )
    if bias is not None and (bias.shape != (vocab,) or bias.dtype != weight.dtype):
        raise ValueError(
            "Decoder bias must have shape [vocab] and match the decoder weight dtype."
        )
    tensors = (
        (hidden, weight, attention_mask)
        if bias is None
        else (hidden, weight, attention_mask, bias)
    )
    if any(t.device != hidden.device for t in tensors):
        raise ValueError("All Sparton inputs must be on the same device.")
    if hidden.device.type != "cuda" or torch.version.cuda is None:
        raise ValueError("Sparton requires an NVIDIA CUDA device.")
    if torch.cuda.get_device_capability(hidden.device)[0] < 8:
        raise ValueError("Sparton requires NVIDIA compute capability 8.0 or later.")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise ValueError("The fused Sparton kernel supports inference only.")
    if batch == 0:
        return hidden.new_empty((0, vocab))

    # Keep module import independent of Triton and GPU initialization for the Hub loader.
    from ._triton import project_max

    scores = project_max(hidden, weight, bias, attention_mask)
    scores.relu_().log1p_()
    if double_log:
        scores.log1p_()
    return scores
