"""Tiled vocabulary projection with a sequence max reduction in each tile."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _project_max(
    Hidden,
    Weight,
    Bias,
    Mask,
    Out,
    tokens,
    seq_tiles,
    vocab: tl.constexpr,
    width: tl.constexpr,
    hidden_batch_stride,
    hidden_token_stride: tl.constexpr,
    hidden_width_stride: tl.constexpr,
    weight_vocab_stride: tl.constexpr,
    weight_width_stride: tl.constexpr,
    bias_stride: tl.constexpr,
    mask_batch_stride,
    mask_token_stride: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Schedule sequence tiles for the same decoder tile together to reuse its weights in L2.
    sequence_tile = tl.program_id(0)
    vocab_tile = tl.program_id(1)
    batch_id = (sequence_tile // seq_tiles).to(tl.int64)
    token_ids = sequence_tile % seq_tiles * BLOCK_T + tl.arange(0, BLOCK_T)
    vocab_ids = vocab_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    hidden_ids = tl.arange(0, BLOCK_H)

    hidden_offsets = (
        batch_id * hidden_batch_stride
        + token_ids[:, None] * hidden_token_stride
        + hidden_ids[None, :] * hidden_width_stride
    )
    weight_offsets = (
        vocab_ids[None, :] * weight_vocab_stride
        + hidden_ids[:, None] * weight_width_stride
    )
    accum = tl.full((BLOCK_T, BLOCK_V), 0, tl.float32)
    for block in range(tl.cdiv(width, BLOCK_H)):
        valid_h = block * BLOCK_H + hidden_ids < width
        h = tl.load(
            Hidden + hidden_offsets + block * BLOCK_H * hidden_width_stride,
            (token_ids[:, None] < tokens) & valid_h[None, :],
            other=0,
        )
        w = tl.load(
            Weight + weight_offsets + block * BLOCK_H * weight_width_stride,
            valid_h[:, None] & (vocab_ids[None, :] < vocab),
            other=0,
        )
        accum = tl.dot(h, w, accum, input_precision="tf32x3")

    dtype: tl.constexpr = Out.dtype.element_ty
    if HAS_BIAS:
        bias = tl.load(Bias + vocab_ids * bias_stride, vocab_ids < vocab, other=0)
        accum += bias[None, :].to(tl.float32)
    # Round at the same points as Linear and attention-mask multiplication.
    mask = tl.load(
        Mask + batch_id * mask_batch_stride + token_ids * mask_token_stride,
        token_ids < tokens,
        other=0,
    )
    logits = accum.to(dtype).to(tl.float32) * mask[:, None].to(dtype).to(tl.float32)
    logits = tl.where(token_ids[:, None] < tokens, logits, float("-inf"))
    maxima = tl.reduce(logits, 0, _maximum)
    tl.store(
        Out + sequence_tile.to(tl.int64) * vocab + vocab_ids, maxima, vocab_ids < vocab
    )


def project_max(hidden, weight, bias, attention_mask):
    batch, tokens, width = hidden.shape
    vocab = weight.shape[0]
    # Smaller tiles limit register usage for the three tensor-core operations in tf32x3.
    full_precision = hidden.dtype == torch.float32
    block_t = 32 if full_precision or tokens <= 32 else 64
    block_v = 64
    block_h = 32 if full_precision else 64
    seq_tiles = triton.cdiv(tokens, block_t)
    vocab_tiles = triton.cdiv(vocab, block_v)
    partial = torch.empty(
        (batch, seq_tiles, vocab), dtype=hidden.dtype, device=hidden.device
    )

    with torch.cuda.device(hidden.device):
        _project_max[(batch * seq_tiles, vocab_tiles)](
            hidden,
            weight,
            bias if bias is not None else weight,
            attention_mask,
            partial,
            tokens,
            seq_tiles,
            vocab,
            width,
            *hidden.stride(),
            *weight.stride(),
            0 if bias is None else bias.stride(0),
            *attention_mask.stride(),
            HAS_BIAS=bias is not None,
            BLOCK_T=block_t,
            BLOCK_V=block_v,
            BLOCK_H=block_h,
        )
    if seq_tiles == 1:
        return partial.squeeze(1)
    return partial.amax(dim=1)
