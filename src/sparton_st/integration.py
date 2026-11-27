"""Apply the fused head while retaining Sentence Transformers' checkpoint format."""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import torch
from sentence_transformers import SparseEncoder as _SparseEncoder
from torch import nn
from transformers.activations import gelu

try:
    from sentence_transformers.sparse_encoder.modules import (
        MLMTransformer,
        SpladePooling,
    )
except ImportError:
    from sentence_transformers.sparse_encoder.models import (
        MLMTransformer,
        SpladePooling,
    )

try:
    from sentence_transformers.base.modules import Transformer
except ImportError:
    Transformer = None

_Model = TypeVar("_Model", bound=_SparseEncoder)
_HEADS = {
    "BertForMaskedLM": ("cls.predictions.transform",),
    "RobertaForMaskedLM": ("lm_head.dense", gelu, "lm_head.layer_norm"),
    "XLMRobertaForMaskedLM": ("lm_head.dense", gelu, "lm_head.layer_norm"),
    "DistilBertForMaskedLM": ("vocab_transform", "activation", "vocab_layer_norm"),
    "ModernBertForMaskedLM": ("head",),
}
_LEGACY_INPUTS = ("input_ids", "attention_mask", "token_type_ids", "inputs_embeds")
_MODEL_INPUTS = (*_LEGACY_INPUTS, "position_ids", "head_mask")
_NATIVE_FEATURES = {
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "cu_seqlens",
    "indices",
    "labels",
    "masked_tokens_mask",
    "logits_to_keep",
    "output_hidden_states",
    "output_attentions",
    "encoder_hidden_states",
    "encoder_attention_mask",
    "past_key_values",
}
_HIDDEN = "_sparton_hidden"
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MASK_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _cuda_supported(weight: torch.Tensor) -> bool:
    return (
        weight.device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(weight.device)[0] >= 8
    )


@lru_cache
def _load_kernel(repo_id: str | None, revision: str | None, trust_remote_code: bool):
    if repo_id is None:
        from ._kernel import sparton_forward

        return sparton_forward

    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise ImportError(
            "Install sparton-st[hub] to use a Hugging Face Hub kernel."
        ) from exc

    pin = {"revision": revision} if revision is not None else {"version": 1}
    module = get_kernel(
        repo_id, backend="cuda", trust_remote_code=trust_remote_code, **pin
    )
    forward = getattr(module, "sparton_forward", None)
    if getattr(module, "SPARTON_API_VERSION", None) != 1 or not callable(forward):
        raise ValueError(f"Kernel {repo_id!r} must expose the Sparton v1 API.")
    return forward


def _masked_lm(transformer):
    model = getattr(transformer, "model", None)
    if model is None:
        model = getattr(transformer, "auto_model", None)
    return model


class SpartonKernel:
    """Sentence Transformers pooling step backed by a local or Hub kernel."""

    def __init__(self, pooling, repo_id, revision, trust_remote_code):
        self.pooling = pooling
        self.options = (repo_id, revision, trust_remote_code)

    def __call__(self, features, **kwargs):
        projection = features.pop(_HIDDEN, None)
        if projection is None:
            return type(self.pooling).forward(self.pooling, features, **kwargs)

        hidden, decoder = projection
        with torch.cuda.device(hidden.device):
            forward = _load_kernel(*self.options)
            features["sentence_embedding"] = forward(
                hidden,
                decoder.weight,
                decoder.bias,
                features["attention_mask"],
                double_log=self.pooling.activation_function == "log1p_relu",
            )
        return features


class _TransformerForward:
    def __init__(self, transformer, pooling, head, legacy):
        self.transformer = transformer
        self.pooling = pooling
        self.head = head
        self.input_names = _LEGACY_INPUTS if legacy else _MODEL_INPUTS

    def __call__(self, features, **kwargs):
        transformer = self.transformer
        mlm = _masked_lm(transformer)
        decoder = mlm.get_output_embeddings()
        mask = features.get("attention_mask")
        if (
            transformer.training
            or self.pooling.training
            or torch.is_grad_enabled()
            or kwargs
            or _NATIVE_FEATURES.intersection(features)
            or getattr(mlm.config, "output_hidden_states", False)
            or getattr(mlm.config, "output_attentions", False)
            or getattr(mlm, "_hf_peft_config_loaded", False)
            or getattr(mlm, "hf_device_map", None)
            or getattr(mlm, "_hf_hook", None)
            or mlm._forward_hooks
            or mlm._forward_pre_hooks
            or self.pooling.pooling_strategy != "max"
            or type(decoder) is not nn.Linear
            or decoder._forward_hooks
            or decoder._forward_pre_hooks
            or decoder.weight.dtype not in _SUPPORTED_DTYPES
            or not isinstance(mask, torch.Tensor)
            or mask.ndim != 2
            or mask.numel() == 0
            or mask.dtype not in _MASK_DTYPES
            or mask.device != decoder.weight.device
            or not _cuda_supported(decoder.weight)
            or torch.is_autocast_enabled("cuda")
        ):
            return type(transformer).forward(transformer, features, **kwargs)

        backbone = mlm.base_model
        # DistilBERT and ModernBERT do not accept token_type_ids on every Transformers version.
        inputs = {name: features[name] for name in self.input_names if name in features}
        if type(mlm).__name__ in {"DistilBertForMaskedLM", "ModernBertForMaskedLM"}:
            inputs.pop("token_type_ids", None)
        hidden = backbone(**inputs, return_dict=True).last_hidden_state
        for step in self.head:
            transform = mlm.get_submodule(step) if isinstance(step, str) else step
            hidden = transform(hidden)

        if (
            hidden.shape[:2] != mask.shape
            or hidden.device != decoder.weight.device
            or hidden.dtype != decoder.weight.dtype
        ):
            return type(transformer).forward(transformer, features, **kwargs)
        features.pop("token_embeddings", None)
        features[_HIDDEN] = (hidden, decoder)
        return features


def kernelize(
    model: _Model,
    *,
    kernel_repo: str | None = None,
    kernel_revision: str | None = None,
    trust_kernel_code: bool = False,
) -> _Model:
    """Enable Sparton on a loaded SPLADE SparseEncoder in place and return it.

    Eligible CUDA inference uses the bundled kernel by default. A Hub kernel
    can implement the same v1 API. Other calls through a supported encoder
    use the original Sentence Transformers forward path.
    """
    if not isinstance(model, _SparseEncoder):
        raise TypeError("kernelize expects a Sentence Transformers SparseEncoder.")
    if kernel_repo is None and (kernel_revision is not None or trust_kernel_code):
        raise ValueError(
            "Specify kernel_repo when using kernel_revision or trust_kernel_code."
        )
    children = list(model.children())
    if len(children) != 2:
        raise ValueError(
            "Sparton expects a two-module SPLADE encoder (MLM Transformer + SpladePooling)."
        )

    transformer, pooling = children
    if type(pooling) is not SpladePooling or pooling.pooling_strategy != "max":
        raise ValueError("Sparton requires the built-in max SpladePooling module.")
    if type(transformer) not in (MLMTransformer, Transformer):
        raise ValueError(
            "Sparton requires a built-in fill-mask Transformer or legacy MLMTransformer."
        )
    task = getattr(transformer, "transformer_task", "fill-mask")
    if task != "fill-mask" or getattr(transformer, "backend", "torch") != "torch":
        raise ValueError("Sparton requires a PyTorch fill-mask Transformer.")
    modality = getattr(transformer, "modality_config", {}).get("text")
    if modality is not None:
        output = modality["method_output_name"]
        if (
            modality["method"] != "forward"
            or output not in ("logits", ("logits",), ["logits"])
            or transformer.module_output_name != "token_embeddings"
        ):
            raise ValueError(
                "Sparton requires the standard fill-mask Transformer output."
            )

    options = (kernel_repo, kernel_revision, trust_kernel_code)
    if isinstance(pooling.forward, SpartonKernel):
        if pooling.forward.options != options:
            raise ValueError(
                "This encoder already uses Sparton with different kernel options."
            )
        return model
    if "forward" in vars(transformer) or "forward" in vars(pooling):
        raise ValueError(
            "Sparton requires the original Transformer and SpladePooling forward methods."
        )

    mlm = _masked_lm(transformer)
    head = _HEADS.get(type(mlm).__name__)
    if head is None or not type(mlm).__module__.startswith("transformers.models."):
        raise ValueError(
            "Sparton supports standard BERT, RoBERTa, XLM-RoBERTa, DistilBERT and ModernBERT MLMs."
        )
    decoder = mlm.get_output_embeddings()
    if type(decoder) is not nn.Linear:
        raise ValueError("Sparton requires the standard linear MLM decoder.")

    # Keep the registered ST classes and parameter ownership so save_pretrained
    # writes a normal, portable SPLADE checkpoint without a custom code dependency.
    transformer.forward = _TransformerForward(
        transformer, pooling, head, legacy=not hasattr(transformer, "transformer_task")
    )
    pooling.forward = SpartonKernel(pooling, *options)
    for name in ("embedding_dimension", "word_embedding_dimension"):
        if hasattr(pooling, name) and getattr(pooling, name) is None:
            setattr(pooling, name, decoder.out_features)
    return model


class SparseEncoder(_SparseEncoder):
    """Sentence Transformers SparseEncoder with an opt-in Sparton constructor."""

    def __init__(
        self,
        *args,
        use_sparton: bool = False,
        kernel_repo: str | None = None,
        kernel_revision: str | None = None,
        trust_kernel_code: bool = False,
        **kwargs,
    ):
        if not use_sparton and (
            kernel_repo is not None or kernel_revision is not None or trust_kernel_code
        ):
            raise ValueError("Set use_sparton=True when configuring a Sparton kernel.")
        super().__init__(*args, **kwargs)
        if use_sparton:
            kernelize(
                self,
                kernel_repo=kernel_repo,
                kernel_revision=kernel_revision,
                trust_kernel_code=trust_kernel_code,
            )
