"""Fused SPLADE inference for Sentence Transformers."""

from .integration import SparseEncoder, kernelize

__all__ = ["SparseEncoder", "kernelize"]
