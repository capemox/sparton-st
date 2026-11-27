import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sentence_transformers import SparseEncoder as NativeSparseEncoder
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    DistilBertConfig,
    ModernBertConfig,
    RobertaConfig,
    XLMRobertaConfig,
)

from sparton_st import SparseEncoder, integration, kernelize

TEXTS = ["sparse retrieval", "a"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sparse", "retrieval", "a"] + [
    f"unused{i}" for i in range(61)
]


@pytest.fixture(
    scope="module",
    params=["bert", "roberta", "xlm-roberta", "distilbert", "modernbert"],
)
def checkpoint(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_file)).save_pretrained(path)
    if request.param in {"distilbert", "modernbert"}:
        config_file = path / "tokenizer_config.json"
        tokenizer_config = json.loads(config_file.read_text())
        tokenizer_config["model_input_names"] = ["input_ids", "attention_mask"]
        config_file.write_text(json.dumps(tokenizer_config), encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(path)
        assert "token_type_ids" not in tokenizer(TEXTS)
    common = {"vocab_size": len(VOCAB), "pad_token_id": 0}
    if request.param == "distilbert":
        config = DistilBertConfig(
            dim=32, hidden_dim=64, n_layers=1, n_heads=4, **common
        )
    else:
        configs = {
            "bert": BertConfig,
            "roberta": RobertaConfig,
            "xlm-roberta": XLMRobertaConfig,
            "modernbert": ModernBertConfig,
        }
        config = configs[request.param](
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            **common,
        )
    if request.param == "modernbert":
        config.reference_compile = False
    with torch.random.fork_rng():
        torch.manual_seed(42)
        AutoModelForMaskedLM.from_config(config).save_pretrained(path)
    return path


def test_cpu_matches_and_native_checkpoint_roundtrips(checkpoint, tmp_path):
    model = NativeSparseEncoder(str(checkpoint), device="cpu")
    expected = model.encode(TEXTS, convert_to_tensor=True)
    assert kernelize(model) is model
    assert kernelize(model) is model
    torch.testing.assert_close(
        model.encode(TEXTS, convert_to_tensor=True).to_dense(), expected.to_dense()
    )
    with pytest.raises(ValueError, match="different kernel options"):
        kernelize(model, kernel_repo="example/sparton")

    model.save_pretrained(str(tmp_path), create_model_card=False)
    saved = json.loads((tmp_path / "modules.json").read_text())
    assert all(entry["type"].startswith("sentence_transformers.") for entry in saved)
    reloaded = NativeSparseEncoder(str(tmp_path), device="cpu")
    torch.testing.assert_close(
        reloaded.encode(TEXTS, convert_to_tensor=True).to_dense(), expected.to_dense()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_encode_and_forward_use_fused_head(checkpoint, dtype, monkeypatch):
    from sparton_st._kernel import sparton_forward

    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return sparton_forward(*args, **kwargs)

    monkeypatch.setattr(integration, "_load_kernel", lambda *_: counted)
    model = SparseEncoder(str(checkpoint), device="cuda").to(dtype=dtype)
    model[1].activation_function = "log1p_relu"
    expected = model.encode(TEXTS, convert_to_tensor=True)
    kernelize(model)
    actual = model.encode(TEXTS, convert_to_tensor=True)
    assert actual.layout == expected.layout
    assert calls == [(2, 4, 32)]
    torch.testing.assert_close(
        actual.to_dense(), expected.to_dense(), atol=1e-4, rtol=2e-3
    )

    model.train()
    inputs = model.tokenizer(TEXTS, return_tensors="pt", padding=True).to("cuda")
    scores = model(dict(inputs))["sentence_embedding"]
    scores.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert len(calls) == 1


def test_constructor_kernelizes(checkpoint):
    model = SparseEncoder(str(checkpoint), device="cpu", use_sparton=True)
    assert isinstance(model, NativeSparseEncoder)
    assert isinstance(model[1].forward, integration.SpartonKernel)
    get_dimension = (
        getattr(model, "get_embedding_dimension", None)
        or model.get_sentence_embedding_dimension
    )
    assert get_dimension() == len(VOCAB)


@pytest.mark.parametrize("revision", [None, "4be9a5f"])
def test_hub_loader_pins_the_kernel_and_checks_its_api(monkeypatch, revision):
    kernels = ModuleType("kernels")
    calls = []

    def forward(*args, **kwargs):
        return None

    def get_kernel(repo, **options):
        calls.append((repo, options))
        return SimpleNamespace(SPARTON_API_VERSION=1, sparton_forward=forward)

    monkeypatch.setitem(sys.modules, "kernels", kernels)
    kernels.get_kernel = get_kernel
    integration._load_kernel.cache_clear()
    try:
        assert integration._load_kernel("example/sparton", revision, False) is forward
        pin = {"version": 1} if revision is None else {"revision": revision}
        assert calls == [
            ("example/sparton", {"backend": "cuda", "trust_remote_code": False, **pin})
        ]

        kernels.get_kernel = lambda *args, **kwargs: SimpleNamespace(
            SPARTON_API_VERSION=2
        )
        with pytest.raises(ValueError, match="v1 API"):
            integration._load_kernel("example/wrong-api", revision, False)
    finally:
        integration._load_kernel.cache_clear()
