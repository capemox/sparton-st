import pytest
import torch
from sparton_st._kernel import sparton_forward
from torch.nn import functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="NVIDIA CUDA SM80+ required",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [7, 65])
@pytest.mark.parametrize("double_log", [False, True])
def test_matches_torch_with_strides_padding_and_tail_tiles(dtype, tokens, double_log):
    generator = torch.Generator(device="cuda").manual_seed(42)

    def random(*shape):
        return (
            torch.randn(*shape, generator=generator, device="cuda", dtype=dtype) * 0.16
        )

    hidden = random(3, 37, tokens).transpose(1, 2)
    weight = random(101, 74)[:, ::2]
    bias = random(202)[::2] if double_log else None
    attention_mask = torch.ones((3, tokens * 2), dtype=torch.int64, device="cuda")[
        :, ::2
    ]
    attention_mask[0, ::3] = 0
    attention_mask[1] = 0
    if tokens == 7:
        attention_mask = attention_mask.bool()

    actual = sparton_forward(
        hidden, weight, bias, attention_mask, double_log=double_log
    )
    # Contiguous operands select the fused-bias reference. A strided native
    # Linear can round the matmul to the output dtype before adding the bias.
    logits = F.linear(
        hidden.contiguous(),
        weight.contiguous(),
        None if bias is None else bias.contiguous(),
    )
    expected = (logits * attention_mask.unsqueeze(-1)).relu().log1p()
    if double_log:
        expected = expected.log1p()
    torch.testing.assert_close(actual, expected.amax(dim=1))
    assert actual[1].count_nonzero().item() == 0


def test_kernel_rejects_gradients():
    hidden = torch.zeros(1, 2, 3, device="cuda", requires_grad=True)
    weight = torch.zeros(4, 3, device="cuda")
    mask = torch.ones(1, 2, device="cuda", dtype=torch.bool)
    with pytest.raises(ValueError, match="inference only"):
        sparton_forward(hidden, weight, None, mask)
