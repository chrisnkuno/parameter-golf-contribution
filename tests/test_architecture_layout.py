from __future__ import annotations

import pytest
import torch

import train_gpt as tg


def build_model(**overrides) -> tg.GPT:
    kwargs = dict(
        vocab_size=1024,
        num_layers=9,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=2,
        mlp_hidden=0,
        num_shared_blocks=0,
        num_untied_tail_blocks=0,
        local_mixer_prefix_layers=0,
        local_mixer_kernel_size=5,
        zeros_middle_layers=0,
        xsa_tail_layers=0,
        tie_embeddings=True,
        tied_embed_init_std=0.5,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.0,
    )
    kwargs.update(overrides)
    return tg.GPT(**kwargs)


def test_xsa_tail_marks_only_last_layers_without_sharing() -> None:
    model = build_model(xsa_tail_layers=4)
    use_xsa = [block.attn.use_xsa for block in model.logical_blocks]
    assert len(model.blocks) == 9
    assert use_xsa == [False, False, False, False, False, True, True, True, True]


def test_shared_prefix_reuses_blocks_and_limits_xsa_to_tail() -> None:
    model = build_model(num_shared_blocks=2, num_untied_tail_blocks=4, xsa_tail_layers=4)
    logical_to_registered = [next(i for i, rb in enumerate(model.blocks) if rb is lb) for lb in model.logical_blocks]
    use_xsa = [block.attn.use_xsa for block in model.logical_blocks]
    assert len(model.blocks) == 6
    assert len(model.logical_blocks) == 9
    assert logical_to_registered == [0, 1, 0, 1, 0, 2, 3, 4, 5]
    assert use_xsa == [False, False, False, False, False, True, True, True, True]


def test_shared_xsa_requires_enough_untied_tail_layers() -> None:
    with pytest.raises(ValueError, match="xsa_tail_layers must be <= num_untied_tail_blocks"):
        build_model(num_shared_blocks=2, num_untied_tail_blocks=3, xsa_tail_layers=4)


def test_local_mixer_marks_only_prefix_layers() -> None:
    model = build_model(local_mixer_prefix_layers=3, xsa_tail_layers=2)
    use_local = [getattr(block.attn, "use_local_mixer", False) for block in model.logical_blocks]
    use_zeros = [getattr(block.attn, "use_zeros", False) for block in model.logical_blocks]
    use_xsa = [getattr(block.attn, "use_xsa", False) for block in model.logical_blocks]
    assert use_local == [True, True, True, False, False, False, False, False, False]
    assert use_zeros == [False] * 9
    assert use_xsa == [False, False, False, False, False, False, False, True, True]


def test_zoned_layout_marks_middle_zeros_layers() -> None:
    model = build_model(local_mixer_prefix_layers=1, zeros_middle_layers=4, xsa_tail_layers=2)
    use_local = [getattr(block.attn, "use_local_mixer", False) for block in model.logical_blocks]
    use_zeros = [getattr(block.attn, "use_zeros", False) for block in model.logical_blocks]
    use_xsa = [getattr(block.attn, "use_xsa", False) for block in model.logical_blocks]
    assert use_local == [True, False, False, False, False, False, False, False, False]
    assert use_zeros == [False, True, True, True, True, False, False, False, False]
    assert use_xsa == [False, False, False, False, False, False, False, True, True]


def test_zone_counts_must_fit_within_num_layers() -> None:
    with pytest.raises(
        ValueError,
        match="local_mixer_prefix_layers, zeros_middle_layers, and xsa_tail_layers must fit within num_layers",
    ):
        build_model(local_mixer_prefix_layers=4, zeros_middle_layers=3, xsa_tail_layers=3)


def test_local_mixer_not_supported_with_shared_blocks() -> None:
    with pytest.raises(ValueError, match="local_mixer_prefix_layers is not supported with shared blocks yet"):
        build_model(num_shared_blocks=2, num_untied_tail_blocks=4, local_mixer_prefix_layers=2)


def test_zeros_not_supported_with_shared_blocks() -> None:
    with pytest.raises(ValueError, match="zeros_middle_layers is not supported with shared blocks yet"):
        build_model(num_shared_blocks=2, num_untied_tail_blocks=4, zeros_middle_layers=2)


def test_zeros_sm_weights_are_zero_sum_on_causal_prefix() -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 5, 5)
    gate_first = torch.sigmoid(torch.randn(2, 4, 5))
    gate_high = torch.sigmoid(torch.randn(2, 4, 5))
    weights = tg.compute_zeros_sm_weights(scores, gate_first, gate_high)
    causal = torch.tril(torch.ones((5, 5), dtype=torch.bool))
    masked = weights.masked_fill(~causal.view(1, 1, 5, 5), 0.0)
    row_sums = masked.sum(dim=-1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5, rtol=1e-5)


def test_zeros_sm_weights_vanish_for_uniform_logits() -> None:
    scores = torch.zeros(1, 2, 4, 4)
    gate_first = torch.full((1, 2, 4), 0.3)
    gate_high = torch.full((1, 2, 4), 0.7)
    weights = tg.compute_zeros_sm_weights(scores, gate_first, gate_high)
    assert torch.allclose(weights, torch.zeros_like(weights), atol=1e-6, rtol=1e-6)


def test_apply_xsa_transform_gate_limits() -> None:
    torch.manual_seed(0)
    attn_out = torch.randn(2, 4, 5, 8)
    value_heads = torch.randn(2, 2, 5, 8)

    almost_off = tg.apply_xsa_transform(
        attn_out,
        value_heads,
        num_heads=4,
        num_kv_heads=2,
        gate_logits=torch.full((4,), -20.0),
    )
    assert torch.allclose(almost_off, attn_out, atol=1e-6, rtol=1e-6)

    full_on = tg.apply_xsa_transform(
        attn_out,
        value_heads,
        num_heads=4,
        num_kv_heads=2,
        gate_logits=torch.full((4,), 20.0),
    )
    assert full_on.shape == attn_out.shape
    assert torch.isfinite(full_on).all()
    assert not torch.allclose(full_on, attn_out)
