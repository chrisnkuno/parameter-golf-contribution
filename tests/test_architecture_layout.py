from __future__ import annotations

import pytest

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
