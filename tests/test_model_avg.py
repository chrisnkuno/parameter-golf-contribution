from __future__ import annotations

import torch
from torch import nn

import train_gpt as tg


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.bias = nn.Parameter(torch.tensor([0.5, -0.5]))
        self.register_buffer("buf", torch.tensor([7.0]))


def test_init_model_avg_state_copies_named_parameters() -> None:
    module = TinyModule()
    avg_state = tg.init_model_avg_state(module)
    assert set(avg_state) == {"weight", "bias"}
    assert torch.allclose(avg_state["weight"], module.weight.float())
    assert torch.allclose(avg_state["bias"], module.bias.float())


def test_update_ema_state_blends_old_and_new_weights() -> None:
    module = TinyModule()
    avg_state = tg.init_model_avg_state(module)
    with torch.no_grad():
        module.weight.fill_(5.0)
        module.bias.fill_(1.0)
    tg.update_ema_state(avg_state, module, decay=0.5)
    assert torch.allclose(avg_state["weight"], torch.tensor([[3.0, 3.5], [4.0, 4.5]]))
    assert torch.allclose(avg_state["bias"], torch.tensor([0.75, 0.25]))


def test_update_swa_state_tracks_running_mean() -> None:
    module = TinyModule()
    avg_state = tg.init_model_avg_state(module)
    count = 0
    count = tg.update_swa_state(avg_state, module, count)
    with torch.no_grad():
        module.weight.fill_(5.0)
        module.bias.fill_(1.0)
    count = tg.update_swa_state(avg_state, module, count)
    assert count == 2
    assert torch.allclose(avg_state["weight"], torch.tensor([[3.0, 3.5], [4.0, 4.5]]))
    assert torch.allclose(avg_state["bias"], torch.tensor([0.75, 0.25]))


def test_materialize_averaged_state_dict_preserves_buffers_and_param_dtypes() -> None:
    module = TinyModule().bfloat16()
    avg_state = {
        "weight": torch.full((2, 2), 9.0, dtype=torch.float32),
        "bias": torch.full((2,), -3.0, dtype=torch.float32),
    }
    state = tg.materialize_averaged_state_dict(module, avg_state)
    assert state["weight"].dtype == torch.bfloat16
    assert state["bias"].dtype == torch.bfloat16
    assert torch.allclose(state["weight"].float(), torch.full((2, 2), 9.0))
    assert torch.allclose(state["bias"].float(), torch.full((2,), -3.0))
    assert torch.allclose(state["buf"].float(), torch.tensor([7.0]))
