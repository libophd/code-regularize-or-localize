from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from jepa_lm.systems import KVCapture, attention_projections
from jepa_lm.trunks.llama import HFLlamaTrunk


def _tiny_trunk(vocab_size: int = 100) -> HFLlamaTrunk:
    return HFLlamaTrunk(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        vocab_size=vocab_size,
        max_position_embeddings=64,
    )


def test_capture_populates_all_layers_with_right_shape() -> None:
    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))
    with KVCapture(trunk) as cap:
        trunk(input_ids)
        k, v = cap.captured()
    assert len(k) == len(v) == 2
    assert cap.d_kv == 16
    for t in (*k, *v):
        assert t.shape == (2, 8, 16)
        assert torch.isfinite(t).all()


def test_captured_k_matches_direct_kproj() -> None:
    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))

    pairs = attention_projections(trunk)
    manual: dict[int, torch.Tensor] = {}
    handles = [
        pairs[i][0].register_forward_hook(
            lambda _m, _i, out, i=i: manual.__setitem__(i, out)
        )
        for i in range(len(pairs))
    ]
    try:
        with KVCapture(trunk) as cap:
            trunk(input_ids)
            k, _ = cap.captured()
    finally:
        for h in handles:
            h.remove()
    for i in range(len(pairs)):
        assert torch.equal(k[i], manual[i])


def test_capture_no_detach_grad_flows_under_grad() -> None:
    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))
    with KVCapture(trunk) as cap:
        trunk(input_ids)
        k, v = cap.captured()
    assert k[0].requires_grad and v[0].requires_grad
    loss = sum(t.pow(2).mean() for t in k)
    loss.backward()
    assert trunk.model.layers[0].self_attn.k_proj.weight.grad is not None


def test_capture_under_no_grad_yields_constants() -> None:
    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))
    with KVCapture(trunk) as cap, torch.no_grad():
        trunk(input_ids)
        k, v = cap.captured()
    assert not k[0].requires_grad and not v[0].requires_grad


def test_reset_inside_context_allows_recapture() -> None:
    torch.manual_seed(0)
    trunk = _tiny_trunk()
    a = torch.randint(0, 100, (1, 8))
    b = torch.randint(0, 100, (1, 8))
    with KVCapture(trunk) as cap:
        with torch.no_grad():
            trunk(a)
        k_first, _ = cap.captured()
        k_first0 = k_first[0].clone()
        cap.reset()
        with torch.no_grad():
            trunk(b)
        k_second, _ = cap.captured()
    assert not torch.equal(k_first0, k_second[0])


def test_hooks_removed_on_exit() -> None:
    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))
    with KVCapture(trunk) as cap:
        trunk(input_ids)
    cap.reset()
    trunk(input_ids)
    assert all(t is None for t in cap.k_per_layer)


def test_captured_before_forward_raises() -> None:
    trunk = _tiny_trunk()
    with KVCapture(trunk) as cap:
        with pytest.raises(RuntimeError):
            cap.captured()


def test_non_llama_module_rejected() -> None:
    with pytest.raises(TypeError):
        attention_projections(torch.nn.Linear(4, 4))


def test_capture_q_matches_direct_qproj() -> None:
    from jepa_lm.systems.kv_capture import query_projections

    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))
    q_projs = query_projections(trunk)
    manual: dict[int, torch.Tensor] = {}
    handles = [
        q_projs[i].register_forward_hook(lambda _m, _i, out, i=i: manual.__setitem__(i, out))
        for i in range(len(q_projs))
    ]
    try:
        with KVCapture(trunk, capture_q=True) as cap:
            trunk(input_ids)
            q = cap.captured_q()
    finally:
        for h in handles:
            h.remove()
    assert len(q) == 2
    for i in range(len(q_projs)):
        assert torch.equal(q[i], manual[i])


def test_captured_q_requires_capture_q_flag() -> None:
    trunk = _tiny_trunk()
    input_ids = torch.randint(0, 100, (2, 8))
    with KVCapture(trunk) as cap:
        trunk(input_ids)
        with pytest.raises(RuntimeError):
            cap.captured_q()
